"""
One-time backfill: re-read statement Excel files on Drive and update account_no +
transaction_fingerprint without touching match_* columns.

Must run on a machine with Google Drive mounted (HQ PC Task Scheduler), not cloud agents.
"""

from __future__ import annotations

import json
import os
import sys
from decimal import Decimal
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

from src.kcw.bank_statement import (
    compute_transaction_fingerprint,
    detect_bank_from_path,
    extract_account_from_file,
    is_short_account_no,
)
from src.kcw.paths import raw_dir, repo_root
from src.kcw.tar import supabase_db_url


def _load_env() -> None:
    for candidate in (repo_root() / ".env", Path.cwd() / ".env"):
        if candidate.is_file():
            load_dotenv(candidate)


def statement_dir() -> Path:
    return raw_dir() / "statement"


def resolve_file_path(source_path: str | None, original_filename: str) -> Path | None:
    candidates: list[Path] = []
    if source_path:
        candidates.append(Path(source_path))

    base = statement_dir()
    bank = detect_bank_from_path(original_filename) or detect_bank_from_path(
        str(source_path or "")
    )
    if bank:
        candidates.append(base / bank / original_filename)

    for folder in ("KBANK", "KTB"):
        candidates.append(base / folder / original_filename)

    for path in candidates:
        if path.is_file():
            return path
    return None


def connect_pg():
    return psycopg2.connect(supabase_db_url(), sslmode="require")


def fetch_import_files(conn) -> list[dict]:
    sql = """
    select id, original_filename, source_path, account_no, bank_name
    from bank.statement_import_files
  where account_no is null
     or length(trim(account_no)) <= 5
     or trim(account_no) ~ '^[0-9]{3,5}$'
    order by original_filename
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "original_filename": r[1],
            "source_path": r[2],
            "account_no": r[3],
            "bank_name": r[4],
        }
        for r in rows
    ]


def fetch_lines_for_file(conn, file_id: str) -> list[dict]:
    sql = """
    select id, account_no, txn_date, amount, direction, description,
           bank_reference, balance_after, transaction_fingerprint
    from bank.statement_lines
    where source_file_id = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (file_id,))
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "account_no": r[1],
            "txn_date": r[2],
            "amount": r[3],
            "direction": r[4],
            "description": r[5],
            "bank_reference": r[6],
            "balance_after": r[7],
            "transaction_fingerprint": r[8],
        }
        for r in rows
    ]


def fingerprint_exists(conn, fingerprint: str, exclude_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            select 1 from bank.statement_lines
            where transaction_fingerprint = %s and id <> %s
            limit 1
            """,
            (fingerprint, exclude_id),
        )
        return cur.fetchone() is not None


def backfill(apply: bool = False, verbose: bool = True) -> int:
    _load_env()

    stmt_root = statement_dir()
    if not stmt_root.is_dir():
        print(
            f"Statement folder not found: {stmt_root}\n"
            "Set KCW_DRIVE_ROOT or KCW_ANALYTICS_DATA_ROOT and run on the HQ PC "
            "with Google Drive mounted."
        )
        return 1

    conn = connect_pg()
    files = fetch_import_files(conn)
    print(f"Import files with short account_no: {len(files)}")
    print(f"Statement root: {stmt_root}")

    stats = {
        "files_seen": 0,
        "files_missing": 0,
        "files_no_full_account": 0,
        "files_updated": 0,
        "lines_updated": 0,
        "lines_skipped_collision": 0,
        "lines_already_full": 0,
    }

    try:
        for meta in files:
            stats["files_seen"] += 1
            file_id = meta["id"]
            filename = meta["original_filename"]
            old_account = (meta["account_no"] or "").strip()

            path = resolve_file_path(meta["source_path"], filename)
            if path is None:
                stats["files_missing"] += 1
                print(f"[missing] {filename} (not on Drive at {stmt_root})")
                continue

            full_account = extract_account_from_file(str(path)).strip()
            if not full_account:
                stats["files_no_full_account"] += 1
                print(f"[no account in xls] {filename}")
                continue

            if not is_short_account_no(old_account) and old_account == full_account:
                print(f"[skip file meta] {filename} already {full_account}")
                continue

            lines = fetch_lines_for_file(conn, file_id)
            file_line_updates = 0
            would_update = 0

            for line in lines:
                line_account = (line["account_no"] or "").strip()
                if not is_short_account_no(line_account) and line_account == full_account:
                    stats["lines_already_full"] += 1
                    continue

                new_fp = compute_transaction_fingerprint(
                    account_no=full_account,
                    txn_date=line["txn_date"],
                    amount=line["amount"],
                    direction=line["direction"],
                    description=line["description"],
                    bank_reference=line["bank_reference"],
                    balance_after=line["balance_after"],
                )

                if new_fp == line["transaction_fingerprint"] and line_account == full_account:
                    continue

                if fingerprint_exists(conn, new_fp, str(line["id"])):
                    stats["lines_skipped_collision"] += 1
                    print(
                        f"[collision] {filename} line {line['id']}: "
                        f"fingerprint already exists"
                    )
                    continue

                if verbose and apply:
                    print(
                        f"  line {line['id']}: {line_account!r} -> {full_account!r}"
                    )

                would_update += 1
                if apply:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            update bank.statement_lines
                            set account_no = %s, transaction_fingerprint = %s
                            where id = %s
                            """,
                            (full_account, new_fp, line["id"]),
                        )
                    file_line_updates += 1

            stats["lines_updated"] += file_line_updates if apply else would_update

            if apply and (
                file_line_updates > 0
                or is_short_account_no(old_account)
                and old_account != full_account
            ):
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        update bank.statement_import_files
                        set account_no = %s,
                            raw_metadata = coalesce(raw_metadata, '{}'::jsonb)
                              || %s::jsonb
                        where id = %s
                        """,
                        (
                            full_account,
                            json.dumps(
                                {
                                    "account_no": full_account,
                                    "account_backfill_source": str(path),
                                }
                            ),
                            file_id,
                        ),
                    )
                conn.commit()
                stats["files_updated"] += 1
                print(
                    f"[updated file] {filename}: {old_account!r} -> {full_account!r} "
                    f"({file_line_updates} lines)"
                )
            elif not apply:
                print(
                    f"[dry-run] {filename}: {old_account!r} -> {full_account!r} "
                    f"({would_update} lines would update)"
                )
            elif file_line_updates == 0 and old_account == full_account:
                print(f"[ok] {filename} metadata only ({full_account})")

    finally:
        conn.close()

    print("\n=== Summary ===")
    for key, val in stats.items():
        print(f"  {key}: {val}")
    if not apply:
        print("Dry run only. Re-run with --apply to write changes.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv or sys.argv[1:]
    apply = "--apply" in argv
    return backfill(apply=apply)


if __name__ == "__main__":
    raise SystemExit(main())
