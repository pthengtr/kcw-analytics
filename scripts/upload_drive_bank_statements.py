"""Upload Drive bank statement Excel files to the kcw-v2 Edge Function.

Replaces local parse/insert in notebooks/02_bank_statement_import_test.ipynb.
The sole parser is production Edge Function ``import-bank-statement`` (auto_v2).

Auth: SUPABASE_SERVICE_ROLE_KEY (HQ automation). Web UI continues to use user JWT + RBAC.

Env (from repo .env / process):
  SUPABASE_URL
  SUPABASE_SERVICE_ROLE_KEY
  BANK_STATEMENT_BASE_DIR  (optional; default <analytics>/01_raw/statement via KCW_DRIVE_ROOT)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
BANK_FOLDERS = ("KBANK", "KTB")
STATEMENT_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}


def _safe_print(msg: str = "", *, flush: bool = False) -> None:
    """Print without crashing on Windows cp874 / non-UTF8 consoles."""
    try:
        print(msg, flush=flush)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(msg.encode(enc, errors="replace").decode(enc, errors="replace"), flush=flush)


def load_env() -> None:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv()


def list_statement_files(base_dir: Path) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for bank in BANK_FOLDERS:
        folder = base_dir / bank
        if not folder.is_dir():
            _safe_print(f"WARNING: folder not found: {folder}")
            continue
        for path in sorted(folder.iterdir()):
            if path.name.startswith("~$"):
                continue
            if path.suffix.lower() not in STATEMENT_EXTENSIONS:
                continue
            if not path.is_file():
                continue
            out.append((bank, path))
    return out


def upload_one(
    *,
    session: requests.Session,
    url: str,
    service_key: str,
    bank_name: str,
    path: Path,
    timeout: float,
) -> dict:
    with path.open("rb") as fh:
        files = {
            "file": (
                path.name,
                fh,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        }
        data = {"bank_name": bank_name}
        headers = {
            "Authorization": f"Bearer {service_key}",
            "apikey": service_key,
        }
        resp = session.post(
            url,
            headers=headers,
            files=files,
            data=data,
            timeout=timeout,
        )
    try:
        body = resp.json()
    except Exception:
        body = {"error": resp.text[:500]}
    if not isinstance(body, dict):
        body = {"error": str(body)}
    body["_http_status"] = resp.status_code
    body["_path"] = str(path)
    body["_bank"] = bank_name
    return body


def main(argv: list[str] | None = None) -> int:
    # HQ Windows workers often use cp874; avoid UnicodeEncodeError on arrows/Thai.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    load_env()
    from src.kcw.paths import raw_dir

    default_base = Path(os.getenv("BANK_STATEMENT_BASE_DIR") or (raw_dir() / "statement"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=default_base,
        help="Drive folder containing KBANK/ and KTB/ subfolders",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files only; do not upload",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Per-file HTTP timeout seconds",
    )
    args = parser.parse_args(argv)

    supabase_url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or ""
    if not args.dry_run and (not supabase_url or not service_key):
        _safe_print("ERROR: Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env")
        return 2

    files = list_statement_files(args.base_dir)
    _safe_print(f"base_dir={args.base_dir}")
    _safe_print(f"files={len(files)}")
    if args.dry_run:
        for bank, path in files:
            _safe_print(f"  {bank}\t{path.name}")
        return 0

    fn_url = f"{supabase_url}/functions/v1/import-bank-statement"
    counts = {"imported": 0, "skipped": 0, "failed": 0, "other": 0}
    session = requests.Session()

    for bank, path in files:
        _safe_print(f"> {bank} {path.name} ...", flush=True)
        try:
            body = upload_one(
                session=session,
                url=fn_url,
                service_key=service_key,
                bank_name=bank,
                path=path,
                timeout=args.timeout,
            )
        except Exception as exc:
            counts["failed"] += 1
            _safe_print(f"  FAILED transport: {exc}")
            continue

        status = str(body.get("status") or "")
        http = body.get("_http_status")
        if http and int(http) >= 400:
            counts["failed"] += 1
            _safe_print(
                f"  FAILED http={http}: {json.dumps(body, ensure_ascii=True)[:400]}"
            )
            continue

        if status in counts:
            counts[status] += 1
        else:
            counts["other"] += 1

        inserted = body.get("inserted_count")
        dup = body.get("duplicate_count")
        rows = body.get("row_count")
        _safe_print(
            f"  {status} rows={rows} inserted={inserted} duplicates={dup} "
            f"account={body.get('account_no')}"
        )

    _safe_print("--- summary ---")
    _safe_print(json.dumps(counts, indent=2, ensure_ascii=True))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
