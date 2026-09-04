"""HQ → SYP ICMAS product-master sync (dry-run by default).

Pushes catalog / price / description fields from HQ PARTS9 dbo.ICMAS to SYP.
Never overwrites branch inventory (QTY*), bins (LOCATION*), or related
branch-local fields. SYP-only BCODEs are left untouched.

Reports every run under logs/icmas_master_sync/<timestamp>/ and appends
runs/index.jsonl. Optionally mirrors to Drive KCW-Data/ops/icmas_master_sync/.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import urllib.parse
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
from sqlalchemy import text

from src.kcw import paths
from src.kcw.extract_parts9 import mssql_engine
from src.kcw.mssql_host import pick_mssql_server

# Identity / key — never UPDATE these as payload columns.
KEY_COL = "BCODE"
IDENTITY_COL = "ID"

# Branch-local: never overwrite on SYP; INSERT uses defaults below.
BRANCH_DENYLIST: frozenset[str] = frozenset(
    {
        IDENTITY_COL,
        "LOCATION1",
        "LOCATION2",
        "STOCKNO",
        "QTYBEG1",
        "QTYBEG2",
        "QTYBEGA",
        "QTYBEGB",
        "QTYBEGC",
        "QTYOH1",
        "QTYOH2",
        "QTYOHA",
        "QTYOHB",
        "QTYOHC",
        "QTYMIN",
        "QTYMAX",
        "QTYGET",
        "QTYPUT",
        "DATEAUDIT",
        "DATEUPDATE",
        "DATELAST",
        "SALELAST",
        "SALEDATE",
        "COSTAVG",
        "COSTLAST",
    }
)

# Explicit master allowlist (HQ → SYP). Anything not listed is left alone.
MASTER_ALLOWLIST: frozenset[str] = frozenset(
    {
        "JOURMODE",
        "XCODE",
        "MCODE",
        "PCODE",
        "ACODE",
        "DESCR",
        "MODEL",
        "BRAND",
        "OEM",
        "VENDOR",
        "MAIN",
        "SUB",
        "PART",
        "UI1",
        "UI2",
        "UI3",
        "UI4",
        "MTP2",
        "MTP3",
        "MTP4",
        "STATUS",
        "SERIAL",
        "MIX",
        "EXMPT",
        "ISVAT",
        "CODE1",
        "CODE2",
        "CODE3",
        "CODE4",
        "SIZE1",
        "SIZE2",
        "SIZE3",
        "PRICELIST",
        "DATELIST",
        "PRICE1",
        "PRICE2",
        "PRICE3",
        "PRICE4",
        "PRICE5",
        "MARKUP1",
        "MARKUP2",
        "MARKUP3",
        "MARKUP4",
        "MARKUP5",
        "PRICEM1",
        "PRICEM2",
        "PRICEM3",
        "PRICEM4",
        "PRICEM5",
        "PBDATE",
        "PEDATE",
        "PPRICE1",
        "PPRICE2",
        "PMTP2",
        "COSTSET1",
        "COSTSET2",
        "COSTSET3",
        "COSTSET4",
        "DISCNT",
        "DISCNT1",
        "DISCNT2",
        "DISCNT3",
        "DISCNT4",
        "COSTNET",
        "COSTBEG1",
        "COSTBEG2",
        "REMARKS",
        "CANCELED",
    }
)

# INSERT defaults for branch columns (do not copy HQ stock/bins).
BRANCH_INSERT_DEFAULTS: dict[str, Any] = {
    "LOCATION1": None,
    "LOCATION2": None,
    "STOCKNO": None,
    "QTYBEG1": 0.0,
    "QTYBEG2": 0.0,
    "QTYBEGA": 0.0,
    "QTYBEGB": 0.0,
    "QTYBEGC": 0.0,
    "QTYOH1": 0.0,
    "QTYOH2": 0.0,
    "QTYOHA": 0.0,
    "QTYOHB": 0.0,
    "QTYOHC": 0.0,
    "QTYMIN": 0.0,
    "QTYMAX": 0.0,
    "QTYGET": 0.0,
    "QTYPUT": 0,
    "DATEAUDIT": None,
    "DATEUPDATE": None,
    "DATELAST": None,
    "SALELAST": None,
    "SALEDATE": None,
    "COSTAVG": None,
    "COSTLAST": None,
}

APPLY_BATCH_SIZE = 200


@dataclass
class ColumnChange:
    bcode: str
    column: str
    old_syp: Any
    new_hq: Any


@dataclass
class SyncPlan:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    syp_only: list[str] = field(default_factory=list)
    column_changes: list[ColumnChange] = field(default_factory=list)
    # bcode -> {col: hq_value} for apply
    update_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    # bcode -> full insert row (no ID)
    insert_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class SyncResult:
    mode: str
    run_id: str
    report_dir: Path
    summary: dict[str, Any]
    plan: SyncPlan
    applied_added: int = 0
    applied_updated: int = 0
    errors: list[str] = field(default_factory=list)


def normalize_bcode(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def normalize_value(value: Any) -> Any:
    """Canonical form for equality checks across HQ/SYP."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Decimal):
        if value.is_nan():
            return None
        value = float(value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        try:
            ts = pd.Timestamp(value)
            if pd.isna(ts):
                return None
            return ts.to_pydatetime().replace(microsecond=0).isoformat(sep=" ")
        except Exception:
            return str(value)
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", errors="replace")
        except Exception:
            return str(value)
    if isinstance(value, str):
        s = value.strip()
        return s if s != "" else None
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float):
        if abs(value) < 1e-12:
            return 0.0
        return round(value, 6)
    # pandas / numpy scalars
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def sql_bind_value(value: Any) -> Any:
    """Coerce pandas/numpy nulls so pyodbc can bind (NaN/NaT → NULL)."""
    if value is None:
        return None
    try:
        # covers NaN, NaT, pd.NA
        if value is pd.NA or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Decimal) and value.is_nan():
        return None
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime().replace(microsecond=0)
    if isinstance(value, (datetime, date)):
        return value
    # numpy scalar → python
    try:
        import numpy as np

        if isinstance(value, np.generic):
            if np.isnan(value):
                return None
            return value.item()
    except Exception:
        pass
    if isinstance(value, str):
        return value
    return value


def sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: sql_bind_value(v) for k, v in row.items()}


def values_equal(a: Any, b: Any) -> bool:
    na, nb = normalize_value(a), normalize_value(b)
    if na is None and nb is None:
        return True
    if isinstance(na, float) and isinstance(nb, float):
        return math.isclose(na, nb, rel_tol=0, abs_tol=1e-6)
    # numeric cross-type (1 vs 1.0)
    if isinstance(na, (int, float)) and isinstance(nb, (int, float)):
        return math.isclose(float(na), float(nb), rel_tol=0, abs_tol=1e-6)
    return na == nb


def master_columns_present(columns: Iterable[str]) -> list[str]:
    cols = {str(c) for c in columns}
    return sorted(c for c in MASTER_ALLOWLIST if c in cols)


def sql_literal_for_report(value: Any) -> str:
    n = normalize_value(value)
    if n is None:
        return ""
    return str(n)


def load_icmas_frame(site: str, bcode_filter: Sequence[str] | None = None) -> pd.DataFrame:
    engine = mssql_engine(site)
    if bcode_filter:
        codes = [normalize_bcode(c) for c in bcode_filter if normalize_bcode(c)]
        if not codes:
            return pd.DataFrame()
        # Parameterized IN via temp values — keep batches small for filters.
        frames: list[pd.DataFrame] = []
        chunk = 400
        for i in range(0, len(codes), chunk):
            part = codes[i : i + chunk]
            placeholders = ", ".join(f":b{j}" for j in range(len(part)))
            params = {f"b{j}": part[j] for j in range(len(part))}
            q = text(
                f"""
                SELECT *
                FROM dbo.ICMAS WITH (NOLOCK)
                WHERE LTRIM(RTRIM(BCODE)) IN ({placeholders})
                """
            )
            with engine.connect() as conn:
                frames.append(pd.read_sql(q, conn, params=params))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    with engine.connect() as conn:
        return pd.read_sql(text("SELECT * FROM dbo.ICMAS WITH (NOLOCK)"), conn)


def frame_to_bcode_map(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if df.empty:
        return out
    records = df.to_dict(orient="records")
    for row in records:
        bcode = normalize_bcode(row.get(KEY_COL))
        if not bcode:
            continue
        # Last wins if duplicates (rare).
        out[bcode] = row
    return out


def build_sync_plan(
    hq_rows: dict[str, dict[str, Any]],
    syp_rows: dict[str, dict[str, Any]],
    *,
    master_cols: Sequence[str],
    limit: int | None = None,
) -> SyncPlan:
    plan = SyncPlan()
    hq_codes = sorted(hq_rows.keys())
    syp_codes = set(syp_rows.keys())

    considered = 0
    for bcode in hq_codes:
        if limit is not None and considered >= limit:
            break
        considered += 1
        hq = hq_rows[bcode]
        if bcode not in syp_codes:
            insert_row: dict[str, Any] = {KEY_COL: bcode}
            for col in master_cols:
                insert_row[col] = hq.get(col)
            for col, default in BRANCH_INSERT_DEFAULTS.items():
                insert_row[col] = default
            plan.added.append(bcode)
            plan.insert_payloads[bcode] = sanitize_row(insert_row)
            continue

        syp = syp_rows[bcode]
        changes: dict[str, Any] = {}
        for col in master_cols:
            hq_v = hq.get(col)
            syp_v = syp.get(col)
            if not values_equal(hq_v, syp_v):
                changes[col] = hq_v
                plan.column_changes.append(
                    ColumnChange(
                        bcode=bcode,
                        column=col,
                        old_syp=syp_v,
                        new_hq=hq_v,
                    )
                )
        if changes:
            plan.updated.append(bcode)
            plan.update_payloads[bcode] = sanitize_row(changes)
        else:
            plan.unchanged.append(bcode)

    # SYP-only: never touch; always list full set (not limited by --limit on HQ side).
    for bcode in sorted(syp_codes - set(hq_rows.keys())):
        plan.syp_only.append(bcode)

    return plan


def mssql_writer_engine(site: str = "syp"):
    """SQLAlchemy engine with writer credentials (apply / queue mark-done)."""
    from sqlalchemy import create_engine

    site = site.lower()
    if site not in ("hq", "syp"):
        raise ValueError("site must be 'hq' or 'syp'")

    if site == "syp":
        prefix = "PARTS9_SYP"
        default_server = os.getenv("KSS_PC_SERVER", "kss-pc")
    else:
        prefix = "PARTS9_HQ"
        default_server = os.getenv("KSS_SERVER", "KSS")

    server = os.getenv(f"{prefix}_SERVER") or default_server
    # HQ box often uses POS_MSSQL_* as the local/HQ server
    if site == "hq" and not os.getenv(f"{prefix}_SERVER"):
        server = os.getenv("POS_MSSQL_SERVER") or server
    server = pick_mssql_server(server)
    database = (
        os.getenv(f"{prefix}_DATABASE")
        or (os.getenv("POS_MSSQL_DATABASE") if site == "hq" else None)
        or "PARTS9"
    )
    user = (
        os.getenv(f"{prefix}_WRITER_USER")
        or os.getenv("POS_MSSQL_WRITER_USERNAME")
        or os.getenv(f"{prefix}_USER")
        or (os.getenv("POS_MSSQL_USERNAME") if site == "hq" else None)
    )
    password = (
        os.getenv(f"{prefix}_WRITER_PASSWORD")
        or os.getenv("POS_MSSQL_WRITER_PASSWORD")
        or os.getenv(f"{prefix}_PASSWORD")
        or (os.getenv("POS_MSSQL_PASSWORD") if site == "hq" else None)
    )
    driver = os.getenv("MSSQL_ODBC_DRIVER", "ODBC Driver 17 for SQL Server")
    if not user or not password:
        raise RuntimeError(
            f"{site.upper()} writer credentials missing. Set PARTS9_{site.upper()}_WRITER_USER/"
            "PASSWORD (or POS_MSSQL_WRITER_USERNAME/PASSWORD)."
        )
    odbc_str = (
        f"DRIVER={{{driver}}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={user};"
        f"PWD={password};"
        "TrustServerCertificate=yes;"
    )
    return create_engine("mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(odbc_str))



def _quote_ident(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"unsafe column name: {name!r}")
    return f"[{name}]"


def apply_plan(plan: SyncPlan, *, master_cols: Sequence[str]) -> tuple[int, int, list[str]]:
    """INSERT new + UPDATE changed master columns on SYP. Returns (added, updated, errors)."""
    errors: list[str] = []
    applied_added = 0
    applied_updated = 0
    engine = mssql_writer_engine("syp")

    # INSERT
    insert_items = list(plan.insert_payloads.items())
    insert_denied = False
    for i in range(0, len(insert_items), APPLY_BATCH_SIZE):
        if insert_denied:
            break
        batch = insert_items[i : i + APPLY_BATCH_SIZE]
        try:
            with engine.begin() as conn:
                for bcode, row in batch:
                    cols = [KEY_COL] + [c for c in master_cols if c in row] + [
                        c for c in BRANCH_INSERT_DEFAULTS if c in row
                    ]
                    # Dedupe while preserving order
                    seen: set[str] = set()
                    ordered: list[str] = []
                    for c in cols:
                        if c not in seen:
                            seen.add(c)
                            ordered.append(c)
                    col_sql = ", ".join(_quote_ident(c) for c in ordered)
                    param_sql = ", ".join(f":{c}" for c in ordered)
                    params = {c: sql_bind_value(row.get(c)) for c in ordered}
                    conn.execute(
                        text(
                            f"INSERT INTO dbo.ICMAS ({col_sql}) VALUES ({param_sql})"
                        ),
                        params,
                    )
                    applied_added += 1
        except Exception as exc:
            msg = str(exc)
            if "INSERT permission was denied" in msg or "permission was denied" in msg.lower():
                errors.append(
                    "INSERT skipped: permission denied on dbo.ICMAS for writer login "
                    f"({exc.__class__.__name__}). Grant INSERT to finish new SKUs. "
                    f"Remaining inserts not attempted ({len(insert_items) - i})."
                )
                insert_denied = True
                break
            # Whole batch rolled back — retry row-by-row
            for bcode, row in batch:
                try:
                    with engine.begin() as conn:
                        cols = [KEY_COL] + [c for c in master_cols if c in row] + [
                            c for c in BRANCH_INSERT_DEFAULTS if c in row
                        ]
                        seen: set[str] = set()
                        ordered: list[str] = []
                        for c in cols:
                            if c not in seen:
                                seen.add(c)
                                ordered.append(c)
                        col_sql = ", ".join(_quote_ident(c) for c in ordered)
                        param_sql = ", ".join(f":{c}" for c in ordered)
                        params = {c: sql_bind_value(row.get(c)) for c in ordered}
                        conn.execute(
                            text(
                                f"INSERT INTO dbo.ICMAS ({col_sql}) VALUES ({param_sql})"
                            ),
                            params,
                        )
                        applied_added += 1
                except Exception as row_exc:
                    row_msg = str(row_exc)
                    if "INSERT permission was denied" in row_msg:
                        errors.append(
                            "INSERT skipped: permission denied on dbo.ICMAS. "
                            f"Remaining inserts not attempted ({len(insert_items) - i})."
                        )
                        insert_denied = True
                        break
                    errors.append(
                        f"INSERT {bcode}: {type(row_exc).__name__}: {row_exc}"
                    )

    # UPDATE
    update_items = list(plan.update_payloads.items())
    for i in range(0, len(update_items), APPLY_BATCH_SIZE):
        batch = update_items[i : i + APPLY_BATCH_SIZE]
        try:
            with engine.begin() as conn:
                for bcode, changes in batch:
                    if not changes:
                        continue
                    # Refuse any branch column slipping through
                    bad = set(changes) & BRANCH_DENYLIST
                    if bad:
                        errors.append(f"UPDATE {bcode}: refused branch cols {sorted(bad)}")
                        continue
                    sets = ", ".join(f"{_quote_ident(c)} = :{c}" for c in changes)
                    params = {c: sql_bind_value(v) for c, v in changes.items()}
                    params["bcode"] = bcode
                    result = conn.execute(
                        text(
                            f"""
                            UPDATE dbo.ICMAS
                            SET {sets}
                            WHERE LTRIM(RTRIM(BCODE)) = :bcode
                            """
                        ),
                        params,
                    )
                    if result.rowcount == 0:
                        errors.append(f"UPDATE {bcode}: 0 rows matched")
                    else:
                        applied_updated += 1
        except Exception:
            for bcode, changes in batch:
                if not changes:
                    continue
                bad = set(changes) & BRANCH_DENYLIST
                if bad:
                    errors.append(f"UPDATE {bcode}: refused branch cols {sorted(bad)}")
                    continue
                try:
                    with engine.begin() as conn:
                        sets = ", ".join(f"{_quote_ident(c)} = :{c}" for c in changes)
                        params = {c: sql_bind_value(v) for c, v in changes.items()}
                        params["bcode"] = bcode
                        result = conn.execute(
                            text(
                                f"""
                                UPDATE dbo.ICMAS
                                SET {sets}
                                WHERE LTRIM(RTRIM(BCODE)) = :bcode
                                """
                            ),
                            params,
                        )
                        if result.rowcount == 0:
                            errors.append(f"UPDATE {bcode}: 0 rows matched")
                        else:
                            applied_updated += 1
                except Exception as row_exc:
                    errors.append(
                        f"UPDATE {bcode}: {type(row_exc).__name__}: {row_exc}"
                    )

    return applied_added, applied_updated, errors


def report_root() -> Path:
    return paths.ensure_dir(paths.log_dir() / "icmas_master_sync")


def drive_report_root() -> Path | None:
    try:
        root = paths.drive_root() / "KCW-Data" / "ops" / "icmas_master_sync"
        return paths.ensure_dir(root)
    except Exception:
        return None


def write_report(
    *,
    mode: str,
    plan: SyncPlan,
    run_id: str,
    hq_count: int,
    syp_count: int,
    master_cols: Sequence[str],
    applied_added: int = 0,
    applied_updated: int = 0,
    errors: Sequence[str] = (),
    duration_s: float = 0.0,
    mirror_drive: bool = True,
    limit: int | None = None,
    bcode_filter: Sequence[str] | None = None,
) -> tuple[Path, dict[str, Any]]:
    root = report_root()
    runs_dir = paths.ensure_dir(root / "runs")
    out = paths.ensure_dir(root / run_id)

    summary: dict[str, Any] = {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": run_id,
        "mode": mode,
        "applied": mode == "apply",
        "hq_rows": hq_count,
        "syp_rows": syp_count,
        "added": len(plan.added),
        "updated": len(plan.updated),
        "unchanged": len(plan.unchanged),
        "syp_only": len(plan.syp_only),
        "column_changes": len(plan.column_changes),
        "applied_added": applied_added,
        "applied_updated": applied_updated,
        "errors": len(errors),
        "error_messages": list(errors),
        "duration_s": round(duration_s, 2),
        "master_columns": list(master_cols),
        "branch_denylist": sorted(BRANCH_DENYLIST),
        "limit": limit,
        "bcode_filter": list(bcode_filter) if bcode_filter else None,
        "path": run_id,
    }

    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (out / "added.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["BCODE"])
        for b in plan.added:
            w.writerow([b])

    with (out / "updated.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["BCODE", "column", "old_syp", "new_hq"])
        for ch in plan.column_changes:
            w.writerow(
                [
                    ch.bcode,
                    ch.column,
                    sql_literal_for_report(ch.old_syp),
                    sql_literal_for_report(ch.new_hq),
                ]
            )

    with (out / "syp_only.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["BCODE"])
        for b in plan.syp_only:
            w.writerow([b])

    if errors:
        (out / "errors.csv").write_text(
            "error\n" + "\n".join(csv_escape(e) for e in errors) + "\n",
            encoding="utf-8",
        )

    # Human summary
    top_changes = plan.column_changes[:40]
    lines = [
        f"# ICMAS master sync report `{run_id}`",
        "",
        f"- mode: **{mode}**",
        f"- HQ rows: {hq_count} · SYP rows: {syp_count}",
        f"- added: {len(plan.added)} · updated SKUs: {len(plan.updated)} · "
        f"column diffs: {len(plan.column_changes)} · unchanged: {len(plan.unchanged)} · "
        f"syp_only: {len(plan.syp_only)}",
        f"- applied_added: {applied_added} · applied_updated: {applied_updated} · errors: {len(errors)}",
        f"- duration_s: {summary['duration_s']}",
        "",
        "## Branch fields never overwritten",
        "",
        ", ".join(sorted(BRANCH_DENYLIST)),
        "",
        "## Sample column diffs (first 40)",
        "",
        "| BCODE | column | old_syp | new_hq |",
        "| --- | --- | --- | --- |",
    ]
    for ch in top_changes:
        lines.append(
            f"| {ch.bcode} | {ch.column} | "
            f"{_md_cell(sql_literal_for_report(ch.old_syp))} | "
            f"{_md_cell(sql_literal_for_report(ch.new_hq))} |"
        )
    if not top_changes:
        lines.append("| — | — | — | — |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # index.jsonl
    index_path = runs_dir / "index.jsonl"
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    # latest/ convenience copy
    latest = root / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)
    shutil.copytree(out, latest)

    if mirror_drive:
        drive_root = drive_report_root()
        if drive_root is not None:
            dest = drive_root / run_id
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(out, dest)
            # Keep a Drive copy of the index too
            try:
                shutil.copy2(index_path, drive_root / "index.jsonl")
            except Exception:
                pass
            summary["drive_path"] = str(dest)
        else:
            summary["drive_path"] = None
            summary["drive_mirror_error"] = "Drive root unavailable"

    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return out, summary


def csv_escape(s: str) -> str:
    if any(c in s for c in '",\n'):
        return '"' + s.replace('"', '""') + '"'
    return s


def _md_cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")[:120]


def run_sync(
    *,
    apply: bool = False,
    bcode_filter: Sequence[str] | None = None,
    limit: int | None = None,
    mirror_drive: bool = True,
) -> SyncResult:
    import time

    t0 = time.perf_counter()
    mode = "apply" if apply else "dry_run"
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    print(f"[icmas-master] mode={mode} run_id={run_id}")
    print("[icmas-master] loading HQ ICMAS…")
    hq_df = load_icmas_frame("hq", bcode_filter=bcode_filter)
    print(f"[icmas-master] HQ rows={len(hq_df)}")
    print("[icmas-master] loading SYP ICMAS…")
    syp_df = load_icmas_frame("syp", bcode_filter=bcode_filter)
    print(f"[icmas-master] SYP rows={len(syp_df)}")

    cols = set(hq_df.columns) | set(syp_df.columns)
    master_cols = master_columns_present(cols)
    denied_present = sorted(c for c in BRANCH_DENYLIST if c in cols)
    print(
        f"[icmas-master] master_cols={len(master_cols)} "
        f"branch_denied_present={len(denied_present)}"
    )

    hq_map = frame_to_bcode_map(hq_df)
    syp_map = frame_to_bcode_map(syp_df)
    plan = build_sync_plan(hq_map, syp_map, master_cols=master_cols, limit=limit)
    print(
        f"[icmas-master] plan added={len(plan.added)} updated={len(plan.updated)} "
        f"unchanged={len(plan.unchanged)} syp_only={len(plan.syp_only)} "
        f"col_diffs={len(plan.column_changes)}"
    )

    applied_added = 0
    applied_updated = 0
    errors: list[str] = []
    if apply:
        print("[icmas-master] applying to SYP…")
        applied_added, applied_updated, errors = apply_plan(plan, master_cols=master_cols)
        print(
            f"[icmas-master] applied added={applied_added} updated={applied_updated} "
            f"errors={len(errors)}"
        )

    duration_s = time.perf_counter() - t0
    report_dir, summary = write_report(
        mode=mode,
        plan=plan,
        run_id=run_id,
        hq_count=len(hq_map),
        syp_count=len(syp_map),
        master_cols=master_cols,
        applied_added=applied_added,
        applied_updated=applied_updated,
        errors=errors,
        duration_s=duration_s,
        mirror_drive=mirror_drive,
        limit=limit,
        bcode_filter=bcode_filter,
    )
    print(f"[icmas-master] report={report_dir}")
    return SyncResult(
        mode=mode,
        run_id=run_id,
        report_dir=report_dir,
        summary=summary,
        plan=plan,
        applied_added=applied_added,
        applied_updated=applied_updated,
        errors=list(errors),
    )


QUEUE_TABLE = "dbo.ICMAS_MASTER_SYNC_QUEUE"


@dataclass
class QueueDrainResult:
    pending_claimed: int
    bcodes: list[str]
    applied_added: int
    applied_updated: int
    marked_done: int
    marked_error: int
    errors: list[str]
    sync: SyncResult | None = None


def fetch_pending_queue(
    *,
    limit: int = 500,
    hq_engine=None,
) -> list[dict[str, Any]]:
    """Claim pending queue rows from HQ (read-only select; mark later)."""
    eng = hq_engine or mssql_writer_engine("hq")
    with eng.connect() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT TOP (:lim) queue_id, bcode, event_type, queued_at, status
                FROM {QUEUE_TABLE} WITH (READPAST)
                WHERE status = N'pending'
                ORDER BY queue_id
                """
            ),
            {"lim": int(limit)},
        ).mappings().all()
    return [dict(r) for r in rows]


def mark_queue_rows(
    queue_ids: Sequence[int],
    *,
    status: str,
    error_msg: str | None = None,
    hq_engine=None,
) -> int:
    if not queue_ids:
        return 0
    eng = hq_engine or mssql_writer_engine("hq")
    ids = [int(i) for i in queue_ids]
    marked = 0
    # Parameterized IN batches
    chunk = 200
    with eng.begin() as conn:
        for i in range(0, len(ids), chunk):
            part = ids[i : i + chunk]
            placeholders = ", ".join(f":id{j}" for j in range(len(part)))
            params: dict[str, Any] = {f"id{j}": part[j] for j in range(len(part))}
            params["status"] = status
            params["err"] = (error_msg or "")[:1000] if error_msg else None
            result = conn.execute(
                text(
                    f"""
                    UPDATE {QUEUE_TABLE}
                    SET status = :status,
                        processed_at = SYSUTCDATETIME(),
                        error_msg = :err
                    WHERE queue_id IN ({placeholders})
                      AND status = N'pending'
                    """
                ),
                params,
            )
            marked += result.rowcount or 0
    return marked


def drain_queue(
    *,
    limit: int = 500,
    mirror_drive: bool = False,
) -> QueueDrainResult:
    """
    Poll HQ ICMAS_MASTER_SYNC_QUEUE and push claimed BCODEs to SYP.

    Requires queue table + trigger installed on KSS (scripts/sql/icmas_master_sync_queue.sql).
    """
    print(f"[icmas-queue] fetching pending (limit={limit})…")
    try:
        pending = fetch_pending_queue(limit=limit)
    except Exception as exc:
        msg = str(exc)
        soft = "Invalid object name" in msg or "ICMAS_MASTER_SYNC_QUEUE" in msg
        print(f"[icmas-queue] ERROR queue fetch failed: {type(exc).__name__}: {exc}")
        if soft:
            print(
                "[icmas-queue] HINT: install scripts/sql/icmas_master_sync_queue.sql "
                "on KSS (PARTS9) as admin, then re-run."
            )
        return QueueDrainResult(
            pending_claimed=0,
            bcodes=[],
            applied_added=0,
            applied_updated=0,
            marked_done=0,
            marked_error=0,
            # Soft-fail missing table so the 2-min timer does not alarm until DDL is applied
            errors=[] if soft else [f"queue fetch failed: {type(exc).__name__}: {exc}"],
        )

    if not pending:
        print("[icmas-queue] empty")
        return QueueDrainResult(
            pending_claimed=0,
            bcodes=[],
            applied_added=0,
            applied_updated=0,
            marked_done=0,
            marked_error=0,
            errors=[],
        )

    queue_ids = [int(r["queue_id"]) for r in pending]
    bcodes = sorted({normalize_bcode(r["bcode"]) for r in pending if normalize_bcode(r["bcode"])})
    print(f"[icmas-queue] claimed rows={len(queue_ids)} distinct_bcodes={len(bcodes)}")

    sync = run_sync(
        apply=True,
        bcode_filter=bcodes,
        mirror_drive=mirror_drive,
    )
    errors = list(sync.errors)
    if errors:
        marked_error = mark_queue_rows(
            queue_ids,
            status="error",
            error_msg="; ".join(errors)[:1000],
        )
        print(f"[icmas-queue] marked error={marked_error}")
        return QueueDrainResult(
            pending_claimed=len(queue_ids),
            bcodes=bcodes,
            applied_added=sync.applied_added,
            applied_updated=sync.applied_updated,
            marked_done=0,
            marked_error=marked_error,
            errors=errors,
            sync=sync,
        )

    marked_done = mark_queue_rows(queue_ids, status="done")
    print(f"[icmas-queue] marked done={marked_done}")
    return QueueDrainResult(
        pending_claimed=len(queue_ids),
        bcodes=bcodes,
        applied_added=sync.applied_added,
        applied_updated=sync.applied_updated,
        marked_done=marked_done,
        marked_error=0,
        errors=[],
        sync=sync,
    )

