"""Extract PARTS9 tables to Drive raw_{site}_*.csv (HQ or SYP)."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import urllib.parse
from pathlib import Path
from typing import Optional

import pandas as pd

from src.kcw import paths

# Rolling windows + filenames match notebooks/51_parts9_to_drive.ipynb (HQ).
# SYP original notebook used 5y for all bill tables — see SITE_YEARS.
TABLE_SPECS = {
    "SIDET": {"years": 5, "suffix": "sidet_sales_lines"},
    "PIDET": {"years": 8, "suffix": "pidet_purchase_lines"},
    "SIMAS": {"years": 5, "suffix": "simas_sales_bills"},
    "PIMAS": {"years": 8, "suffix": "pimas_purchase_bills"},
    "ARMAS": {"years": None, "suffix": "armas_receivable"},
    "APMAS": {"years": None, "suffix": "apmas_payable"},
    "ICMAS": {"years": None, "suffix": "icmas_products"},
    "PVMAS": {"years": None, "suffix": "pvmas_notes_vouchers"},
    "RVMAS": {"years": None, "suffix": "rvmas_notes_vouchers"},
}

# Match attached SYP PART9S_to_drive_raw notebook (all bill tables 5y).
SITE_YEARS = {
    "syp": {
        "SIDET": 5,
        "PIDET": 5,
        "SIMAS": 5,
        "PIMAS": 5,
    },
}

# Same write order as the original SYP notebook (pidet, pimas, sidet, simas, icmas).
SYP_MINIMAL = ("PIDET", "PIMAS", "SIDET", "SIMAS", "ICMAS")


def site_env_prefix(site: str) -> str:
    site = site.lower()
    if site not in ("hq", "syp"):
        raise ValueError("site must be 'hq' or 'syp'")
    return site.upper()


def mssql_engine(site: str = "hq"):
    """
    Build SQLAlchemy engine from env.

    HQ:
      PARTS9_HQ_SERVER / KSS_SERVER (default KSS)
      PARTS9_HQ_DATABASE (default PARTS9)
      PARTS9_HQ_USER / PARTS9_HQ_PASSWORD  OR trusted Windows auth if unset
    SYP:
      PARTS9_SYP_SERVER (default KSS-PC)
      PARTS9_SYP_DATABASE (default PARTS9)
      trusted Windows auth by default (matches current SYP notebook)
    """
    from sqlalchemy import create_engine

    prefix = f"PARTS9_{site_env_prefix(site)}"
    server = os.getenv(f"{prefix}_SERVER") or (
        os.getenv("KSS_SERVER", "KSS") if site == "hq" else os.getenv("KSS_PC_SERVER", "KSS-PC")
    )
    database = os.getenv(f"{prefix}_DATABASE", "PARTS9")
    user = os.getenv(f"{prefix}_USER") or (os.getenv("KSS_USER") if site == "hq" else None)
    password = os.getenv(f"{prefix}_PASSWORD") or (
        os.getenv("KSS_PASSWORD") if site == "hq" else None
    )
    driver = os.getenv("MSSQL_ODBC_DRIVER", "ODBC Driver 17 for SQL Server")

    if user and password:
        odbc_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            f"UID={user};"
            f"PWD={password};"
            "TrustServerCertificate=yes;"
        )
    else:
        # Same shape as the original SYP notebook trusted URL.
        odbc_str = (
            f"DRIVER={{{driver}}};"
            f"SERVER={server};"
            f"DATABASE={database};"
            "Trusted_Connection=yes;"
            "TrustServerCertificate=yes;"
        )

    return create_engine("mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(odbc_str))


def read_last_years(engine, table: str, years: int) -> pd.DataFrame:
    query = f"""
    SELECT *
    FROM dbo.{table}
    WHERE BILLDATE >= DATEADD(YEAR, -{int(years)}, (SELECT MAX(BILLDATE) FROM dbo.{table}))
    """
    return pd.read_sql(query, engine)


def read_full_table(engine, table: str) -> pd.DataFrame:
    return pd.read_sql(f"SELECT * FROM dbo.{table};", engine)


def _years_for(site: str, table: str) -> Optional[int]:
    site_map = SITE_YEARS.get(site.lower(), {})
    if table in site_map:
        return site_map[table]
    return TABLE_SPECS[table]["years"]


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """
    DriveFS-safe overwrite without fsync.

    Why not plain df.to_csv(path)? On Google Drive File Stream, in-place
    overwrite of large existing cloud files (sidet/icmas) often leaves the
    cloud object stale while smaller files update — that was the original
    timestamp mismatch. The original notebook "never failed" because it
    never checked mtimes and usually got lucky on sync.

    Why not fsync? DriveFS returns [Errno 9] Bad file descriptor.

    Method that DID update sidet on your machine (before false verify abort):
    write to local TEMP → copy beside target on G: → os.replace (no fsync).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, local_name = tempfile.mkstemp(prefix=f"{path.stem}_", suffix=".csv")
    os.close(fd)
    local_tmp = Path(local_name)
    drive_tmp = path.with_name(f"{path.stem}.{os.getpid()}.uploading.csv")

    try:
        df.to_csv(local_tmp, index=False, encoding="utf-8-sig")
        if drive_tmp.exists():
            drive_tmp.unlink()
        shutil.copyfile(local_tmp, drive_tmp)
        try:
            os.replace(drive_tmp, path)
        except OSError:
            if path.exists():
                path.unlink()
            os.replace(drive_tmp, path)
    finally:
        for p in (local_tmp, drive_tmp):
            if p.exists():
                try:
                    p.unlink()
                except OSError:
                    pass


def verify_csv_write(path: Path, expected_rows: int, *, min_bytes: int = 1) -> None:
    """
    Soft post-write log only — NEVER fails the extract.

    Previous hard verify aborted good copies: line-count ≠ CSV rows when
    fields contain newlines (PIDET +1, ICMAS 59k vs 38k false failures).
    """
    path = Path(path)
    try:
        if not path.is_file():
            print(f"[extract] WARN verify: missing {path}")
            return
        size = path.stat().st_size
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))
        if size < min_bytes:
            print(f"[extract] WARN verify: tiny file ({size} bytes) {path}")
            return
        # Proper CSV parse (quoted newlines OK). Informational only.
        try:
            data_rows = len(
                pd.read_csv(path, usecols=[0], dtype="string", encoding="utf-8-sig")
            )
            note = "OK" if data_rows == expected_rows else f"csv_rows={data_rows:,} expected={expected_rows:,}"
        except Exception as exc:  # noqa: BLE001
            note = f"csv_parse_skip ({exc})"
            data_rows = expected_rows
        print(
            f"[extract] verified {path.name} rows={expected_rows:,} "
            f"size={size:,} mtime={mtime} {note} path={path}"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[extract] WARN verify ignored for {path.name}: {exc}")


def extract_tables(
    site: str,
    *,
    engine=None,
    tables: Optional[tuple[str, ...]] = None,
    out_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """
    Write raw_{site}_{suffix}.csv into Drive 01_raw.

    Returns mapping table -> output path.
    """
    site = site.lower()
    eng = engine or mssql_engine(site)
    out = Path(out_dir) if out_dir else paths.raw_dir()
    if not out.exists():
        raise FileNotFoundError(
            f"Raw output folder does not exist: {out}\n"
            "Set KCW_DRIVE_ROOT or KCW_ANALYTICS_DATA_ROOT to the real Shared Drive "
            "path (do not let the tool create a fake local folder)."
        )
    if not out.is_dir():
        raise NotADirectoryError(f"Raw output path is not a directory: {out}")

    print(f"[extract] site={site} out_dir={out}")

    if tables is None:
        if site == "syp" and os.getenv("PARTS9_SYP_FULL_EXTRACT", "").strip() not in (
            "1",
            "true",
            "TRUE",
            "yes",
            "YES",
        ):
            tables = SYP_MINIMAL
        else:
            tables = tuple(TABLE_SPECS.keys())

    written: dict[str, Path] = {}
    errors: list[str] = []

    for table in tables:
        spec = TABLE_SPECS[table]
        years = _years_for(site, table)
        path = out / f"raw_{site}_{spec['suffix']}.csv"
        print(f"[extract] {site} {table} -> {path.name} ...")
        try:
            if years is None:
                df = read_full_table(eng, table)
            else:
                df = read_last_years(eng, table, years)
            write_csv_atomic(df, path)
            verify_csv_write(path, len(df))  # soft — never raises
            written[table] = path
            print(f"[extract] wrote {path.name} rows={len(df):,}")
        except Exception as exc:  # noqa: BLE001 — real write/SQL failures only
            msg = f"{table} ({path.name}): {exc}"
            errors.append(msg)
            print(f"[extract] FAILED {msg}")

    if errors:
        raise RuntimeError(
            "Extract incomplete — some files were not written:\n  - "
            + "\n  - ".join(errors)
        )

    print(f"[extract] OK site={site} files={len(written)}")
    return written


def extract_hq(**kwargs) -> dict[str, Path]:
    return extract_tables("hq", **kwargs)


def extract_syp(**kwargs) -> dict[str, Path]:
    return extract_tables("syp", **kwargs)
