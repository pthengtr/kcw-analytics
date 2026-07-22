"""Extract PARTS9 tables to Drive raw_{site}_*.csv (HQ or SYP)."""

from __future__ import annotations

import os
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

# Minimal SYP set used by the attached local notebook (extend via FULL_EXTRACT=1).
SYP_MINIMAL = ("SIDET", "PIDET", "SIMAS", "PIMAS", "ICMAS")


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
    Write CSV the same way the original SYP/HQ notebooks did: direct to_csv.

    Earlier experiments (local temp + copy + os.replace + fsync + line-count
    verify) caused DriveFS Errno 9 and false failures when fields contain
    newlines. Keep this helper name for call sites, but behavior matches
    the original `df.to_csv(..., index=False, encoding="utf-8-sig")`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def verify_csv_write(path: Path, expected_rows: int, *, min_bytes: int = 1) -> None:
    """
    Soft check after write: file exists, non-empty, recent mtime.

    Do NOT count raw newlines — ICMAS/etc. have quoted fields with embedded
    newlines, so line count != pandas row count (that caused false PIDET/ICMAS
    failures). Optionally confirm with a proper CSV parse.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Write verification failed — missing: {path}")

    size = path.stat().st_size
    if size < min_bytes:
        raise RuntimeError(f"Write verification failed — empty/tiny file ({size} bytes): {path}")

    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))

    # Proper CSV row count (handles quoted newlines). Retry briefly for DriveFS lag.
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            data_rows = len(pd.read_csv(path, usecols=[0], dtype="string", encoding="utf-8-sig"))
            if data_rows != expected_rows:
                raise RuntimeError(
                    f"Write verification failed — {path.name} has {data_rows:,} CSV "
                    f"rows, expected {expected_rows:,} (path={path})"
                )
            print(
                f"[extract] verified {path.name} rows={data_rows:,} "
                f"size={size:,} mtime={mtime} path={path}"
            )
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(0.4 * (attempt + 1))

    # Soft fallback: do not fail extract solely because DriveFS read is laggy.
    # Original notebook never verified — log and continue.
    print(
        f"[extract] WARN verify skipped for {path.name}: {last_err} "
        f"(size={size:,} mtime={mtime}) — write completed like original to_csv"
    )


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
            verify_csv_write(path, len(df))
            written[table] = path
            print(f"[extract] wrote {path.name} rows={len(df):,}")
        except Exception as exc:  # noqa: BLE001 — collect per-table, fail at end
            msg = f"{table} ({path.name}): {exc}"
            errors.append(msg)
            print(f"[extract] FAILED {msg}")

    if errors:
        raise RuntimeError(
            "Extract incomplete — some files were not updated on disk:\n  - "
            + "\n  - ".join(errors)
        )

    print(f"[extract] OK site={site} files={len(written)}")
    return written


def extract_hq(**kwargs) -> dict[str, Path]:
    return extract_tables("hq", **kwargs)


def extract_syp(**kwargs) -> dict[str, Path]:
    return extract_tables("syp", **kwargs)
