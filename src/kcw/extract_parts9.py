"""Extract PARTS9 tables to Drive raw_{site}_*.csv (HQ or SYP)."""

from __future__ import annotations

import os
import urllib.parse
from pathlib import Path
from typing import Optional

import pandas as pd

from src.kcw import paths

# Rolling windows + filenames match notebooks/51_parts9_to_drive.ipynb
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
    out = Path(out_dir) if out_dir else paths.ensure_dir(paths.raw_dir())
    out.mkdir(parents=True, exist_ok=True)

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
    for table in tables:
        spec = TABLE_SPECS[table]
        print(f"[extract] {site} {table} ...")
        if spec["years"] is None:
            df = read_full_table(eng, table)
        else:
            df = read_last_years(eng, table, spec["years"])
        path = out / f"raw_{site}_{spec['suffix']}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        written[table] = path
        print(f"[extract] wrote {path.name} rows={len(df):,}")

    return written


def extract_hq(**kwargs) -> dict[str, Path]:
    return extract_tables("hq", **kwargs)


def extract_syp(**kwargs) -> dict[str, Path]:
    return extract_tables("syp", **kwargs)
