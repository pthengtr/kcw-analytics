"""Upload Drive raw_*.csv files into raw_kcw.* via staging replace."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd

from src.kcw import paths
from src.kcw.tar import supabase_db_url

# HQ A extracts these full masters; upload them to Supabase after Drive write.
ARMAS_APMAS_UPLOADS = (
    {
        "csv_name": "raw_hq_armas_receivable.csv",
        "main_table": "raw_kcw.raw_hq_armas_receivable",
        "staging_table": "raw_kcw.raw_hq_armas_receivable_stg",
    },
    {
        "csv_name": "raw_hq_apmas_payable.csv",
        "main_table": "raw_kcw.raw_hq_apmas_payable",
        "staging_table": "raw_kcw.raw_hq_apmas_payable_stg",
    },
)

# Purchase orders: both sites (POs can be raised at HQ or SYP).
POMAS_PODET_UPLOADS = (
    {
        "csv_name": "raw_hq_pomas_purchase_orders.csv",
        "main_table": "raw_kcw.raw_hq_pomas_purchase_orders",
        "staging_table": "raw_kcw.raw_hq_pomas_purchase_orders_stg",
        "site": "hq",
    },
    {
        "csv_name": "raw_hq_podet_purchase_order_lines.csv",
        "main_table": "raw_kcw.raw_hq_podet_purchase_order_lines",
        "staging_table": "raw_kcw.raw_hq_podet_purchase_order_lines_stg",
        "site": "hq",
    },
    {
        "csv_name": "raw_syp_pomas_purchase_orders.csv",
        "main_table": "raw_kcw.raw_syp_pomas_purchase_orders",
        "staging_table": "raw_kcw.raw_syp_pomas_purchase_orders_stg",
        "site": "syp",
    },
    {
        "csv_name": "raw_syp_podet_purchase_order_lines.csv",
        "main_table": "raw_kcw.raw_syp_podet_purchase_order_lines",
        "staging_table": "raw_kcw.raw_syp_podet_purchase_order_lines_stg",
        "site": "syp",
    },
)

# Purchase invoices: HQ only (actual purchases / AP happen at HQ).
PIMAS_PIDET_UPLOADS = (
    {
        "csv_name": "raw_hq_pimas_purchase_bills.csv",
        "main_table": "raw_kcw.raw_hq_pimas_purchase_bills",
        "staging_table": "raw_kcw.raw_hq_pimas_purchase_bills_stg",
    },
    {
        "csv_name": "raw_hq_pidet_purchase_lines.csv",
        "main_table": "raw_kcw.raw_hq_pidet_purchase_lines",
        "staging_table": "raw_kcw.raw_hq_pidet_purchase_lines_stg",
    },
)

# Product masters: both sites (Drive extract already writes these CSVs).
ICMAS_UPLOADS = (
    {
        "csv_name": "raw_hq_icmas_products.csv",
        "main_table": "raw_kcw.raw_hq_icmas_products",
        "staging_table": "raw_kcw.raw_hq_icmas_products_stg",
        "site": "hq",
    },
    {
        "csv_name": "raw_syp_icmas_products.csv",
        "main_table": "raw_kcw.raw_syp_icmas_products",
        "staging_table": "raw_kcw.raw_syp_icmas_products_stg",
        "site": "syp",
    },
)

# Receipt / notes vouchers: HQ only (includes RC* receipt vouchers).
RVMAS_UPLOADS = (
    {
        "csv_name": "raw_hq_rvmas_notes_vouchers.csv",
        "main_table": "raw_kcw.raw_hq_rvmas_notes_vouchers",
        "staging_table": "raw_kcw.raw_hq_rvmas_notes_vouchers_stg",
    },
)

# Payment / notes vouchers: HQ only (includes P* and KCPN* payment vouchers).
PVMAS_UPLOADS = (
    {
        "csv_name": "raw_hq_pvmas_notes_vouchers.csv",
        "main_table": "raw_kcw.raw_hq_pvmas_notes_vouchers",
        "staging_table": "raw_kcw.raw_hq_pvmas_notes_vouchers_stg",
    },
)

# Stock-order / pending-receive tracker: HQ + SYP (ICLOW; ค้างรับ = ORDERED=Y RECEIVED=N).
ICLOW_UPLOADS = (
    {
        "csv_name": "raw_hq_iclow_stock_orders.csv",
        "main_table": "raw_kcw.raw_hq_iclow_stock_orders",
        "staging_table": "raw_kcw.raw_hq_iclow_stock_orders_stg",
        "site": "hq",
    },
    {
        "csv_name": "raw_syp_iclow_stock_orders.csv",
        "main_table": "raw_kcw.raw_syp_iclow_stock_orders",
        "staging_table": "raw_kcw.raw_syp_iclow_stock_orders_stg",
        "site": "syp",
    },
)

# Daily HQ A upload set (run after SYP + HQ extracts land on Drive).
DAILY_RAW_UPLOADS = (
    ARMAS_APMAS_UPLOADS
    + POMAS_PODET_UPLOADS
    + PIMAS_PIDET_UPLOADS
    + ICMAS_UPLOADS
    + RVMAS_UPLOADS
    + PVMAS_UPLOADS
    + ICLOW_UPLOADS
)


def refresh_table_via_staging_df(
    conn,
    df: pd.DataFrame,
    main_table: str,
    staging_table: str,
    source_file: str | None = None,
) -> dict:
    """
    Load DataFrame into staging, validate, then replace main from staging.

    Matches notebooks/90_csv_to_supabase.ipynb behavior:
      delete staging -> COPY df -> optional _source_file stamp ->
      delete main -> insert main select * from staging
    """
    if df is None or df.empty:
        raise ValueError(f"Input DataFrame is empty for {main_table}. Main table not touched.")

    load_df = df.copy()
    cols = load_df.columns.tolist()
    quoted_cols = ", ".join(f'"{c}"' for c in cols)

    load_df = load_df.astype("object").where(pd.notna(load_df), None)

    buffer = StringIO()
    load_df.to_csv(
        buffer,
        index=False,
        header=True,
        encoding="utf-8",
        quoting=csv.QUOTE_MINIMAL,
    )
    buffer.seek(0)

    copy_sql = f"""
    COPY {staging_table} ({quoted_cols})
    FROM STDIN WITH CSV HEADER
    """

    with conn.cursor() as cur:
        cur.execute(
            f"""
            create table if not exists {staging_table}
            (like {main_table} including all)
            """
        )
        cur.execute(f"delete from {staging_table}")

        with cur.copy(copy_sql) as copy:
            while data := buffer.read(1024 * 1024):
                copy.write(data)

        if source_file is not None:
            schema, table = (
                staging_table.split(".", 1)
                if "." in staging_table
                else ("public", staging_table)
            )
            cur.execute(
                """
                select 1
                from information_schema.columns
                where table_schema = %s
                  and table_name = %s
                  and column_name = '_source_file'
                """,
                (schema, table),
            )
            if cur.fetchone() is not None:
                cur.execute(
                    f"""
                    update {staging_table}
                    set _source_file = %s
                    where _source_file is null
                    """,
                    (source_file,),
                )

        cur.execute(f"select count(*) from {staging_table}")
        staging_count = cur.fetchone()[0]
        if staging_count == 0:
            raise ValueError(
                f"Staging load produced 0 rows for {staging_table}. Main table not touched."
            )

        cur.execute(f"delete from {main_table}")
        cur.execute(
            f"""
            insert into {main_table}
            select * from {staging_table}
            """
        )
        cur.execute(f"select count(*) from {main_table}")
        main_count = cur.fetchone()[0]

    conn.commit()
    return {
        "status": "ok",
        "staging_rows": staging_count,
        "main_rows": main_count,
        "source_file": source_file,
        "main_table": main_table,
        "staging_table": staging_table,
    }


def _read_raw_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Raw CSV not found: {path}")
    return pd.read_csv(path, dtype="string", encoding="utf-8-sig", low_memory=False)


def upload_raw_specs(
    specs: tuple[dict, ...],
    *,
    raw_folder: Optional[Path] = None,
    db_url: Optional[str] = None,
    label: str = "raw",
) -> list[dict]:
    """Read Drive CSVs named in specs and replace matching raw_kcw tables."""
    import psycopg

    raw = Path(raw_folder) if raw_folder else paths.raw_dir()
    url = db_url or supabase_db_url()
    results: list[dict] = []

    print(f"[upload-raw] label={label} raw_dir={raw} files={len(specs)}")
    with psycopg.connect(url) as conn:
        for spec in specs:
            csv_path = raw / spec["csv_name"]
            print(f"[upload-raw] reading {csv_path.name} ...")
            df = _read_raw_csv(csv_path)
            print(f"[upload-raw] {csv_path.name} rows={len(df):,} cols={len(df.columns)}")
            result = refresh_table_via_staging_df(
                conn=conn,
                df=df,
                main_table=spec["main_table"],
                staging_table=spec["staging_table"],
                source_file=spec["csv_name"],
            )
            print(
                f"[upload-raw] loaded {spec['csv_name']} -> {spec['main_table']} "
                f"rows={result['main_rows']:,}"
            )
            results.append(result)

    print(f"[upload-raw] OK {label} files={len(results)}")
    return results


def upload_armas_apmas(
    *,
    raw_folder: Optional[Path] = None,
    db_url: Optional[str] = None,
) -> list[dict]:
    """
    Read Drive raw_hq_armas_receivable.csv / raw_hq_apmas_payable.csv
    and replace the matching raw_kcw tables via staging.
    """
    return upload_raw_specs(
        ARMAS_APMAS_UPLOADS,
        raw_folder=raw_folder,
        db_url=db_url,
        label="armas/apmas",
    )


def upload_pomas_podet(
    site: str | None = None,
    *,
    raw_folder: Optional[Path] = None,
    db_url: Optional[str] = None,
) -> list[dict]:
    """
    Upload POMAS/PODET Drive CSVs to raw_kcw.

    site=None -> both HQ and SYP
    site='hq'|'syp' -> that site only
    """
    if site is None:
        specs = POMAS_PODET_UPLOADS
        label = "pomas/podet"
    else:
        site = site.lower()
        if site not in ("hq", "syp"):
            raise ValueError("site must be 'hq', 'syp', or None")
        specs = tuple(s for s in POMAS_PODET_UPLOADS if s["site"] == site)
        label = f"pomas/podet-{site}"
    return upload_raw_specs(
        specs,
        raw_folder=raw_folder,
        db_url=db_url,
        label=label,
    )


def upload_iclow(
    site: str | None = None,
    *,
    raw_folder: Optional[Path] = None,
    db_url: Optional[str] = None,
) -> list[dict]:
    """
    Upload ICLOW Drive CSVs to raw_kcw.

    site=None -> both HQ and SYP
    site='hq'|'syp' -> that site only

    Pending receive (ค้างรับ): ORDERED='Y' AND RECEIVED='N' AND CANCELED='N'.
    """
    if site is None:
        specs = ICLOW_UPLOADS
        label = "iclow"
    else:
        site = site.lower()
        if site not in ("hq", "syp"):
            raise ValueError("site must be 'hq', 'syp', or None")
        specs = tuple(s for s in ICLOW_UPLOADS if s["site"] == site)
        label = f"iclow-{site}"
    return upload_raw_specs(
        specs,
        raw_folder=raw_folder,
        db_url=db_url,
        label=label,
    )


def upload_icmas(
    site: str | None = None,
    *,
    raw_folder: Optional[Path] = None,
    db_url: Optional[str] = None,
) -> list[dict]:
    """
    Upload ICMAS (product / stock master) Drive CSVs to raw_kcw.

    site=None -> both HQ and SYP
    site='hq'|'syp' -> that site only
    """
    if site is None:
        specs = ICMAS_UPLOADS
        label = "icmas"
    else:
        site = site.lower()
        if site not in ("hq", "syp"):
            raise ValueError("site must be 'hq', 'syp', or None")
        specs = tuple(s for s in ICMAS_UPLOADS if s["site"] == site)
        label = f"icmas-{site}"
    return upload_raw_specs(
        specs,
        raw_folder=raw_folder,
        db_url=db_url,
        label=label,
    )


def upload_po_related(
    site: str | None = None,
    *,
    raw_folder: Optional[Path] = None,
    db_url: Optional[str] = None,
) -> list[dict]:
    """
    Upload PO-related Drive CSVs to raw_kcw: POMAS/PODET + ICMAS + ICLOW.

    site=None -> both HQ and SYP
    site='hq'|'syp' -> that site only
    """
    results: list[dict] = []
    results.extend(upload_pomas_podet(site, raw_folder=raw_folder, db_url=db_url))
    results.extend(upload_icmas(site, raw_folder=raw_folder, db_url=db_url))
    results.extend(upload_iclow(site, raw_folder=raw_folder, db_url=db_url))
    return results


def upload_daily_raw(
    *,
    raw_folder: Optional[Path] = None,
    db_url: Optional[str] = None,
) -> list[dict]:
    """
    Daily Drive -> Supabase raw upload after SYP + HQ extracts:

      - HQ ARMAS / APMAS
      - HQ + SYP POMAS / PODET
      - HQ PIMAS / PIDET (purchase invoices; HQ only)
      - HQ + SYP ICMAS (product masters)
      - HQ RVMAS (receipt / notes vouchers; includes RC*)
      - HQ PVMAS (payment / notes vouchers; includes P* / KCPN*)
      - HQ + SYP ICLOW (stock-order / pending-receive tracker)
    """
    return upload_raw_specs(
        DAILY_RAW_UPLOADS,
        raw_folder=raw_folder,
        db_url=db_url,
        label="daily-raw",
    )
