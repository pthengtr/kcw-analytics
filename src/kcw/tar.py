"""Daily TAR / 3TAR / CNTAR / 3CNTAR bill generation with catch-up + skip-if-done."""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from src.kcw import paths
from src.kcw.utils import get_nonvat_sales_lines_last_purchase_vat

DateLike = Union[str, date, datetime, pd.Timestamp]

POS_COLS = [
    "run_id",
    "billdate",
    "billno",
    "bcode",
    "detail",
    "qty",
    "mtp",
    "ui",
    "price",
    "amount",
]
NEG_COLS = POS_COLS + ["po"]

COL_MAP = {
    "BILLDATE": "billdate",
    "BILLNO": "billno",
    "BCODE": "bcode",
    "DETAIL": "detail",
    "QTY": "qty",
    "MTP": "mtp",
    "UI": "ui",
    "PRICE": "price",
    "AMOUNT": "amount",
    "PO": "po",
}


def supabase_db_url(password: Optional[str] = None) -> str:
    """
    Build pooler URL from env.

    Preferred: SUPABASE_DB_URL (full connection string)
    Else: DB_PASSWORD + optional SUPABASE_DB_HOST / SUPABASE_DB_USER
    """
    full = os.getenv("SUPABASE_DB_URL")
    if full:
        return full

    pw = password or os.getenv("DB_PASSWORD")
    if not pw:
        raise RuntimeError("Set SUPABASE_DB_URL or DB_PASSWORD in the environment.")

    user = os.getenv(
        "SUPABASE_DB_USER",
        "postgres.jdzitzsucntqbjvwiwxm",
    )
    host = os.getenv(
        "SUPABASE_DB_HOST",
        "aws-0-ap-southeast-1.pooler.supabase.com",
    )
    port = os.getenv("SUPABASE_DB_PORT", "5432")
    db = os.getenv("SUPABASE_DB_NAME", "postgres")
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


def build_run_id(run_date: DateLike, prefix: str = "TEST") -> str:
    d = pd.to_datetime(run_date)
    return f"{prefix}_{d.strftime('%Y%m%d')}"


def to_date(value: DateLike) -> date:
    return pd.to_datetime(value).date()


def filter_by_date(df: pd.DataFrame, date_col: str, target_date: DateLike) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.date
    target = to_date(target_date)
    return out[out[date_col] == target].copy()


def split_negative_amount(
    df: pd.DataFrame,
    amount_col: str = "AMOUNT",
    *,
    copy: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if amount_col not in df.columns:
        raise ValueError(f"{amount_col} not found in dataframe")
    amount = pd.to_numeric(df[amount_col], errors="coerce")
    mask_neg = amount < 0
    df_negative = df.loc[mask_neg]
    df_positive = df.loc[~mask_neg]
    if copy:
        df_negative = df_negative.copy()
        df_positive = df_positive.copy()
    if verbose:
        print(
            f"[split_negative_amount] negative={len(df_negative):,} | "
            f"non_negative={len(df_positive):,} | total={len(df):,}"
        )
    return df_negative, df_positive


def join_po_from_simas(
    df_target: pd.DataFrame,
    df_simas: pd.DataFrame,
    *,
    key: str = "BILLNO",
    po_col: str = "PO",
    verbose: bool = True,
) -> pd.DataFrame:
    tgt = df_target.copy()
    sim = df_simas.copy()
    tgt["_JOIN_KEY"] = tgt[key].astype("string").str.strip().str.upper()
    sim["_JOIN_KEY"] = sim[key].astype("string").str.strip().str.upper()
    simas_lookup = sim[["_JOIN_KEY", po_col]].drop_duplicates(subset=["_JOIN_KEY"])
    result = tgt.merge(simas_lookup, on="_JOIN_KEY", how="left").drop(columns=["_JOIN_KEY"])
    if verbose:
        matched = result[po_col].notna().sum()
        print(f"[join_po_from_simas] matched PO rows: {matched:,}/{len(result):,}")
    return result


def process_branch_sales(
    out_df: pd.DataFrame,
    data: dict,
    branch: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_neg, out_pos = split_negative_amount(out_df)
    simas_key = f"raw_{branch}_simas_sales_bills.csv"
    df_simas = data[simas_key].copy()
    out_neg = join_po_from_simas(out_neg, df_simas)
    return out_pos, out_neg


def get_last_two_years_nonvat_sales_lines_last_purchase_vat(
    data: dict,
    *,
    source: str,
    date_col: str = "BILLDATE",
    verbose: bool = True,
) -> pd.DataFrame:
    candidate_years: list[int] = []
    for obj in data.values():
        if isinstance(obj, pd.DataFrame) and date_col in obj.columns:
            y = pd.to_datetime(obj[date_col], errors="coerce").dt.year.max()
            if pd.notna(y):
                candidate_years.append(int(y))
    if not candidate_years:
        raise ValueError(f"Could not find any DataFrame in `data` with a valid {date_col}")

    latest_year = max(candidate_years)
    years = [latest_year - 1, latest_year]
    outs = []
    for y in years:
        out = get_nonvat_sales_lines_last_purchase_vat(data, year=y, source=source)
        out = out.copy()
        out["YEAR"] = y
        outs.append(out)
    result = pd.concat(outs, ignore_index=True)
    if verbose:
        print(f"[last_two_years_nonvat] source={source} years={years} rows={len(result):,}")
    return result


def load_raw_csvs(raw_folder: Optional[Path] = None) -> dict[str, pd.DataFrame]:
    folder = Path(raw_folder) if raw_folder else paths.raw_dir()
    if not folder.is_dir():
        raise FileNotFoundError(f"Raw folder not found: {folder}")

    data: dict[str, pd.DataFrame] = {}
    for path in sorted(folder.glob("*.csv")):
        data[path.name] = pd.read_csv(
            path,
            dtype={"BCODE": "string", "ITEMNO": "string", "BILLNO": "string"},
            encoding="utf-8-sig",
            low_memory=False,
        )
    if not data:
        raise FileNotFoundError(f"No CSV files in {folder}")
    print(f"[load_raw_csvs] loaded {len(data)} files from {folder}")
    return data


def prepare_eligible_frames(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    hq = get_last_two_years_nonvat_sales_lines_last_purchase_vat(data, source="hq")
    syp = get_last_two_years_nonvat_sales_lines_last_purchase_vat(data, source="syp")
    return hq, syp


def _apply_day_filters(df: pd.DataFrame, *, site: str) -> pd.DataFrame:
    out = df.copy()
    mask = out["BILLNO"].astype("string").str.contains("TF", na=False)
    out = out.loc[~mask].copy()

    billno = out["BILLNO"].astype("string").str.strip().str.upper()
    if site == "hq":
        mask = billno.str.startswith("DN", na=False)
    else:
        mask = billno.str.startswith(("DN", "3DN"), na=False)
    out = out.loc[~mask].copy()

    mask = out["BCODE"].astype("string").str.startswith(("70", "91"), na=False)
    out = out.loc[~mask].copy()
    return out


def run_sql(
    sql: str,
    params: Optional[Sequence] = None,
    *,
    db_url: Optional[str] = None,
    fetch: bool = False,
):
    import psycopg2

    url = db_url or supabase_db_url()
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    try:
        cur.execute(sql, params)
        rows = cur.fetchall() if fetch else None
        conn.commit()
        return rows
    finally:
        cur.close()
        conn.close()


def copy_csv_to_supabase(
    csv_path: Path,
    table_name: str,
    columns: Sequence[str],
    *,
    schema: str = "billgen",
    db_url: Optional[str] = None,
    truncate_first: bool = False,
) -> None:
    import psycopg

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    url = db_url or supabase_db_url()
    col_sql = ", ".join(columns)
    full_table = f"{schema}.{table_name}"

    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            if truncate_first:
                cur.execute(f"truncate table {full_table};")
            copy_sql = f"""
                COPY {full_table} ({col_sql})
                FROM STDIN
                WITH (FORMAT CSV, HEADER TRUE, ENCODING 'UTF8')
            """
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                with cur.copy(copy_sql) as copy:
                    while chunk := f.read(1024 * 1024):
                        copy.write(chunk)
        conn.commit()
    print(f"Loaded {csv_path.name} -> {full_table}")


def max_fin_billdate(db_url: Optional[str] = None) -> Optional[date]:
    rows = run_sql(
        "select billgen.max_fin_billdate();",
        db_url=db_url,
        fetch=True,
    )
    if not rows or rows[0][0] is None:
        # Fallback if migration not applied yet
        rows = run_sql(
            """
            select max(billdate) from (
                select billdate from billgen.fin_tar_lines
                union all
                select billdate from billgen.fin_3tar_lines
                union all
                select billdate from billgen.fin_cntar_lines
                union all
                select billdate from billgen.fin_3cntar_lines
            ) t;
            """,
            db_url=db_url,
            fetch=True,
        )
    value = rows[0][0] if rows else None
    return value if value is None else to_date(value)


def is_day_processed(bill_date: DateLike, db_url: Optional[str] = None) -> bool:
    d = to_date(bill_date)
    try:
        rows = run_sql(
            "select billgen.is_day_processed(%s::date);",
            (d,),
            db_url=db_url,
            fetch=True,
        )
        return bool(rows and rows[0][0])
    except Exception:
        rows = run_sql(
            """
            select exists (
                select 1 from billgen.fin_tar_lines where billdate = %s
                union all
                select 1 from billgen.fin_3tar_lines where billdate = %s
                union all
                select 1 from billgen.fin_cntar_lines where billdate = %s
                union all
                select 1 from billgen.fin_3cntar_lines where billdate = %s
            );
            """,
            (d, d, d, d),
            db_url=db_url,
            fetch=True,
        )
        return bool(rows and rows[0][0])


def process_all_bill_types_day(
    run_id: str,
    bill_date: DateLike,
    *,
    db_url: Optional[str] = None,
) -> None:
    d = to_date(bill_date)
    try:
        run_sql(
            "select billgen.process_all_bill_types_day(%s, %s::date);",
            (run_id, d),
            db_url=db_url,
        )
    except Exception as exc:
        # Migration not applied: fall back to sequential calls (same order).
        msg = str(exc).lower()
        if "process_all_bill_types_day" not in msg and "function billgen.process_all" not in msg:
            raise
        print("[tar] process_all_bill_types_day missing; running sequential process_*_day")
        for fn in (
            "process_tar_day",
            "process_3tar_day",
            "process_cntar_day",
            "process_3cntar_day",
        ):
            run_sql(
                f"select billgen.{fn}(%s, %s::date);",
                (run_id, d),
                db_url=db_url,
            )
    print(f"Processed all bill types: run_id={run_id}, bill_date={d}")


def eligible_max_billdate(hq: pd.DataFrame, syp: pd.DataFrame) -> Optional[date]:
    frames = []
    for df in (hq, syp):
        if df is None or df.empty or "BILLDATE" not in df.columns:
            continue
        frames.append(pd.to_datetime(df["BILLDATE"], errors="coerce"))
    if not frames:
        return None
    latest = pd.concat(frames).max()
    if pd.isna(latest):
        return None
    return latest.date()


def iter_catchup_dates(
    *,
    today: Optional[date] = None,
    eligible_end: Optional[date] = None,
    db_url: Optional[str] = None,
    start_override: Optional[date] = None,
) -> list[date]:
    """
    Dates from (max_fin + 1) through min(today, eligible_end), inclusive.

    Empty days are included so catch-up walks forward; process_day no-ops on empty staging.
    """
    today = today or date.today()
    end = today
    if eligible_end is not None:
        end = min(end, eligible_end)

    if start_override is not None:
        start = start_override
    else:
        max_fin = max_fin_billdate(db_url=db_url)
        start = (max_fin + timedelta(days=1)) if max_fin else end

    if start > end:
        return []

    dates: list[date] = []
    d = start
    while d <= end:
        dates.append(d)
        d += timedelta(days=1)
    return dates


def run_bill_generation_for_day(
    run_date: DateLike,
    *,
    data: dict,
    hq_eligible: pd.DataFrame,
    syp_eligible: pd.DataFrame,
    db_url: Optional[str] = None,
    run_id_prefix: str = "TEST",
    skip_if_done: bool = True,
) -> str:
    """
    Stage one day and process all bill types.

    Returns status: 'skipped' | 'processed' | 'empty'
    """
    d = to_date(run_date)
    run_id = build_run_id(d, run_id_prefix)
    url = db_url or supabase_db_url()

    if skip_if_done and is_day_processed(d, db_url=url):
        print(f"[tar] skip-if-done: {d} already in fin_*")
        return "skipped"

    hq = _apply_day_filters(filter_by_date(hq_eligible, "BILLDATE", d), site="hq")
    syp = _apply_day_filters(filter_by_date(syp_eligible, "BILLDATE", d), site="syp")

    out_hq_pos, out_hq_neg = process_branch_sales(hq, data, branch="hq")
    out_syp_pos, out_syp_neg = process_branch_sales(syp, data, branch="syp")

    pos_cols = ["BILLDATE", "BILLNO", "BCODE", "DETAIL", "QTY", "MTP", "UI", "PRICE", "AMOUNT"]
    neg_cols = pos_cols + ["PO"]

    def _prep(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        wanted = NEG_COLS if "PO" in cols else POS_COLS
        if df.empty:
            return pd.DataFrame(columns=wanted)
        out = df[cols].copy().rename(columns=COL_MAP)
        out["run_id"] = run_id
        return out[wanted]

    hq_pos = _prep(out_hq_pos, pos_cols)
    syp_pos = _prep(out_syp_pos, pos_cols)
    hq_neg = _prep(out_hq_neg, neg_cols)
    syp_neg = _prep(out_syp_neg, neg_cols)

    total_rows = len(hq_pos) + len(syp_pos) + len(hq_neg) + len(syp_neg)
    print(
        f"[tar] {d} run_id={run_id} rows "
        f"hq_pos={len(hq_pos)} hq_neg={len(hq_neg)} "
        f"syp_pos={len(syp_pos)} syp_neg={len(syp_neg)}"
    )

    tar_dir = paths.ensure_dir(paths.tar_output_dir("TAR"))
    tar3_dir = paths.ensure_dir(paths.tar_output_dir("3TAR"))
    hq_pos_csv = tar_dir / "out_hq_pos.csv"
    hq_neg_csv = tar_dir / "out_hq_neg.csv"
    syp_pos_csv = tar3_dir / "out_syp_pos.csv"
    syp_neg_csv = tar3_dir / "out_syp_neg.csv"

    hq_pos.to_csv(hq_pos_csv, index=False, encoding="utf-8-sig")
    hq_neg.to_csv(hq_neg_csv, index=False, encoding="utf-8-sig")
    syp_pos.to_csv(syp_pos_csv, index=False, encoding="utf-8-sig")
    syp_neg.to_csv(syp_neg_csv, index=False, encoding="utf-8-sig")

    copy_csv_to_supabase(hq_pos_csv, "stg_tar_lines", POS_COLS, db_url=url, truncate_first=True)
    copy_csv_to_supabase(hq_neg_csv, "stg_cntar_lines", NEG_COLS, db_url=url, truncate_first=True)
    copy_csv_to_supabase(syp_pos_csv, "stg_3tar_lines", POS_COLS, db_url=url, truncate_first=True)
    copy_csv_to_supabase(syp_neg_csv, "stg_3cntar_lines", NEG_COLS, db_url=url, truncate_first=True)

    process_all_bill_types_day(run_id, d, db_url=url)
    return "empty" if total_rows == 0 else "processed"


def run_catchup(
    *,
    raw_folder: Optional[Path] = None,
    today: Optional[date] = None,
    start_date: Optional[DateLike] = None,
    end_date: Optional[DateLike] = None,
    db_url: Optional[str] = None,
    skip_if_done: bool = True,
    run_id_prefix: str = "TEST",
) -> dict[str, int]:
    """
    Load raw CSVs once, then process every missing day up to today (or end_date).

    Idempotent: already-processed days are skipped when skip_if_done=True.
    """
    url = db_url or supabase_db_url()
    data = load_raw_csvs(raw_folder)
    hq_eligible, syp_eligible = prepare_eligible_frames(data)

    today = today or date.today()
    eligible_end = eligible_max_billdate(hq_eligible, syp_eligible)
    if end_date is not None:
        eligible_end = to_date(end_date) if eligible_end is None else min(eligible_end, to_date(end_date))

    start_override = to_date(start_date) if start_date is not None else None
    dates = iter_catchup_dates(
        today=today,
        eligible_end=eligible_end,
        db_url=url,
        start_override=start_override,
    )

    summary = {"processed": 0, "skipped": 0, "empty": 0, "dates": len(dates)}
    print(
        f"[tar] catch-up dates={len(dates)} "
        f"range={dates[0] if dates else None}..{dates[-1] if dates else None} "
        f"eligible_end={eligible_end}"
    )

    for d in dates:
        status = run_bill_generation_for_day(
            d,
            data=data,
            hq_eligible=hq_eligible,
            syp_eligible=syp_eligible,
            db_url=url,
            run_id_prefix=run_id_prefix,
            skip_if_done=skip_if_done,
        )
        summary[status] = summary.get(status, 0) + 1

    print(f"[tar] catch-up done: {summary}")
    return summary


def delete_fin_for_day(bill_date: DateLike, *, db_url: Optional[str] = None) -> None:
    """Explicit reprocess helper: remove fin_* rows for one billdate (does not rewind seq)."""
    d = to_date(bill_date)
    for table in (
        "fin_tar_lines",
        "fin_3tar_lines",
        "fin_cntar_lines",
        "fin_3cntar_lines",
    ):
        run_sql(
            f"delete from billgen.{table} where billdate = %s;",
            (d,),
            db_url=db_url,
        )
    print(f"[tar] deleted fin_* rows for {d} (seq not rewound — review bill_seq_control)")
