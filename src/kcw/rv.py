"""Daily RV / 3RV bill generation. Separate from TAR. Persists to billgen.fin_rv_*."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from src.kcw import paths
from src.kcw.tar import (
    build_run_id,
    copy_csv_to_supabase,
    filter_by_date,
    load_raw_csvs,
    run_sql,
    supabase_db_url,
    to_date,
)
from src.kcw.utils import get_vat_sales_lines_last_purchase_nonvat, is_transfer_stock_billno

DateLike = Union[str, date, datetime, pd.Timestamp]

LINE_COLS = [
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
    "last_cost",
]

_BILL_SEQ = re.compile(r"-(\d+)$")


def buddhist_yyyymm(d: date) -> str:
    return f"{(d.year + 543) % 100:02d}{d.month:02d}"


def bill_prefix(kind: str, d: date) -> str:
    return f"{kind}{buddhist_yyyymm(d)}-"


def max_issued_seq(names: list[str], prefix: str) -> int:
    """Highest NNN already used by bill numbers or PDF stems starting with prefix."""
    best = 0
    for name in names:
        stem = Path(str(name)).stem
        if not stem.startswith(prefix):
            continue
        match = _BILL_SEQ.search(stem)
        if match:
            best = max(best, int(match.group(1)))
    return best


def apply_rv_day_filters(df: pd.DataFrame, *, site: str) -> pd.DataFrame:
    """Notebook 20 drops, plus stock-transfer bills. Negatives stay on the RV."""
    if df is None or df.empty:
        return df.iloc[0:0].copy() if df is not None else pd.DataFrame()
    out = df.loc[~is_transfer_stock_billno(df["BILLNO"])].copy()
    billno = out["BILLNO"].astype("string").str.strip().str.upper()
    if site == "hq":
        drop = billno.str.startswith(("TAR", "CN", "DN"), na=False)
    else:
        drop = billno.str.startswith(("3TAR", "3CN", "3DN"), na=False)
    out = out.loc[~drop].copy()
    service = out["BCODE"].astype("string").str.startswith(("70", "91"), na=False)
    return out.loc[~service].copy()


def enrich_last_cost(
    sales_lines: pd.DataFrame,
    purchases: pd.DataFrame,
    *,
    bcode_col: str = "BCODE",
    date_col: str = "BILLDATE",
) -> pd.DataFrame:
    """Unit cost of the latest purchase on or before the sale date, per BCODE."""
    sales = sales_lines.copy()
    purch = purchases.copy()
    sales[bcode_col] = sales[bcode_col].astype("string").str.strip()
    purch[bcode_col] = purch[bcode_col].astype("string").str.strip()
    sales[date_col] = pd.to_datetime(sales[date_col], errors="coerce").astype("datetime64[ns]")
    purch[date_col] = pd.to_datetime(purch[date_col], errors="coerce").astype("datetime64[ns]")
    sales = sales.dropna(subset=[bcode_col, date_col])
    sales = sales[sales[bcode_col] != ""].copy()
    purch = purch.dropna(subset=[bcode_col, date_col])
    purch = purch[purch[bcode_col] != ""].copy()

    for col in ("AMOUNT", "QTY", "MTP"):
        purch[col] = pd.to_numeric(purch[col], errors="coerce")
    purch = purch.dropna(subset=["AMOUNT", "QTY", "MTP"])
    purch = purch[(purch["QTY"] != 0) & (purch["MTP"] > 0)].copy()
    units = purch["QTY"] * purch["MTP"]
    purch = purch.assign(__LASTCOST__=purch["AMOUNT"] / units.replace(0, pd.NA))
    purch = purch.dropna(subset=["__LASTCOST__"])
    purch_key = purch[[bcode_col, date_col, "__LASTCOST__"]].sort_values(
        [date_col, bcode_col], kind="mergesort"
    )
    sales = sales.sort_values([date_col, bcode_col], kind="mergesort")
    merged = pd.merge_asof(
        sales,
        purch_key,
        left_on=date_col,
        right_on=date_col,
        by=bcode_col,
        direction="backward",
        allow_exact_matches=True,
    )
    merged["LAST_COST"] = pd.to_numeric(merged["__LASTCOST__"], errors="coerce")
    return merged.drop(columns=["__LASTCOST__"])


def refill_last_cost_from_icmas(df: pd.DataFrame, icmas: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["BCODE"] = result["BCODE"].astype("string").str.strip().str.upper()
    lookup = icmas[["BCODE", "COSTNET"]].copy()
    lookup["BCODE"] = lookup["BCODE"].astype("string").str.strip().str.upper()
    lookup["COSTNET"] = pd.to_numeric(lookup["COSTNET"], errors="coerce")
    lookup = lookup.drop_duplicates(subset=["BCODE"])
    result["LAST_COST"] = pd.to_numeric(result["LAST_COST"], errors="coerce")
    result = result.merge(lookup, on="BCODE", how="left")
    invalid = result["LAST_COST"].isna() | (result["LAST_COST"] == 0)
    result.loc[invalid, "LAST_COST"] = result.loc[invalid, "COSTNET"]
    return result.drop(columns=["COSTNET"])


def apply_cost_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Drop zero sale amount and missing cost, then price the line at last cost."""
    if df is None or df.empty:
        return df.iloc[0:0].copy() if df is not None else pd.DataFrame()
    out = df.copy()
    out["LAST_COST"] = pd.to_numeric(out["LAST_COST"], errors="coerce")
    sale_amount = pd.to_numeric(out["AMOUNT"], errors="coerce")
    out = out[
        out["LAST_COST"].notna()
        & (out["LAST_COST"] != 0)
        & sale_amount.notna()
        & (sale_amount != 0)
    ].copy()
    if out.empty:
        return out
    qty = pd.to_numeric(out["QTY"], errors="coerce").fillna(0)
    mtp = pd.to_numeric(out["MTP"], errors="coerce").fillna(0)
    out["PRICE"] = out["LAST_COST"].round(2)
    out["AMOUNT"] = out["PRICE"] * qty * mtp
    return out


def get_last_two_years_vat_sales_last_purchase_nonvat(
    data: dict,
    *,
    source: str,
) -> pd.DataFrame:
    years = _latest_two_years(data)
    frames = []
    for year in years:
        part = get_vat_sales_lines_last_purchase_nonvat(data, year=year, source=source)
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _latest_two_years(data: dict) -> list[int]:
    years: list[int] = []
    for obj in data.values():
        if isinstance(obj, pd.DataFrame) and "BILLDATE" in obj.columns:
            parsed = pd.to_datetime(obj["BILLDATE"], errors="coerce")
            if parsed.notna().any():
                years.append(int(parsed.dt.year.max()))
    if not years:
        raise ValueError("Could not find a BILLDATE column in raw CSVs")
    latest = max(years)
    return [latest - 1, latest]


def prepare_eligible_frames(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    purchases = data["raw_hq_pidet_purchase_lines.csv"]
    icmas = data["raw_hq_icmas_products.csv"]
    frames = []
    for source in ("hq", "syp"):
        sales = get_last_two_years_vat_sales_last_purchase_nonvat(data, source=source)
        if sales.empty:
            frames.append(sales)
            continue
        priced = enrich_last_cost(sales, purchases)
        priced = refill_last_cost_from_icmas(priced, icmas)
        frames.append(apply_cost_lines(priced))
    return frames[0], frames[1]


def max_fin_billdate(db_url: Optional[str] = None) -> Optional[date]:
    rows = run_sql("select billgen.max_fin_rv_billdate();", db_url=db_url, fetch=True)
    value = rows[0][0] if rows else None
    return None if value is None else to_date(value)


def is_day_processed(bill_date: DateLike, db_url: Optional[str] = None) -> bool:
    d = to_date(bill_date)
    rows = run_sql(
        "select billgen.is_rv_day_processed(%s::date);",
        (d,),
        db_url=db_url,
        fetch=True,
    )
    return bool(rows and rows[0][0])


def _month_has_fin(kind: str, d: date, *, db_url: Optional[str]) -> bool:
    table = "fin_rv_lines" if kind == "RV" else "fin_3rv_lines"
    rows = run_sql(
        f"""
        select exists (
            select 1 from billgen.{table}
            where billdate >= %s and billdate < %s
        );
        """,
        (date(d.year, d.month, 1), _next_month(d)),
        db_url=db_url,
        fetch=True,
    )
    return bool(rows and rows[0][0])


def _next_month(d: date) -> date:
    if d.month == 12:
        return date(d.year + 1, 1, 1)
    return date(d.year, d.month + 1, 1)


def _issued_names(kind: str, year: int, month: int) -> list[str]:
    folder = paths.rv_output_dir(kind) / f"{kind}_{year}_{month}"
    names: list[str] = []
    csv_path = folder / "CSV" / f"{kind}_{year}_{month}.csv"
    if csv_path.is_file():
        try:
            frame = pd.read_csv(csv_path, dtype=str, usecols=["NEW_BILLNO"])
        except ValueError:
            frame = pd.DataFrame()
        if not frame.empty and "NEW_BILLNO" in frame.columns:
            names.extend(frame["NEW_BILLNO"].dropna().astype(str).tolist())
    pdf_dir = folder / "PDF"
    if pdf_dir.is_dir():
        names.extend(path.stem for path in pdf_dir.glob("*.pdf"))
    return names


def _stage_frame(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=LINE_COLS)
    out = pd.DataFrame(
        {
            "run_id": run_id,
            "billdate": pd.to_datetime(df["BILLDATE"], errors="coerce").dt.date,
            "billno": df["BILLNO"].astype("string"),
            "bcode": df["BCODE"].astype("string"),
            "detail": df["DETAIL"].astype("string") if "DETAIL" in df.columns else "",
            "qty": pd.to_numeric(df["QTY"], errors="coerce"),
            "mtp": pd.to_numeric(df["MTP"], errors="coerce"),
            "ui": df["UI"].astype("string") if "UI" in df.columns else "",
            "price": pd.to_numeric(df["PRICE"], errors="coerce"),
            "amount": pd.to_numeric(df["AMOUNT"], errors="coerce"),
            "last_cost": pd.to_numeric(df["LAST_COST"], errors="coerce"),
        }
    )
    return out[LINE_COLS]


def _read_month_csv(kind: str, year: int, month: int) -> pd.DataFrame:
    path = paths.rv_output_dir(kind) / f"{kind}_{year}_{month}" / "CSV" / f"{kind}_{year}_{month}.csv"
    if not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if frame.empty or "NEW_BILLNO" not in frame.columns:
        return pd.DataFrame()
    return frame


def _issued_rows_for_fin(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows already printed by notebook 20, priced at last cost, keeping NEW_BILLNO."""
    if frame.empty:
        return pd.DataFrame(columns=LINE_COLS + ["new_billno"])
    out = frame.copy()
    out = out[pd.to_datetime(out["BILLDATE"], errors="coerce").notna()].copy()
    out["NEW_BILLNO"] = out["NEW_BILLNO"].astype("string").str.strip()
    out = out[out["NEW_BILLNO"].notna() & (out["NEW_BILLNO"] != "") & (out["NEW_BILLNO"] != "<NA>")].copy()
    out["LAST_COST"] = pd.to_numeric(out["LAST_COST"], errors="coerce")
    out = out[out["LAST_COST"].notna() & (out["LAST_COST"] != 0)].copy()
    if out.empty:
        return pd.DataFrame(columns=LINE_COLS + ["new_billno"])
    billdate = pd.to_datetime(out["BILLDATE"], errors="coerce").dt.date
    qty = pd.to_numeric(out["QTY"], errors="coerce").fillna(0)
    mtp = pd.to_numeric(out["MTP"], errors="coerce").fillna(0)
    price = out["LAST_COST"].round(2)
    return pd.DataFrame(
        {
            "run_id": [build_run_id(d) for d in billdate],
            "billdate": billdate,
            "billno": out["BILLNO"].astype("string"),
            "bcode": out["BCODE"].astype("string"),
            "detail": out["DETAIL"].astype("string") if "DETAIL" in out.columns else "",
            "qty": qty,
            "mtp": mtp,
            "ui": out["UI"].astype("string") if "UI" in out.columns else "",
            "price": price,
            "amount": price * qty * mtp,
            "last_cost": out["LAST_COST"],
            "new_billno": out["NEW_BILLNO"].astype("string").str.strip(),
        }
    )


def adopt_issued_month(bill_date: DateLike, *, db_url: Optional[str] = None) -> None:
    """Copy this month's already-printed RV rows into fin_* and continue the sequence.

    Notebook 20 owns receipts printed before this pipeline. Adopting them keeps
    those bill numbers and leaves the PDFs in place. Later days are numbered
    after the highest printed sequence.
    """
    d = to_date(bill_date)
    url = db_url or supabase_db_url()
    tables = {"RV": "fin_rv_lines", "3RV": "fin_3rv_lines"}
    for kind, table in tables.items():
        if _month_has_fin(kind, d, db_url=url):
            continue
        issued = max_issued_seq(_issued_names(kind, d.year, d.month), bill_prefix(kind, d))
        rows = _issued_rows_for_fin(_read_month_csv(kind, d.year, d.month))
        if not rows.empty:
            staging = paths.ensure_dir(paths.rv_output_dir(kind) / "_staging")
            csv_path = staging / f"adopt_{kind}.csv"
            rows[LINE_COLS + ["new_billno"]].to_csv(csv_path, index=False, encoding="utf-8-sig")
            copy_csv_to_supabase(
                csv_path,
                table,
                LINE_COLS + ["new_billno"],
                db_url=url,
                truncate_first=False,
            )
            print(f"[rv] adopted {len(rows)} printed {kind} lines into {table}")
        if issued <= 0:
            continue
        yyyymm = buddhist_yyyymm(d)
        run_sql(
            """
            insert into billgen.bill_seq_control (bill_type, yyyymm, last_seq)
            values (%s, %s, %s)
            on conflict (bill_type, yyyymm) do update
            set last_seq = greatest(billgen.bill_seq_control.last_seq, excluded.last_seq),
                updated_at = now()
            where billgen.bill_seq_control.last_seq < excluded.last_seq;
            """,
            (kind, yyyymm, issued),
            db_url=url,
        )
        print(f"[rv] {kind} {yyyymm} last_seq={issued}")


def catchup_start(
    *,
    today: date,
    eligible_end: Optional[date],
    max_fin: Optional[date],
    start_override: Optional[date] = None,
) -> Optional[date]:
    end = today if eligible_end is None else min(today, eligible_end)
    if start_override is not None:
        start = start_override
    elif max_fin is not None:
        start = max_fin + timedelta(days=1)
    else:
        start = end
    if start > end:
        return None
    return start


def iter_catchup_dates(
    *,
    today: Optional[date] = None,
    eligible_end: Optional[date] = None,
    db_url: Optional[str] = None,
    start_override: Optional[date] = None,
) -> list[date]:
    today = today or date.today()
    start = catchup_start(
        today=today,
        eligible_end=eligible_end,
        max_fin=max_fin_billdate(db_url=db_url),
        start_override=start_override,
    )
    if start is None:
        return []
    end = today if eligible_end is None else min(today, eligible_end)
    dates: list[date] = []
    d = start
    while d <= end:
        dates.append(d)
        d += timedelta(days=1)
    return dates


def run_bill_generation_for_day(
    run_date: DateLike,
    *,
    hq_eligible: pd.DataFrame,
    syp_eligible: pd.DataFrame,
    db_url: Optional[str] = None,
    run_id_prefix: str = "TEST",
    skip_if_done: bool = True,
) -> str:
    d = to_date(run_date)
    run_id = build_run_id(d, run_id_prefix)
    url = db_url or supabase_db_url()

    if skip_if_done and is_day_processed(d, db_url=url):
        print(f"[rv] skip-if-done: {d} already in fin_rv_*")
        return "skipped"

    hq = apply_rv_day_filters(filter_by_date(hq_eligible, "BILLDATE", d), site="hq")
    syp = apply_rv_day_filters(filter_by_date(syp_eligible, "BILLDATE", d), site="syp")
    hq_lines = _stage_frame(hq, run_id)
    syp_lines = _stage_frame(syp, run_id)
    print(f"[rv] {d} run_id={run_id} hq={len(hq_lines)} syp={len(syp_lines)}")

    staging = paths.ensure_dir(paths.rv_output_dir("RV") / "_staging")
    hq_csv = staging / "out_hq.csv"
    syp_csv = staging / "out_syp.csv"
    hq_lines.to_csv(hq_csv, index=False, encoding="utf-8-sig")
    syp_lines.to_csv(syp_csv, index=False, encoding="utf-8-sig")

    copy_csv_to_supabase(hq_csv, "stg_rv_lines", LINE_COLS, db_url=url, truncate_first=True)
    copy_csv_to_supabase(syp_csv, "stg_3rv_lines", LINE_COLS, db_url=url, truncate_first=True)
    run_sql(
        "select billgen.process_all_rv_types_day(%s, %s::date);",
        (run_id, d),
        db_url=url,
    )
    print(f"[rv] processed run_id={run_id} bill_date={d}")
    return "empty" if (hq_lines.empty and syp_lines.empty) else "processed"


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
    url = db_url or supabase_db_url()
    today = today or date.today()
    adopt_issued_month(today, db_url=url)
    data = load_raw_csvs(raw_folder)
    hq_eligible, syp_eligible = prepare_eligible_frames(data)
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
        f"[rv] catch-up dates={len(dates)} "
        f"range={dates[0] if dates else None}..{dates[-1] if dates else None}"
    )
    for d in dates:
        status = run_bill_generation_for_day(
            d,
            hq_eligible=hq_eligible,
            syp_eligible=syp_eligible,
            db_url=url,
            run_id_prefix=run_id_prefix,
            skip_if_done=skip_if_done,
        )
        summary[status] = summary.get(status, 0) + 1
    print(f"[rv] catch-up done: {summary}")
    return summary


def delete_fin_for_day(bill_date: DateLike, *, db_url: Optional[str] = None) -> None:
    """Remove fin_rv_* rows for one day. Does not rewind bill_seq_control."""
    d = to_date(bill_date)
    for table in ("fin_rv_lines", "fin_3rv_lines"):
        run_sql(
            f"delete from billgen.{table} where billdate = %s;",
            (d,),
            db_url=db_url,
        )
    print(f"[rv] deleted fin_rv_* rows for {d} (seq not rewound)")
