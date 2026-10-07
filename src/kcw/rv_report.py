"""RV / 3RV PDF + CSV from billgen.fin_rv_*. Does not renumber bills."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from sqlalchemy import create_engine, text

from src.kcw import paths
from src.kcw.rv import buddhist_yyyymm
from src.kcw.tar import supabase_db_url, to_date
from src.kcw.tar_report import _write_pdf_atomic, font_paths

DateLike = Union[str, date, datetime, pd.Timestamp]

TH_MONTHS_ABBR = [
    "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
    "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค.",
]


def thai_date(value) -> str:
    dt = pd.to_datetime(value).to_pydatetime()
    return f"{dt.day} {TH_MONTHS_ABBR[dt.month - 1]} {dt.year + 543}"


def thai_baht_text(amount) -> str:
    x = float(amount) if amount is not None else 0.0
    if x < 0:
        return "ลบ" + thai_baht_text(-x)
    baht = int(x)
    satang = int(round((x - baht) * 100))
    if satang == 100:
        baht += 1
        satang = 0
    words = _thai_read_integer(baht) + "บาท"
    return words + ("ถ้วน" if satang == 0 else _thai_read_integer(satang) + "สตางค์")


def _thai_read_integer(num: int) -> str:
    if num == 0:
        return "ศูนย์"
    units = ["", "สิบ", "ร้อย", "พัน", "หมื่น", "แสน"]
    digits = ["ศูนย์", "หนึ่ง", "สอง", "สาม", "สี่", "ห้า", "หก", "เจ็ด", "แปด", "เก้า"]

    def read_under_million(n: int) -> str:
        text_n = ""
        chars = list(map(int, str(n)))
        length = len(chars)
        for i, digit in enumerate(chars):
            pos = length - i - 1
            if digit == 0:
                continue
            if pos == 0:
                text_n += "เอ็ด" if (digit == 1 and length > 1) else digits[digit]
            elif pos == 1:
                if digit == 1:
                    text_n += "สิบ"
                elif digit == 2:
                    text_n += "ยี่สิบ"
                else:
                    text_n += digits[digit] + "สิบ"
            else:
                text_n += digits[digit] + units[pos]
        return text_n

    parts: list[int] = []
    while num > 0:
        parts.append(num % 1_000_000)
        num //= 1_000_000
    out = ""
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == 0:
            continue
        out += read_under_million(parts[i])
        if i != 0:
            out += "ล้าน"
    return out


def _money(value) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return ""


def _qty(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    return str(int(number)) if number.is_integer() else f"{number:,.2f}"


def _esc(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def month_pdf_dir(kind: str, year: int, month: int) -> Path:
    return paths.rv_output_dir(kind) / f"{kind}_{year}_{month}" / "PDF"


def month_csv_dir(kind: str, year: int, month: int) -> Path:
    return paths.rv_output_dir(kind) / f"{kind}_{year}_{month}" / "CSV"


def remap_rv(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "billdate": "BILLDATE",
            "billno": "BILLNO",
            "new_billno": "NEW_BILLNO",
            "bcode": "BCODE",
            "detail": "DETAIL",
            "qty": "QTY",
            "mtp": "MTP",
            "ui": "UI",
            "price": "PRICE",
            "amount": "AMOUNT",
            "last_cost": "LAST_COST",
        }
    )


def _read_sql_df(sql: str, params: Optional[dict] = None, *, db_url: Optional[str] = None) -> pd.DataFrame:
    engine = create_engine(db_url or supabase_db_url())
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def build_one_receipt_pdf(group_df: pd.DataFrame, pdf_path: Path) -> None:
    regular, bold, signature = font_paths()
    new_billno = str(group_df["NEW_BILLNO"].iloc[0])
    billdate = thai_date(group_df["BILLDATE"].iloc[0])
    branch_text = "สี่แยกพัฒนา" if new_billno.startswith("3") else "สำนักงานใหญ่"

    df = group_df.copy()
    df["QTY"] = pd.to_numeric(df.get("QTY", 0), errors="coerce").fillna(0)
    df["MTP"] = pd.to_numeric(df.get("MTP", 0), errors="coerce").fillna(0)
    unit = pd.to_numeric(df.get("PRICE", df.get("LAST_COST")), errors="coerce").round(2)
    df["UNIT_PRICE"] = unit
    df["AMOUNT_CALC"] = df["UNIT_PRICE"].fillna(0) * df["QTY"] * df["MTP"]
    grand_total = float(df["AMOUNT_CALC"].sum())

    rows = []
    for _, row in df.iterrows():
        unit_price = _money(row["UNIT_PRICE"]) if pd.notna(row["UNIT_PRICE"]) else "UNKNOWN"
        rows.append(
            "<tr>"
            f"<td class=\"c-bcode\">{_esc(row.get('BCODE', ''))}</td>"
            f"<td class=\"c-detail\">{_esc(row.get('DETAIL', '')).replace(chr(10), ' ')}</td>"
            f"<td class=\"c-num\">{unit_price}</td>"
            f"<td class=\"c-num\">{_qty(row.get('QTY', 0))}</td>"
            f"<td class=\"c-unit\">{_esc(row.get('UI', ''))}</td>"
            f"<td class=\"c-num\">{_qty(row.get('MTP', 1))}</td>"
            f"<td class=\"c-num\">{_money(row.get('AMOUNT_CALC', 0))}</td>"
            "</tr>"
        )
    rows_html = "\n".join(rows)
    sig_url = Path(signature).resolve().as_uri()
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    @font-face {{
      font-family: "Sarabun";
      src: url("{Path(regular).resolve().as_uri()}");
    }}
    @font-face {{
      font-family: "Sarabun";
      src: url("{Path(bold).resolve().as_uri()}");
      font-weight: bold;
    }}
    @page {{ size: A4; margin: 18px 24px; }}
    body {{ font-family: "Sarabun"; font-size: 14px; line-height: 1.35; color: #000; }}
    .title {{ font-size: 22px; font-weight: bold; margin-bottom: 8px; }}
    .hdr {{ display: flex; justify-content: flex-end; gap: 24px; margin-bottom: 10px; }}
    .hdr-col {{ text-align: right; }}
    .lbl {{ font-weight: bold; }}
    .block {{ margin: 6px 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border: 1px solid #000; padding: 4px 6px; vertical-align: top; }}
    th {{ background: #f2f2f2; text-align: center; font-weight: bold; }}
    .c-bcode {{ width: 14%; }}
    .c-detail {{ width: 46%; }}
    .c-unit {{ width: 8%; text-align: center; }}
    .c-num {{ width: 10%; text-align: right; white-space: nowrap; }}
    .totals {{ margin-top: 10px; text-align: right; font-weight: bold; font-size: 16px; }}
    .words {{ margin-top: 2px; text-align: right; font-size: 13px; font-weight: normal; }}
    .sign-wrap {{ margin-top: 18px; display: flex; justify-content: flex-end; }}
    .sign-table {{ border-collapse: collapse; width: 320px; }}
    .sign-table td {{ border: none; padding: 2px 6px; }}
    .sign-img {{ width: 188px; height: 48px; object-fit: contain; display: block; }}
    .note {{ margin-top: 10px; text-align: right; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="hdr">
    <div class="hdr-col">
      <div class="title">ใบสำคัญรับเงิน</div>
      <div><span class="lbl">เลขที่:</span> {_esc(new_billno)}</div>
      <div><span class="lbl">วันที่:</span> {_esc(billdate)}</div>
    </div>
  </div>
  <div class="block"><span class="lbl">ข้าพเจ้า:</span> นางสาวนฤมล วิทยผโลทัย (ผู้ขายสินค้า)</div>
  <div class="block"><span class="lbl">ที่อยู่:</span> 305 หมู่ 1 ตำบล ชุมแสง อำเภอ วังจันทร์ จังหวัด ระยอง</div>
  <div class="block"><span class="lbl">เลขประจำตัวผู้เสียภาษี:</span> 1-2001-99001-42-8</div>
  <div class="block" style="margin-top:10px;">
    ได้รับเงินจาก บริษัทเกียรติชัยอะไหล่ยนต์ 2007 จำกัด ({_esc(branch_text)}) (ผู้ซื้อ) ดังรายการต่อไปนี้
  </div>
  <table>
    <thead>
      <tr>
        <th>รหัสสินค้า</th><th>รายการ</th><th>ราคา/หน่วยย่อย</th>
        <th>จำนวน</th><th>หน่วย</th><th>บรรจุ</th><th>รวมยอดเงิน</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
  <div class="totals">รวมทั้งสิ้น: {_money(grand_total)}</div>
  <div class="words">จำนวนเงิน (ตัวอักษร): {_esc(thai_baht_text(grand_total))}</div>
  <div class="sign-wrap">
    <table class="sign-table">
      <tr>
        <td style="text-align:right; width:110px;">ผู้รับเงิน</td>
        <td style="text-align:right;"><img class="sign-img" src="{sig_url}"></td>
      </tr>
      <tr>
        <td style="text-align:right;">ผู้จ่ายเงิน</td>
        <td style="text-align:right;"><img class="sign-img" src="{sig_url}"></td>
      </tr>
    </table>
  </div>
  <div class="note">หมายเหตุ: แนบสำเนาบัตรประชาชนผู้รับเงิน</div>
</body>
</html>
"""
    _write_pdf_atomic(html, pdf_path)


def build_receipts(df: pd.DataFrame, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty or "NEW_BILLNO" not in df.columns:
        print(f"[rv-report] 0 receipts -> {out_dir}")
        return 0
    work = df[df["NEW_BILLNO"].notna() & (df["NEW_BILLNO"].astype("string").str.strip() != "")]
    groups = list(work.groupby("NEW_BILLNO", sort=True))
    written = 0
    print(f"[rv-report] {len(groups)} receipts -> {out_dir}")
    for new_billno, group in groups:
        dest = out_dir / f"{new_billno}.pdf"
        if dest.exists():
            continue
        build_one_receipt_pdf(group, dest)
        written += 1
    return written


def _append_month_csv(kind: str, year: int, month: int, rows: pd.DataFrame, bill_date: date) -> None:
    """Write that day's lines beside the notebook month CSV. Do not rewrite it."""
    if rows is None or rows.empty:
        return
    keep = [
        "BILLDATE", "BILLNO", "NEW_BILLNO", "BCODE", "DETAIL",
        "QTY", "MTP", "UI", "PRICE", "AMOUNT", "LAST_COST",
    ]
    out = rows[[c for c in keep if c in rows.columns]].copy()
    path = month_csv_dir(kind, year, month) / f"{kind}_{year}_{month}_{bill_date.strftime('%Y%m%d')}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")


def lines_for_day(bill_date: DateLike, *, db_url: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = to_date(bill_date)
    url = db_url or supabase_db_url()
    hq = remap_rv(
        _read_sql_df(
            "select * from billgen.fin_rv_lines where billdate = :d order by new_billno, bcode",
            {"d": d},
            db_url=url,
        )
    )
    syp = remap_rv(
        _read_sql_df(
            "select * from billgen.fin_3rv_lines where billdate = :d order by new_billno, bcode",
            {"d": d},
            db_url=url,
        )
    )
    return hq, syp


def output_rv_report_daily(run_date: DateLike, *, db_url: Optional[str] = None) -> dict[str, int]:
    d = to_date(run_date)
    hq, syp = lines_for_day(d, db_url=db_url)
    hq_n = build_receipts(hq, month_pdf_dir("RV", d.year, d.month))
    syp_n = build_receipts(syp, month_pdf_dir("3RV", d.year, d.month))
    _append_month_csv("RV", d.year, d.month, hq, d)
    _append_month_csv("3RV", d.year, d.month, syp, d)
    return {
        "hq_rv": int(hq["NEW_BILLNO"].nunique()) if not hq.empty else 0,
        "syp_rv": int(syp["NEW_BILLNO"].nunique()) if not syp.empty else 0,
        "hq_pdf_written": hq_n,
        "syp_pdf_written": syp_n,
    }


def live_new_billnos(year: int, month: int, *, db_url: Optional[str] = None) -> set[str]:
    sql = """
        select distinct new_billno from billgen.fin_rv_lines
        where extract(year from billdate) = :y and extract(month from billdate) = :m
        union
        select distinct new_billno from billgen.fin_3rv_lines
        where extract(year from billdate) = :y and extract(month from billdate) = :m
    """
    df = _read_sql_df(sql, {"y": year, "m": month}, db_url=db_url)
    if df.empty:
        return set()
    return {str(v).strip() for v in df.iloc[:, 0].dropna() if str(v).strip()}


def prune_stale_pdfs(year: int, month: int, *, db_url: Optional[str] = None) -> list[Path]:
    """Delete month PDFs whose bill number is no longer in fin_rv_*."""
    live = live_new_billnos(year, month, db_url=db_url)
    if not live:
        return []
    prefix = buddhist_yyyymm(date(year, month, 1))
    removed: list[Path] = []
    for kind in ("RV", "3RV"):
        pdf_dir = month_pdf_dir(kind, year, month)
        if not pdf_dir.is_dir():
            continue
        for path in sorted(pdf_dir.glob("*.pdf")):
            if not path.stem.startswith(f"{kind}{prefix}-"):
                continue
            if path.stem in live:
                continue
            print(f"[rv-report] stale PDF remove: {path.name}")
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed


def iter_dates(start: DateLike, end: DateLike) -> list[date]:
    s, e = to_date(start), to_date(end)
    if s > e:
        s, e = e, s
    out: list[date] = []
    d = s
    while d <= e:
        out.append(d)
        d += timedelta(days=1)
    return out


def fin_dates_for_month(year: int, month: int, *, db_url: Optional[str] = None) -> list[date]:
    sql = """
        select distinct billdate from billgen.fin_rv_lines
        where extract(year from billdate) = :y and extract(month from billdate) = :m
        union
        select distinct billdate from billgen.fin_3rv_lines
        where extract(year from billdate) = :y and extract(month from billdate) = :m
        order by 1
    """
    df = _read_sql_df(sql, {"y": year, "m": month}, db_url=db_url)
    if df.empty:
        return []
    return [to_date(v) for v in df.iloc[:, 0].tolist()]


def run_rv_reports(
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
    *,
    prune_stale: bool = True,
    db_url: Optional[str] = None,
) -> dict[str, object]:
    if start is None:
        today = date.today()
        dates = fin_dates_for_month(today.year, today.month, db_url=db_url)
    else:
        dates = iter_dates(start, end or start)
    summaries = []
    for d in dates:
        summaries.append({"date": d.isoformat(), **output_rv_report_daily(d, db_url=db_url)})
    pruned: list[str] = []
    if prune_stale and dates:
        year, month = dates[0].year, dates[0].month
        pruned = [str(p) for p in prune_stale_pdfs(year, month, db_url=db_url)]
    return {"days": summaries, "pruned": pruned}
