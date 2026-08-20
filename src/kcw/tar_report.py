"""Daily TAR / 3TAR / CNTAR / 3CNTAR PDF + CSV reports (Drive 04_outputs/06_TAR)."""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from weasyprint import HTML

from src.kcw import paths
from src.kcw.tar import build_run_id, supabase_db_url, to_date

DateLike = Union[str, date, datetime, pd.Timestamp]

COMPANY_INFO = {
    "hq": {
        "name": "บริษัท เกียรติชัยอะไหล่ยนต์ 2007 จำกัด (สำนักงานใหญ่)",
        "address": "ที่อยู่ 305 ม.1 ต.ชุมแสง อ.วังจันทร์ จ.ระยอง 21210",
        "phone": "โทร. 038-666-078",
        "tax": "เลขประจำตัวผู้เสียภาษี 0215560000262",
    },
    "syp": {
        "name": "บริษัท เกียรติชัยอะไหล่ยนต์ 2007 จำกัด (สาขาสี่แยกพัฒนา)",
        "address": "ที่อยู่ 16/2 ม.2 ต.ห้วยทับมอญ อ.เขาชะเมา จ.ระยอง 21110",
        "phone": "โทร. 063-2655387, 038-015818",
        "tax": "เลขประจำตัวผู้เสียภาษี 0215560000262 (สาขาที่ 00003)",
    },
}

TH_MONTHS_ABBR = [
    "ม.ค.", "ก.พ.", "มี.ค.", "เม.ย.", "พ.ค.", "มิ.ย.",
    "ก.ค.", "ส.ค.", "ก.ย.", "ต.ค.", "พ.ย.", "ธ.ค.",
]


def get_company_info(new_billno: str) -> dict[str, str]:
    if str(new_billno).startswith("3"):
        return COMPANY_INFO["syp"]
    return COMPANY_INFO["hq"]


def thai_date(d) -> str:
    dt = pd.to_datetime(d).to_pydatetime()
    return f"{dt.day} {TH_MONTHS_ABBR[dt.month - 1]} {dt.year + 543}"


def _money(x) -> str:
    try:
        return f"{float(x):,.2f}"
    except (TypeError, ValueError):
        return ""


def font_paths(root: Optional[Path] = None) -> tuple[Path, Path, Path]:
    base = Path(root) if root else paths.analytics_root()
    fonts = base / "00_fonts"
    return (
        fonts / "THSarabunNew" / "THSarabunNew.ttf",
        fonts / "THSarabunNew" / "THSarabunNew-Bold.ttf",
        fonts / "Signature.jpg",
    )


def month_pdf_dir(kind: str, year: int, month: int, *, root: Optional[Path] = None) -> Path:
    base = Path(root) if root else paths.tar_output_dir(kind)
    return base / f"{kind}_{year}_{month:02d}" / "PDF"


def month_csv_dir(kind: str, year: int, month: int, *, root: Optional[Path] = None) -> Path:
    base = Path(root) if root else paths.tar_output_dir(kind)
    return base / f"{kind}_{year}_{month:02d}" / "CSV"


def remap_cn_to_legacy(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "billdate": "BILLDATE",
            "billno": "BILLNO",
            "new_billno": "NEG_BILLNO",
            "ref_new_billno": "NEW_BILLNO",
            "bcode": "BCODE",
            "detail": "DETAIL",
            "qty": "QTY",
            "mtp": "MTP",
            "ui": "UI",
            "price": "PRICE",
            "amount": "AMOUNT",
            "po": "PO",
        }
    )


def remap_to_legacy(df: pd.DataFrame) -> pd.DataFrame:
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
            "po": "PO",
        }
    )


def build_bill_summary(
    df: pd.DataFrame,
    *,
    billno_col: str = "NEW_BILLNO",
    neg_billno_col: str = "NEG_BILLNO",
    billdate_col: str = "BILLDATE",
    detail_col: str = "DETAIL",
    amount_col: str = "AMOUNT",
    tax_rate: float = 0.07,
    tax_id_value: str = "0000000000000",
) -> pd.DataFrame:
    out = df.copy()
    if out.empty or billno_col not in out.columns:
        cols = [
            billno_col, "TOTAL_AMOUNT", "BILLDATE", "BEFORE_VAT",
            "VAT_AMOUNT", detail_col, "TAX_ID", "SEQ",
        ]
        if neg_billno_col in df.columns:
            cols.insert(1, "NEG_BILLNO")
        return pd.DataFrame(columns=cols)

    out[billno_col] = out[billno_col].astype("string")
    if neg_billno_col in out.columns:
        out[neg_billno_col] = out[neg_billno_col].astype("string")
    if detail_col not in out.columns:
        out[detail_col] = ""
    out[amount_col] = pd.to_numeric(out[amount_col], errors="coerce").fillna(0)

    idx_max_amt = out.groupby(billno_col)[amount_col].idxmax()
    detail_pick = (
        out.loc[idx_max_amt, [billno_col, detail_col]]
        .set_index(billno_col)[detail_col]
    )

    agg_dict = {
        "TOTAL_AMOUNT": (amount_col, "sum"),
        "BILLDATE": (billdate_col, "first"),
    }
    if neg_billno_col in out.columns:
        agg_dict["NEG_BILLNO"] = (neg_billno_col, "first")

    totals = out.groupby(billno_col, as_index=False).agg(**agg_dict)
    divisor = 1 + tax_rate
    totals["BEFORE_VAT"] = (totals["TOTAL_AMOUNT"] / divisor).round(2)
    totals["VAT_AMOUNT"] = (totals["TOTAL_AMOUNT"] - totals["BEFORE_VAT"]).round(2)
    totals[detail_col] = totals[billno_col].map(detail_pick)
    totals["TAX_ID"] = str(tax_id_value).zfill(13)[:13]

    sort_cols = [billno_col]
    if "NEG_BILLNO" in totals.columns and totals["NEG_BILLNO"].notna().any():
        sort_cols = ["NEG_BILLNO", "BILLDATE", billno_col]
    totals = totals.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    totals["SEQ"] = np.arange(1, len(totals) + 1)
    return totals


def _write_pdf_atomic(html: str, dest: Path) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, local_name = tempfile.mkstemp(prefix=f"{dest.stem}_", suffix=".pdf")
    os.close(fd)
    local_tmp = Path(local_name)
    drive_tmp = dest.with_name(f"{dest.stem}.{os.getpid()}.uploading.pdf")
    try:
        HTML(string=html, base_url="/").write_pdf(str(local_tmp))
        if drive_tmp.exists():
            drive_tmp.unlink()
        shutil.copy2(local_tmp, drive_tmp)
        os.replace(drive_tmp, dest)
    finally:
        local_tmp.unlink(missing_ok=True)
        if drive_tmp.exists():
            try:
                drive_tmp.unlink()
            except OSError:
                pass


def build_one_receipt_weasy_vat(
    group_df: pd.DataFrame,
    pdf_path: str | Path,
    *,
    font_regular_path: str,
    font_bold_path: str,
    signature_img_path: str | None = None,
    doc_title: str = "ใบเสร็จรับเงิน/ใบกำกับภาษีอย่างย่อ",
) -> None:
    font_regular_uri = Path(font_regular_path).resolve().as_uri()
    font_bold_uri = Path(font_bold_path).resolve().as_uri()
    if signature_img_path:
        Path(signature_img_path).resolve().as_uri()

    df = group_df.copy()
    new_billno = str(df["NEW_BILLNO"].iloc[0])
    billdate = thai_date(df["BILLDATE"].iloc[0])
    ref_billno = str(df["REF"].iloc[0]) if "REF" in df.columns else ""
    info = get_company_info(new_billno)

    for col, default in {"QTY": 0, "MTP": 1, "PRICE": 0, "AMOUNT": 0}.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    df["VAT_PORTION"] = df["AMOUNT"] * (7.0 / 107.0)
    df["BASE_EXVAT"] = df["AMOUNT"] - df["VAT_PORTION"]
    total_amount = float(df["AMOUNT"].sum())
    total_vat = float(df["VAT_PORTION"].sum())
    total_base = float(df["BASE_EXVAT"].sum())

    rows_html = []
    for _, r in df.iterrows():
        bcode = str(r.get("BCODE", ""))
        detail = str(r.get("DETAIL", ""))
        unit_price = _money(r.get("PRICE", 0))
        qty_val = r.get("QTY", 0)
        try:
            qty = _money(qty_val) if (float(qty_val) % 1) else str(int(float(qty_val)))
        except (TypeError, ValueError):
            qty = str(qty_val)
        unit = str(r.get("UI", ""))
        amount_incl = _money(r.get("AMOUNT", 0))
        rows_html.append(
            f"""
          <tr>
            <td class="c">{bcode}</td>
            <td class="l">{detail}</td>
            <td class="r">{unit_price}</td>
            <td class="r">{qty}</td>
            <td class="c">{unit}</td>
            <td class="r">{amount_incl}</td>
          </tr>
        """
        )

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <style>
    @page {{ size: A4; margin: 18px 24px; }}
    @font-face {{ font-family: "THSarabunNew"; src: url("{font_regular_uri}"); }}
    @font-face {{ font-family: "THSarabunNew"; src: url("{font_bold_uri}"); font-weight: bold; }}
    body {{ font-family: "THSarabunNew"; font-size: 12pt; line-height: 1.35; }}
    .title {{ margin-bottom: 6px; text-align:left; font-weight:700; font-size:20px; }}
    .right {{ text-align: right; }}
    .kv b {{ font-weight: bold; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border: 1px solid #000; padding: 4px 6px; vertical-align: top; }}
    th {{ font-weight: bold; background: #f5f5f5; text-align: center; }}
    .l {{ text-align: left; }}
    .c {{ text-align: center; }}
    .r {{ text-align: right; }}
    .totals {{ margin-top: 10px; width: 100%; }}
    .totals .row {{ display: flex; justify-content: flex-end; gap: 10px; }}
    .totals .label {{ min-width: 140px; text-align: right; font-weight: bold; }}
    .totals .val {{ min-width: 120px; text-align: right; }}
    .header-row{{ display:flex; justify-content:space-between; align-items:flex-start; width:100%; }}
    .company{{ text-align:left; font-size:14px; line-height:1.4; }}
    .company-name{{ font-weight:700; font-size:16px; }}
    .company-line{{ font-size:14px; line-height:1.35; }}
    .company-line.tax{{ margin-top:6px; }}
  </style>
</head>
<body>
  <div class="header-row">
    <div class="company">
      <div class="company-name">{info['name']}</div>
      <div class="company-line">{info['address']}</div>
      <div class="company-line">{info['phone']}</div>
      <div class="company-line tax">{info['tax']}</div>
    </div>
    <div>
      <div class="title">{doc_title}</div>
      <div class="right kv">
        <div><b>เลขที่:</b> {new_billno}</div>
        <div><b>วันที่:</b> {billdate}</div>
        <div><b>อ้างอิง:</b> {ref_billno}</div>
      </div>
    </div>
  </div>
  <table>
    <thead>
      <tr>
        <th style="width: 12%">รหัสสินค้า</th>
        <th style="width: 36%">รายการ</th>
        <th style="width: 10%">ราคา/หน่วย</th>
        <th style="width: 8%">จำนวน</th>
        <th style="width: 8%">หน่วย</th>
        <th style="width: 13%">รวมยอดเงิน<br/>(รวม VAT)</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
  <div class="totals">
    <div class="row"><div class="label">ยอดก่อน VAT:</div><div class="val">{_money(total_base)}</div></div>
    <div class="row"><div class="label">VAT 7%:</div><div class="val">{_money(total_vat)}</div></div>
    <div class="row"><div class="label">รวมทั้งสิ้น (รวม VAT):</div><div class="val">{_money(total_amount)}</div></div>
  </div>
</body>
</html>
"""
    _write_pdf_atomic(html, Path(pdf_path))


def build_receipts_by_new_billno_weasy_vat(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    font_regular_path: str,
    font_bold_path: str,
    signature_img_path: str | None = None,
    doc_title: str = "ใบเสร็จรับเงิน/ใบกำกับภาษีอย่างย่อ",
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty or "NEW_BILLNO" not in df.columns:
        print(f"Generating 0 receipts -> {out_dir}")
        return out_dir

    work = df.copy()
    work = work[work["NEW_BILLNO"].notna() & (work["NEW_BILLNO"].astype("string").str.strip() != "")]
    groups = list(work.groupby("NEW_BILLNO", sort=True))
    total = len(groups)
    print(f"Generating {total} receipts -> {out_dir}")
    for i, (new_billno, g) in enumerate(groups, start=1):
        print(f"  [{i}/{total}] {new_billno}")
        build_one_receipt_weasy_vat(
            g,
            out_dir / f"{new_billno}.pdf",
            font_regular_path=font_regular_path,
            font_bold_path=font_bold_path,
            signature_img_path=signature_img_path,
            doc_title=doc_title,
        )
    return out_dir


def _read_sql_df(sql: str, params: Optional[dict] = None, *, db_url: Optional[str] = None) -> pd.DataFrame:
    engine = create_engine(db_url or supabase_db_url())
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def output_tar_report_daily(
    run_date: DateLike,
    *,
    db_url: Optional[str] = None,
    run_id_prefix: str = "TEST",
) -> dict[str, int]:
    d = to_date(run_date)
    run_id = build_run_id(d, run_id_prefix)
    year, month = d.year, d.month
    url = db_url or supabase_db_url()

    fin_tar = _read_sql_df(
        "select * from billgen.fin_tar_lines where run_id=:r", {"r": run_id}, db_url=url
    )
    fin_3tar = _read_sql_df(
        "select * from billgen.fin_3tar_lines where run_id=:r", {"r": run_id}, db_url=url
    )
    fin_cntar = _read_sql_df(
        "select * from billgen.fin_cntar_lines where run_id=:r", {"r": run_id}, db_url=url
    )
    fin_3cntar = _read_sql_df(
        "select * from billgen.fin_3cntar_lines where run_id=:r", {"r": run_id}, db_url=url
    )

    out_hq_pos = remap_to_legacy(fin_tar)
    out_hq_neg = remap_cn_to_legacy(fin_cntar)
    out_syp_pos = remap_to_legacy(fin_3tar)
    out_syp_neg = remap_cn_to_legacy(fin_3cntar)

    hq_bill_pos_summary = build_bill_summary(out_hq_pos)
    hq_bill_neg_summary = build_bill_summary(out_hq_neg)
    syp_bill_pos_summary = build_bill_summary(out_syp_pos)
    syp_bill_neg_summary = build_bill_summary(out_syp_neg)

    out_hq_neg = out_hq_neg.rename(columns={"NEW_BILLNO": "REF", "NEG_BILLNO": "NEW_BILLNO"})
    out_syp_neg = out_syp_neg.rename(columns={"NEW_BILLNO": "REF", "NEG_BILLNO": "NEW_BILLNO"})

    regular, bold, signature = font_paths()
    font_kw = {
        "font_regular_path": str(regular),
        "font_bold_path": str(bold),
        "signature_img_path": str(signature),
    }

    hq_pdf = month_pdf_dir("TAR", year, month)
    syp_pdf = month_pdf_dir("3TAR", year, month)
    print(f"[tar-report] {d} run_id={run_id}")
    build_receipts_by_new_billno_weasy_vat(out_syp_pos, syp_pdf, **font_kw)
    build_receipts_by_new_billno_weasy_vat(
        out_syp_neg, syp_pdf, **font_kw, doc_title="ใบลดหนี้"
    )
    build_receipts_by_new_billno_weasy_vat(out_hq_pos, hq_pdf, **font_kw)
    build_receipts_by_new_billno_weasy_vat(
        out_hq_neg, hq_pdf, **font_kw, doc_title="ใบลดหนี้"
    )

    hq_csv = month_csv_dir("TAR", year, month)
    syp_csv = month_csv_dir("3TAR", year, month)
    suffix = run_id
    _write_csv(out_hq_pos, hq_csv / f"out_hq_pos_{suffix}.csv")
    _write_csv(out_hq_neg, hq_csv / f"out_hq_neg_{suffix}.csv")
    _write_csv(hq_bill_pos_summary, hq_csv / f"TAR_{year}_{month:02d}_summary_{suffix}.csv")
    _write_csv(hq_bill_neg_summary, hq_csv / f"CNTAR_{year}_{month:02d}_summary_{suffix}.csv")
    _write_csv(out_syp_pos, syp_csv / f"out_syp_pos_{suffix}.csv")
    _write_csv(out_syp_neg, syp_csv / f"out_syp_neg_{suffix}.csv")
    _write_csv(syp_bill_pos_summary, syp_csv / f"3TAR_{year}_{month:02d}_summary_{suffix}.csv")
    _write_csv(syp_bill_neg_summary, syp_csv / f"3CNTAR_{year}_{month:02d}_summary_{suffix}.csv")

    return {
        "hq_tar": int(out_hq_pos["NEW_BILLNO"].nunique()) if not out_hq_pos.empty else 0,
        "hq_cntar": int(out_hq_neg["NEW_BILLNO"].nunique()) if not out_hq_neg.empty else 0,
        "syp_tar": int(out_syp_pos["NEW_BILLNO"].nunique()) if not out_syp_pos.empty else 0,
        "syp_cntar": int(out_syp_neg["NEW_BILLNO"].nunique()) if not out_syp_neg.empty else 0,
    }


def live_new_billnos(year: int, month: int, *, db_url: Optional[str] = None) -> set[str]:
    sql = """
        select distinct new_billno from billgen.fin_tar_lines
        where extract(year from billdate) = :y and extract(month from billdate) = :m
        union
        select distinct new_billno from billgen.fin_3tar_lines
        where extract(year from billdate) = :y and extract(month from billdate) = :m
        union
        select distinct new_billno from billgen.fin_cntar_lines
        where extract(year from billdate) = :y and extract(month from billdate) = :m
        union
        select distinct new_billno from billgen.fin_3cntar_lines
        where extract(year from billdate) = :y and extract(month from billdate) = :m
    """
    df = _read_sql_df(sql, {"y": year, "m": month}, db_url=db_url)
    if df.empty:
        return set()
    return {str(v).strip() for v in df.iloc[:, 0].dropna() if str(v).strip()}


def prune_stale_pdfs(
    year: int,
    month: int,
    *,
    db_url: Optional[str] = None,
    dry_run: bool = False,
) -> list[Path]:
    """Delete month-folder PDFs whose bill number is no longer in fin_*."""
    live = live_new_billnos(year, month, db_url=db_url)
    removed: list[Path] = []
    for kind in ("TAR", "3TAR"):
        pdf_dir = month_pdf_dir(kind, year, month)
        if not pdf_dir.is_dir():
            continue
        for path in sorted(pdf_dir.glob("*.pdf")):
            if path.stem in live:
                continue
            removed.append(path)
            print(f"[tar-report] stale PDF {'would remove' if dry_run else 'remove'}: {path.name}")
            if not dry_run:
                path.unlink(missing_ok=True)
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


def run_tar_reports(
    start: DateLike,
    end: Optional[DateLike] = None,
    *,
    prune_stale: bool = True,
    db_url: Optional[str] = None,
) -> dict[str, object]:
    dates = iter_dates(start, end or start)
    summaries = []
    for d in dates:
        summaries.append({"date": d.isoformat(), **output_tar_report_daily(d, db_url=db_url)})
    pruned: list[str] = []
    if prune_stale and dates:
        year, month = dates[0].year, dates[0].month
        pruned = [str(p) for p in prune_stale_pdfs(year, month, db_url=db_url)]
    return {"days": summaries, "pruned": pruned}
