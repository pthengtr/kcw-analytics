"""Validate AR/AP outstanding report logic from notebooks/33_ar_ap_report.ipynb."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kcw.utils import _parse_billdate  # noqa: E402


def report_cutoff() -> pd.Timestamp:
    """Match notebook logic after cutoff-date fix."""
    return pd.Timestamp.today().normalize().replace(day=1)


def normalize_acctname(name):
    if pd.isna(name):
        return name
    n = str(name).lower()
    for k in ["lazada", "shopee", "tiktok"]:
        if k in n:
            return f"คุณลูกค้าทั่วไป {k}"
    return name


def filter_ar(df_simas: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    df = df_simas.copy()
    df["BILLDATE"] = pd.to_datetime(df["BILLDATE"].astype(str).str.strip(), errors="coerce")
    df["VOUCDATE2"] = pd.to_datetime(df["VOUCDATE2"], errors="coerce")

    mask_billno = df["BILLNO"].str.contains(r"TD|TR|TAD|CN", na=False)
    mask_due_pos = df["DUEAMT"] != 0
    mask_billbefore = df["BILLDATE"] < cutoff
    mask_paidafter = df["VOUCDATE2"] >= cutoff
    mask_null = df["VOUCDATE2"].isna()
    mask_cancel = df["CANCELED"].astype(str).str.strip().str.upper() == "N"
    mask_vat = pd.to_numeric(df["VAT"], errors="coerce").fillna(0) > 0

    df["ACCTNAME"] = df["ACCTNAME"].apply(normalize_acctname)
    return df[
        mask_billno
        & mask_due_pos
        & mask_billbefore
        & (mask_null | mask_paidafter)
        & mask_cancel
        & mask_vat
    ].sort_values(by=["ACCTNAME", "BILLDATE"])


def filter_ap(df_pimas: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    df = df_pimas.copy()
    df["BILLDATE"] = pd.to_datetime(df["BILLDATE"].astype(str).str.strip(), errors="coerce")
    df["VOUCDATE2"] = pd.to_datetime(df["VOUCDATE2"], errors="coerce")

    mask_due_pos = df["DUEAMT"] != 0
    mask_billbefore = df["BILLDATE"] < cutoff
    mask_paidafter = df["VOUCDATE2"] >= cutoff
    mask_null = df["VOUCDATE2"].isna()
    mask_cancel = df["CANCELED"].astype(str).str.strip().str.upper() == "N"
    mask_vat = pd.to_numeric(df["VAT"], errors="coerce").fillna(0) > 0

    df["ACCTNAME"] = df["ACCTNAME"].apply(normalize_acctname)
    return df[
        mask_due_pos
        & mask_billbefore
        & (mask_null | mask_paidafter)
        & mask_cancel
        & mask_vat
    ].sort_values(by=["ACCTNAME", "BILLDATE"])


def make_row(**kwargs):
    base = {
        "BILLNO": "TD6901-001",
        "BILLDATE": "2026-01-15",
        "ACCTNO": "7TEST",
        "ACCTNAME": "Test Customer",
        "VOUCDATE1": None,
        "VOUCDATE2": None,
        "PAID": "N",
        "DUEAMT": 1000.0,
        "VAT": 7.0,
        "CANCELED": "N",
    }
    base.update(kwargs)
    return base


def test_cutoff_excludes_first_of_month_at_midnight():
    """Regression: April 1 00:00 must not appear in the as-of-March-31 report."""
    cutoff_fixed = pd.Timestamp("2026-04-01")
    cutoff_buggy = pd.Timestamp("2026-04-01 17:57:14")
    april_first = pd.Timestamp("2026-04-01 00:00:00")

    assert april_first < cutoff_buggy, "sanity check for old buggy behavior"
    assert not (april_first < cutoff_fixed), "fixed cutoff must exclude April 1 midnight"

    df = pd.DataFrame([make_row(BILLNO="TD6904-001", BILLDATE="2026-04-01")])
    out = filter_ar(df, cutoff_fixed)
    assert out.empty, "April 1 bill should be excluded from March-end report"

    march_last = pd.Timestamp("2026-03-31 00:00:00")
    df_march = pd.DataFrame([make_row(BILLNO="TD6903-031", BILLDATE="2026-03-31")])
    out_march = filter_ar(df_march, cutoff_fixed)
    assert len(out_march) == 1, "March 31 bill should remain included"

    assert report_cutoff().hour == 0
    assert report_cutoff().minute == 0
    print("Cutoff first-of-month regression passed")


def test_ar_scenarios():
    cutoff = pd.Timestamp("2026-03-01")
    df = pd.DataFrame(
        [
            make_row(BILLNO="TD6901-001", PAID="N", VOUCDATE2=None),
            make_row(BILLNO="TD6901-002", PAID="Y", VOUCDATE2="2026-02-20"),
            make_row(BILLNO="TD6901-003", PAID="Y", VOUCDATE2="2026-03-05"),
            make_row(BILLNO="TD6901-004", PAID="Y", VOUCDATE2=None),
            make_row(BILLNO="TD6901-005", DUEAMT=0),
            make_row(BILLNO="CN6901-001", DUEAMT=-500),
            make_row(BILLNO="IV6901-001"),
            make_row(BILLNO="TD6903-001", BILLDATE="2026-03-05"),
            make_row(BILLNO="TD6901-006", CANCELED="Y"),
            make_row(BILLNO="TD6901-007", VAT=0),
            make_row(BILLNO="TD6901-008", ACCTNAME="Shop on Shopee marketplace"),
        ]
    )

    out = filter_ar(df, cutoff)
    billnos = set(out["BILLNO"])
    expected = {
        "TD6901-001",
        "TD6901-003",
        "TD6901-004",
        "CN6901-001",
        "TD6901-008",
    }
    assert billnos == expected, f"AR mismatch: got {billnos}, expected {expected}"
    assert out.loc[out["BILLNO"] == "TD6901-008", "ACCTNAME"].iloc[0] == "คุณลูกค้าทั่วไป shopee"
    print("AR scenario tests passed")


def test_ap_scenarios():
    cutoff = pd.Timestamp("2026-03-01")
    df = pd.DataFrame(
        [
            make_row(BILLNO="IV6901-001", PAID="N", VOUCDATE2=None),
            make_row(BILLNO="IV6901-002", PAID="Y", VOUCDATE2="2026-02-28"),
            make_row(BILLNO="IV6901-003", PAID="Y", VOUCDATE2="2026-03-10"),
            make_row(BILLNO="IV6901-004", PAID="Y", VOUCDATE2=None),
            make_row(BILLNO="IV6901-005", DUEAMT=0),
            make_row(BILLNO="IV6903-001", BILLDATE="2026-03-02"),
        ]
    )

    out = filter_ap(df, cutoff)
    billnos = set(out["BILLNO"])
    expected = {"IV6901-001", "IV6901-003", "IV6901-004"}
    assert billnos == expected, f"AP mismatch: got {billnos}, expected {expected}"
    print("AP scenario tests passed")


def test_billdate_parsing_gap():
    iso_date = "2026-02-28"
    slash_date = "28/02/2026"

    row = make_row(BILLDATE=iso_date)
    df = pd.DataFrame([row])

    notebook_parsed = pd.to_datetime(df["BILLDATE"].astype(str).str.strip(), errors="coerce").iloc[0]
    utils_parsed = _parse_billdate(df["BILLDATE"]).iloc[0]

    assert pd.notna(notebook_parsed)
    assert notebook_parsed == utils_parsed

    df2 = pd.DataFrame([make_row(BILLDATE=slash_date)])
    nb2 = pd.to_datetime(df2["BILLDATE"].astype(str).str.strip(), errors="coerce", dayfirst=True).iloc[0]
    ut2 = _parse_billdate(df2["BILLDATE"]).iloc[0]
    assert pd.notna(ut2)
    assert nb2 == ut2 == pd.Timestamp("2026-02-28")
    print("BILLDATE parsing comparison passed")


if __name__ == "__main__":
    test_cutoff_excludes_first_of_month_at_midnight()
    test_ar_scenarios()
    test_ap_scenarios()
    test_billdate_parsing_gap()
    print("\nAll AR/AP checks passed.")
