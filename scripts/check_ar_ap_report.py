"""Validate AR/AP outstanding report logic from notebooks/33_ar_ap_report.ipynb."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kcw.utils import _parse_billdate  # noqa: E402


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


def test_ar_scenarios():
    cutoff = pd.Timestamp("2026-03-01")
    df = pd.DataFrame(
        [
            make_row(BILLNO="TD6901-001", PAID="N", VOUCDATE2=None),  # unpaid -> include
            make_row(BILLNO="TD6901-002", PAID="Y", VOUCDATE2="2026-02-20"),  # paid before cutoff -> exclude
            make_row(BILLNO="TD6901-003", PAID="Y", VOUCDATE2="2026-03-05"),  # paid after cutoff -> include
            make_row(BILLNO="TD6901-004", PAID="Y", VOUCDATE2=None),  # paid flag but no date -> include
            make_row(BILLNO="TD6901-005", DUEAMT=0),  # zero due -> exclude
            make_row(BILLNO="CN6901-001", DUEAMT=-500),  # credit note -> include
            make_row(BILLNO="IV6901-001"),  # non AR bill type -> exclude
            make_row(BILLNO="TD6903-001", BILLDATE="2026-03-05"),  # after cutoff -> exclude
            make_row(BILLNO="TD6901-006", CANCELED="Y"),  # canceled -> exclude
            make_row(BILLNO="TD6901-007", VAT=0),  # non-vat -> exclude
            make_row(BILLNO="TD6901-008", ACCTNAME="Shop on Shopee marketplace"),  # normalize
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
    """Notebook uses naive to_datetime; utils _parse_billdate handles ISO safely."""
    cutoff = pd.Timestamp("2026-03-01")
    iso_date = "2026-02-28"
    slash_date = "28/02/2026"

    row = make_row(BILLDATE=iso_date)
    df = pd.DataFrame([row])

    notebook_parsed = pd.to_datetime(df["BILLDATE"].astype(str).str.strip(), errors="coerce").iloc[0]
    utils_parsed = _parse_billdate(df["BILLDATE"]).iloc[0]

    assert pd.notna(notebook_parsed), "ISO date should parse in notebook logic"
    assert notebook_parsed == utils_parsed

    df2 = pd.DataFrame([make_row(BILLDATE=slash_date)])
    nb2 = pd.to_datetime(df2["BILLDATE"].astype(str).str.strip(), errors="coerce", dayfirst=True).iloc[0]
    ut2 = _parse_billdate(df2["BILLDATE"]).iloc[0]
    assert pd.notna(ut2)
    assert nb2 == ut2 == pd.Timestamp("2026-02-28")
    print("BILLDATE parsing comparison passed")


def test_net_zero_summary_noise():
    cutoff = pd.Timestamp("2026-03-01")
    df = pd.DataFrame(
        [
            make_row(BILLNO="TD6901-010", DUEAMT=1000),
            make_row(BILLNO="CN6901-010", DUEAMT=-1000),
        ]
    )
    out = filter_ar(df, cutoff)
    summary = out.groupby("ACCTNAME").agg(bills=("BILLNO", "nunique"), total=("DUEAMT", "sum")).reset_index()
    assert summary.iloc[0]["total"] == 0
    assert summary.iloc[0]["bills"] == 2
    print("Net-zero pair kept in detail/summary (known behavior)")


if __name__ == "__main__":
    test_ar_scenarios()
    test_ap_scenarios()
    test_billdate_parsing_gap()
    test_net_zero_summary_noise()
    print("\nAll AR/AP checks passed.")
