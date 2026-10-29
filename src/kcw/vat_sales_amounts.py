"""Canceled tax invoices stay on the sales VAT book with zero amounts."""

from __future__ import annotations

import pandas as pd

SALES_TAX_AMOUNT_COLS = (
    "BEFORETAX",
    "TAX",
    "AFTERTAX",
    "BEFORE_VAT",
    "VAT_AMOUNT",
    "TOTAL_AMOUNT",
    "TOTAL_INCL_VAT",
)


def canceled_billnos_from_simas(frames) -> set[str]:
    """Bill numbers whose SIMAS header is CANCELED=Y."""
    bills: set[str] = set()
    for frame in frames:
        if frame is None or frame.empty:
            continue
        if "BILLNO" not in frame.columns or "CANCELED" not in frame.columns:
            continue
        mask = frame["CANCELED"].astype("string").str.strip().str.upper().eq("Y")
        bills.update(
            frame.loc[mask, "BILLNO"].astype("string").str.strip().str.upper().dropna()
        )
    bills.discard("")
    bills.discard("<NA>")
    return bills


def zero_canceled_sales_amounts(
    df: pd.DataFrame,
    canceled_bills: set[str] | None = None,
    *,
    billno_col: str = "BILLNO",
    canceled_col: str = "CANCELED",
) -> pd.DataFrame:
    """Keep a canceled sales bill on the VAT report and set its tax amounts to 0.

    A bill is canceled when `canceled_col` is Y, or when `billno_col` is in
    `canceled_bills` (SIMAS header lookup). Other columns stay as they are.
    """
    if df is None or df.empty:
        return df

    mask = pd.Series(False, index=df.index)
    if canceled_col in df.columns:
        mask = mask | df[canceled_col].astype("string").str.strip().str.upper().eq("Y")
    if canceled_bills and billno_col in df.columns:
        keys = df[billno_col].astype("string").str.strip().str.upper()
        mask = mask | keys.isin(canceled_bills)
    if not bool(mask.any()):
        return df

    out = df.copy()
    for col in SALES_TAX_AMOUNT_COLS:
        if col not in out.columns:
            continue
        out.loc[mask, col] = 0
    return out
