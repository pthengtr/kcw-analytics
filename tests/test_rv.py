import pandas as pd

from src.kcw.rv import (
    apply_cost_lines,
    apply_rv_day_filters,
    bill_prefix,
    catchup_start,
    max_issued_seq,
)
from datetime import date


def _frame(billnos, bcodes=None, **extra) -> pd.DataFrame:
    rows = {
        "BILLNO": billnos,
        "BCODE": bcodes or ["15010490"] * len(billnos),
    }
    rows.update(extra)
    return pd.DataFrame(rows)


def test_apply_rv_day_filters_drops_transfer_credit_and_service():
    df = _frame(
        ["8K69-0001", "TF6910-001", "CN6910-001", "3CN6910-001", "DN6910-001", "8K69-0002"],
        ["15010490", "15010490", "15010490", "15010490", "15010490", "7001"],
    )
    hq = apply_rv_day_filters(df, site="hq")
    syp = apply_rv_day_filters(df, site="syp")
    assert list(hq["BILLNO"]) == ["8K69-0001", "3CN6910-001"]
    assert list(syp["BILLNO"]) == ["8K69-0001", "CN6910-001", "DN6910-001"]


def test_apply_cost_lines_prices_at_last_cost_and_drops_zero_sale():
    df = pd.DataFrame(
        {
            "BCODE": ["A", "B", "C"],
            "AMOUNT": [100, 0, 50],
            "QTY": [2, 1, 1],
            "MTP": [1, 1, 4],
            "LAST_COST": [10.126, 5, 0],
        }
    )
    out = apply_cost_lines(df)
    assert list(out["BCODE"]) == ["A"]
    assert float(out["PRICE"].iloc[0]) == 10.13
    assert float(out["AMOUNT"].iloc[0]) == 20.26


def test_max_issued_seq_reads_bill_numbers_and_pdf_stems():
    prefix = bill_prefix("RV", date(2026, 10, 7))
    assert prefix == "RV6910-"
    assert max_issued_seq(["RV6910-001", "RV6910-008.pdf", "3RV6910-003"], prefix) == 8
    assert max_issued_seq(["3RV6910-003.pdf"], prefix) == 0


def test_catchup_start_continues_after_last_persisted_day():
    assert catchup_start(
        today=date(2026, 10, 8),
        eligible_end=date(2026, 10, 8),
        max_fin=date(2026, 10, 7),
    ) == date(2026, 10, 8)
    assert catchup_start(
        today=date(2026, 10, 7),
        eligible_end=date(2026, 10, 7),
        max_fin=date(2026, 10, 7),
    ) is None
    assert catchup_start(
        today=date(2026, 10, 7),
        eligible_end=None,
        max_fin=None,
    ) == date(2026, 10, 7)
