import pandas as pd

from src.kcw.tar import _apply_day_filters


def _frame(billnos, bcodes=None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "BILLNO": billnos,
            "BCODE": bcodes or ["15010490"] * len(billnos),
        }
    )


def test_apply_day_filters_excludes_stock_adjustment_bills():
    df = _frame(
        [
            "8K69-0001",
            "SA2608-00050",
            "3SA2608-00001",
            "sa2608-00051",
            "TAD6908-001",
        ]
    )
    hq = _apply_day_filters(df, site="hq")
    syp = _apply_day_filters(df, site="syp")
    assert set(hq["BILLNO"]) == {"8K69-0001", "TAD6908-001"}
    assert set(syp["BILLNO"]) == {"8K69-0001", "TAD6908-001"}


def test_apply_day_filters_still_drops_tf_dn_and_service_bcodes():
    df = _frame(
        ["8K69-0001", "TF6908-001", "DN6908-001", "3DN6908-001", "8K69-0002"],
        ["15010490", "15010490", "15010490", "15010490", "70001234"],
    )
    hq = _apply_day_filters(df, site="hq")
    syp = _apply_day_filters(df, site="syp")
    assert list(hq["BILLNO"]) == ["8K69-0001", "3DN6908-001"]
    assert list(syp["BILLNO"]) == ["8K69-0001"]
