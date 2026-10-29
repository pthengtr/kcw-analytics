import pandas as pd

from src.kcw.vat_sales_amounts import (
    canceled_billnos_from_simas,
    zero_canceled_sales_amounts,
)


def test_canceled_billnos_from_simas_keeps_only_y():
    simas = pd.DataFrame(
        {
            "BILLNO": [" TD6910-115 ", "TAD6910-001", "TD6910-116", None],
            "CANCELED": ["y", " N ", "Y", "Y"],
        }
    )
    assert canceled_billnos_from_simas([simas, pd.DataFrame()]) == {
        "TD6910-115",
        "TD6910-116",
    }


def test_zero_canceled_sales_amounts_from_flag():
    df = pd.DataFrame(
        {
            "BILLNO": ["TD6910-115", "TD6910-116"],
            "CANCELED": ["Y", "N"],
            "BEFORETAX": [440.0, 100.0],
            "TAX": [30.8, 7.0],
            "AFTERTAX": [470.8, 107.0],
            "DETAIL": ["สีสเปรย์", "กรอง"],
        }
    )
    out = zero_canceled_sales_amounts(df)
    canceled = out.loc[out["BILLNO"].eq("TD6910-115")].iloc[0]
    kept = out.loc[out["BILLNO"].eq("TD6910-116")].iloc[0]
    assert canceled["BEFORETAX"] == 0
    assert canceled["TAX"] == 0
    assert canceled["AFTERTAX"] == 0
    assert canceled["DETAIL"] == "สีสเปรย์"
    assert kept["BEFORETAX"] == 100.0
    assert kept["AFTERTAX"] == 107.0


def test_zero_canceled_sales_amounts_from_simas_billnos():
    df = pd.DataFrame(
        {
            "BILLNO": ["tad6901-015", "TD6901-001"],
            "BEFORETAX": [1210.28, 50],
            "TAX": [84.72, 3.5],
            "AFTERTAX": [1295.0, 53.5],
            "BEFORE_VAT": [1210.28, 50],
            "TOTAL_AMOUNT": [1295.0, 53.5],
        }
    )
    out = zero_canceled_sales_amounts(df, {"TAD6901-015"})
    row = out.iloc[0]
    assert row["BEFORETAX"] == 0
    assert row["TAX"] == 0
    assert row["AFTERTAX"] == 0
    assert row["BEFORE_VAT"] == 0
    assert row["TOTAL_AMOUNT"] == 0
    assert out.iloc[1]["BEFORETAX"] == 50


def test_zero_canceled_sales_amounts_leaves_open_bills_untouched():
    df = pd.DataFrame({"BILLNO": ["TD6910-001"], "BEFORETAX": [3600.0], "TAX": [252.0]})
    out = zero_canceled_sales_amounts(df, set())
    assert out is df
    assert out.iloc[0]["BEFORETAX"] == 3600.0
