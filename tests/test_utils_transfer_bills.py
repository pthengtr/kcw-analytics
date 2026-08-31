import pandas as pd

from src.kcw.utils import (
    exclude_transfer_stock_lines,
    get_nonvat_sales_lines_last_purchase_vat,
    get_vat_sales_lines_last_purchase_nonvat,
    is_transfer_stock_billno,
)


def test_is_transfer_stock_billno_matches_tf_prefixes_only():
    bills = pd.Series(
        [
            "TF6908-0098",
            "3TF6908-0001",
            "TFV6908-0001",
            "8K69-0001",
            "TAD6908-001",
            "PI6908-001",
        ]
    )
    assert is_transfer_stock_billno(bills).tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_last_purchase_nonvat_sales_ignores_transfer_purchase():
    bcode = "03018420"
    data = {
        "raw_hq_sidet_sales_lines.csv": pd.DataFrame(
            {
                "BCODE": [bcode],
                "BILLDATE": ["2025-08-15"],
                "ISVAT": ["N"],
                "CANCELED": ["N"],
                "BILLNO": ["8K69-0001"],
            }
        ),
        "raw_hq_pidet_purchase_lines.csv": pd.DataFrame(
            {
                "BCODE": [bcode, bcode],
                "BILLDATE": ["2025-08-10", "2025-08-12"],
                "ISVAT": ["N", "Y"],
                "BILLNO": ["PI6908-001", "TF6908-0098"],
            }
        ),
    }

    out = get_nonvat_sales_lines_last_purchase_vat(data, year=2025, source="hq")
    assert out.empty


def test_last_purchase_vat_sales_ignores_transfer_purchase():
    bcode = "03018420"
    data = {
        "raw_hq_sidet_sales_lines.csv": pd.DataFrame(
            {
                "BCODE": [bcode],
                "BILLDATE": ["2025-08-15"],
                "ISVAT": ["Y"],
                "CANCELED": ["N"],
                "BILLNO": ["8K69-0001"],
            }
        ),
        "raw_hq_pidet_purchase_lines.csv": pd.DataFrame(
            {
                "BCODE": [bcode, bcode],
                "BILLDATE": ["2025-08-10", "2025-08-12"],
                "ISVAT": ["Y", "N"],
                "BILLNO": ["PI6908-001", "3TF6908-0001"],
            }
        ),
    }

    out = get_vat_sales_lines_last_purchase_nonvat(data, year=2025, source="hq")
    assert out.empty


def test_exclude_transfer_stock_lines():
    df = pd.DataFrame({"BILLNO": ["TF6908-001", "8K69-0001"]})
    out = exclude_transfer_stock_lines(df)
    assert list(out["BILLNO"]) == ["8K69-0001"]
