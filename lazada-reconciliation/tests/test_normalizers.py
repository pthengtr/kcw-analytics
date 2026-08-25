"""Tests for money parsing, order numbers, headers, and loaders."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from src.file_detector import FileDetectionError, detect_excel_file, normalize_header
from src.loaders import load_orders, load_wallet
from src.normalizers import MoneyParseError, decimal_sum, normalize_order_number, parse_money
from tests.helpers import write_orders, write_sheet, write_wallet


def test_parse_money_comma():
    assert parse_money("+5,044.55") == Decimal("5044.55")
    assert parse_money("-25,953.71") == Decimal("-25953.71")


def test_parse_money_parentheses_negative():
    assert parse_money("(1,234.50)") == Decimal("-1234.50")


def test_parse_money_empty_is_null_not_zero():
    assert parse_money("") is None
    assert parse_money(None) is None
    assert parse_money("   ") is None


def test_parse_money_currency_and_spaces():
    assert parse_money("฿ 100.10") == Decimal("100.10")
    assert parse_money("THB 20") == Decimal("20.00")


def test_parse_money_invalid_raises():
    with pytest.raises(MoneyParseError):
        parse_money("not-a-number")


def test_order_number_strips_excel_float_artifact():
    assert normalize_order_number("1114692470339693.0") == "1114692470339693"
    assert normalize_order_number(1114692470339693) == "1114692470339693"
    assert normalize_order_number(1114692470339693.0) == "1114692470339693"


def test_order_number_keeps_leading_zeros_when_string():
    assert normalize_order_number("000123") == "000123"


def test_normalize_header_trim_and_case():
    assert normalize_header("  OrderNumber\t") == "ordernumber"
    assert normalize_header("จำนวนเงิน(รวมภาษี)") == "จำนวนเงิน(รวมภาษี)"


def test_header_not_on_first_row(tmp_path: Path):
    path = write_orders(
        tmp_path / "orders.xlsx",
        [["I1", "1001", "10.00", "confirmed", "01 Aug 2026"]],
        title_rows=[["Lazada Seller Center"], ["รายงานคำสั่งซื้อ เดือนทดสอบ"]],
    )
    detected = detect_excel_file(path)
    assert detected.file_type == "orders"
    assert detected.header_row_index == 2
    loaded = load_orders(tmp_path)
    assert loaded.frame.iloc[0]["order_number"] == "1001"
    assert loaded.frame.iloc[0]["source_row"] == 4


def test_grand_total_excluded(tmp_path: Path):
    write_orders(
        tmp_path / "orders.xlsx",
        [
            ["I1", "1001", "10.00", "confirmed", "01 Aug 2026"],
            ["", "Grand Total", "999.00", "", ""],
        ],
    )
    loaded = load_orders(tmp_path)
    assert len(loaded.frame) == 1
    assert loaded.dropped_total_rows == 1
    assert loaded.frame.iloc[0]["order_number"] == "1001"


def test_product_named_total_is_not_treated_as_grand_total(tmp_path: Path):
    write_sheet(
        tmp_path / "orders.xlsx",
        ["orderItemId", "orderNumber", "paidPrice", "status", "createTime", "itemName"],
        [
            ["I1", "1001", "10.00", "confirmed", "01 Aug 2026", "Total"],
            ["I2", "1002", "20.00", "confirmed", "01 Aug 2026", "ปะเก็นรวม"],
            ["", "Grand Total", "999.00", "", "", ""],
        ],
    )
    loaded = load_orders(tmp_path)
    assert loaded.dropped_total_rows == 1
    assert set(loaded.frame["order_number"]) == {"1001", "1002"}
    assert decimal_sum(loaded.frame["signed_amount"]) == Decimal("30.00")


def test_signed_amount_is_decimal_not_float(tmp_path: Path):
    write_orders(tmp_path / "orders.xlsx", [["I1", "1001", 10.1, "confirmed", "01 Aug 2026"]])
    loaded = load_orders(tmp_path)
    value = loaded.frame.iloc[0]["signed_amount"]
    assert isinstance(value, Decimal)
    assert not isinstance(value, float)


def test_pii_columns_dropped_from_loaded_orders(tmp_path: Path):
    write_sheet(
        tmp_path / "orders.xlsx",
        [
            "orderNumber",
            "paidPrice",
            "status",
            "customerName",
            "shippingPhone",
            "billingPostCode",
        ],
        [["1001", "10.00", "confirmed", "นายทดสอบ ไม่ใช่ลูกค้าจริง", "0800000000", "10110"]],
    )
    loaded = load_orders(tmp_path)
    columns = {str(c).casefold() for c in loaded.frame.columns}
    assert "customername" not in columns
    assert "shippingphone" not in columns
    assert "billingpostcode" not in columns
    joined = loaded.frame.astype(str).to_string()
    assert "นายทดสอบ" not in joined
    assert "0800000000" not in joined
    write_orders(
        tmp_path / "orders.xlsx",
        [
            ["I1", "1001", "10.00", "confirmed", "01 Aug 2026"],
            ["", "Grand Total", "999.00", "", ""],
        ],
    )
    loaded = load_orders(tmp_path)
    assert len(loaded.frame) == 1
    assert loaded.dropped_total_rows == 1
    assert loaded.frame.iloc[0]["order_number"] == "1001"


def test_order_number_numeric_excel_cell(tmp_path: Path):
    write_orders(
        tmp_path / "orders.xlsx",
        [["I1", 1114692470339693, "10.00", "confirmed", "01 Aug 2026"]],
    )
    loaded = load_orders(tmp_path)
    assert loaded.frame.iloc[0]["order_number"] == "1114692470339693"
    assert isinstance(loaded.frame.iloc[0]["order_number"], str)


def test_unknown_file_headers_raise(tmp_path: Path):
    write_sheet(tmp_path / "weird.xlsx", ["foo", "bar"], [["a", "b"]])
    with pytest.raises(FileDetectionError) as exc:
        detect_excel_file(tmp_path / "weird.xlsx")
    assert "weird.xlsx" in exc.value.message_th
    assert "header ที่ตรวจพบ" in exc.value.message_th


def test_wallet_uses_amount_not_subtype(tmp_path: Path):
    write_wallet(
        tmp_path / "wallet.xlsx",
        [
            ["T1", "01 Aug 2026", "Withdrawal", "Auto Withdrawal", "-1,000.00", "Paid"],
            ["T2", "02 Aug 2026", "Deposit", "Settlement", "+800.00", "Statement"],
        ],
    )
    loaded = load_wallet(tmp_path)
    assert loaded.frame.iloc[0]["signed_amount"] == Decimal("-1000.00")
    assert loaded.frame.iloc[1]["signed_amount"] == Decimal("800.00")


def test_multiple_files_and_duplicates(tmp_path: Path):
    write_orders(tmp_path / "a.xlsx", [["I1", "1001", "10.00", "confirmed", "01 Aug 2026"]])
    write_orders(tmp_path / "b.xlsx", [["I1", "1001", "10.00", "confirmed", "01 Aug 2026"]])
    loaded = load_orders(tmp_path)
    assert len(loaded.frame) == 2
    assert {p for p in loaded.frame["source_file"]} == {"a.xlsx", "b.xlsx"}
    dup_issues = [i for i in loaded.issues if i.exception_type == "DUPLICATE_ROW"]
    assert dup_issues
