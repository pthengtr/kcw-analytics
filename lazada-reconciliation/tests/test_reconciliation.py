"""Reconciliation scenarios using synthetic workbooks."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from src.loaders import load_finance, load_orders, load_wallet
from src.reconciliation import run_reconciliation
from tests.helpers import write_finance_overview, write_finance_txn, write_orders, write_wallet


def _dirs(tmp_path: Path) -> tuple[Path, Path, Path]:
    orders = tmp_path / "orders"
    finance = tmp_path / "finance"
    wallet = tmp_path / "wallet"
    orders.mkdir()
    finance.mkdir()
    wallet.mkdir()
    return orders, finance, wallet


def _run(orders_dir: Path, finance_dir: Path, wallet_dir: Path, tolerance: str = "0.01"):
    orders = load_orders(orders_dir)
    finance = load_finance(finance_dir)
    wallet = load_wallet(wallet_dir)
    return run_reconciliation(
        orders,
        finance,
        wallet,
        month="2026-08",
        generated_at="2026-08-25 00:00:00",
        tolerance=Decimal(tolerance),
    )


def _wallet_minimal(path: Path) -> None:
    write_wallet(
        path,
        [["T1", "01 Aug 2026", "Deposit", "Settlement", "+100.00", "Statement"]],
    )


def test_one_order_multiple_lines(tmp_path: Path):
    orders_dir, finance_dir, wallet_dir = _dirs(tmp_path)
    write_orders(
        orders_dir / "orders.xlsx",
        [
            ["I1", "1001", "100.00", "confirmed", "01 Aug 2026"],
            ["I2", "1001", "50.00", "confirmed", "01 Aug 2026"],
        ],
    )
    write_finance_overview(
        finance_dir / "finance.xlsx",
        [["1001", "01 Aug 2026", "I1", "SKU-A", "150.00", "0", "ยืนยันแล้ว", "p", "B1", "โอนเงินไปยังยอดของฉันแล้ว", "SC"]],
    )
    _wallet_minimal(wallet_dir / "wallet.xlsx")
    result = _run(orders_dir, finance_dir, wallet_dir)
    row = result.order_summary.iloc[0]
    assert row["order_number"] == "1001"
    assert row["gross_order_amount"] == Decimal("150.00")
    assert row["source_line_count"] == 2
    assert row["mixed_status"] == False


def test_order_not_in_finance(tmp_path: Path):
    orders_dir, finance_dir, wallet_dir = _dirs(tmp_path)
    write_orders(orders_dir / "orders.xlsx", [["I1", "1001", "100.00", "confirmed", "01 Aug 2026"]])
    write_finance_overview(
        finance_dir / "finance.xlsx",
        [["2002", "01 Aug 2026", "I9", "SKU", "80.00", "0", "ยืนยันแล้ว", "p", "B1", "โอนเงินไปยังยอดของฉันแล้ว", "SC"]],
    )
    _wallet_minimal(wallet_dir / "wallet.xlsx")
    result = _run(orders_dir, finance_dir, wallet_dir)
    statuses = set(result.order_vs_finance["match_status"])
    assert "ORDER_NOT_RELEASED" in statuses
    assert "FINANCE_FROM_OTHER_PERIOD" in statuses


def test_finance_not_in_order(tmp_path: Path):
    orders_dir, finance_dir, wallet_dir = _dirs(tmp_path)
    write_orders(orders_dir / "orders.xlsx", [["I1", "1001", "10.00", "canceled", "01 Aug 2026"]])
    write_finance_overview(
        finance_dir / "finance.xlsx",
        [["9999", "01 Aug 2026", "I9", "SKU", "80.00", "0", "ยืนยันแล้ว", "p", "B1", "โอนเงินไปยังยอดของฉันแล้ว", "SC"]],
    )
    _wallet_minimal(wallet_dir / "wallet.xlsx")
    result = _run(orders_dir, finance_dir, wallet_dir)
    finance_only = result.order_vs_finance[result.order_vs_finance["order_number"] == "9999"].iloc[0]
    assert finance_only["match_status"] == "FINANCE_FROM_OTHER_PERIOD"
    assert finance_only["in_orders"] == False


def test_difference_within_and_beyond_tolerance(tmp_path: Path):
    orders_dir, finance_dir, wallet_dir = _dirs(tmp_path)
    write_orders(
        orders_dir / "orders.xlsx",
        [
            ["I1", "1001", "100.00", "confirmed", "01 Aug 2026"],
            ["I2", "1002", "100.00", "confirmed", "01 Aug 2026"],
        ],
    )
    write_finance_txn(
        finance_dir / "finance.xlsx",
        [
            ["1001", "Item price", "99.99", "0", "โอนเงินไปยังยอดของฉันแล้ว", "ยืนยันแล้ว"],
            ["1002", "Item price", "99.98", "0", "โอนเงินไปยังยอดของฉันแล้ว", "ยืนยันแล้ว"],
        ],
    )
    _wallet_minimal(wallet_dir / "wallet.xlsx")
    result = _run(orders_dir, finance_dir, wallet_dir, tolerance="0.01")
    by_order = result.order_vs_finance.set_index("order_number")
    assert by_order.loc["1001", "match_status"] == "MATCHED"
    assert by_order.loc["1002", "match_status"] == "AMOUNT_MISMATCH"


def test_cancelled_order(tmp_path: Path):
    orders_dir, finance_dir, wallet_dir = _dirs(tmp_path)
    write_orders(orders_dir / "orders.xlsx", [["I1", "1001", "100.00", "canceled", "01 Aug 2026"]])
    write_finance_overview(
        finance_dir / "finance.xlsx",
        [["2002", "01 Aug 2026", "I9", "SKU", "10.00", "0", "ยืนยันแล้ว", "p", "B1", "โอนเงินไปยังยอดของฉันแล้ว", "SC"]],
    )
    _wallet_minimal(wallet_dir / "wallet.xlsx")
    result = _run(orders_dir, finance_dir, wallet_dir)
    row = result.order_vs_finance[result.order_vs_finance["order_number"] == "1001"].iloc[0]
    assert row["match_status"] == "CANCELLED_OR_REFUNDED"


def test_refund_and_shipping_fee_voucher(tmp_path: Path):
    orders_dir, finance_dir, wallet_dir = _dirs(tmp_path)
    write_orders(orders_dir / "orders.xlsx", [["I1", "1001", "200.00", "confirmed", "01 Aug 2026"]])
    write_finance_txn(
        finance_dir / "finance.xlsx",
        [
            ["1001", "Item price", "200.00", "0", "โอนเงินไปยังยอดของฉันแล้ว", "ยืนยันแล้ว"],
            ["1001", "Refund", "-20.00", "0", "โอนเงินไปยังยอดของฉันแล้ว", "ยืนยันแล้ว"],
            ["1001", "Shipping fee Voucher Refund to Lazada", "-15.00", "0", "โอนเงินไปยังยอดของฉันแล้ว", "ยืนยันแล้ว"],
        ],
    )
    _wallet_minimal(wallet_dir / "wallet.xlsx")
    result = _run(orders_dir, finance_dir, wallet_dir)
    fin = result.finance_summary.iloc[0]
    assert fin["finance_income"] == Decimal("200.00")
    assert fin["refund"] == Decimal("-20.00")
    assert fin["shipping_fee"] == Decimal("-15.00")
    assert fin["finance_net_amount"] == Decimal("165.00")
    cats = set(result.finance_details["mapped_category"])
    assert "shipping_fee" in cats
    assert "refund" in cats


def test_unknown_freename_not_dropped(tmp_path: Path):
    orders_dir, finance_dir, wallet_dir = _dirs(tmp_path)
    write_orders(orders_dir / "orders.xlsx", [["I1", "1001", "50.00", "confirmed", "01 Aug 2026"]])
    write_finance_txn(
        finance_dir / "finance.xlsx",
        [
            ["1001", "Item price", "50.00", "0", "โอนเงินไปยังยอดของฉันแล้ว", "ยืนยันแล้ว"],
            ["1001", "Mystery Galactic Fee XYZ", "-3.25", "0", "โอนเงินไปยังยอดของฉันแล้ว", "ยืนยันแล้ว"],
        ],
    )
    _wallet_minimal(wallet_dir / "wallet.xlsx")
    result = _run(orders_dir, finance_dir, wallet_dir)
    assert (result.finance_details["mapped_category"] == "unknown").any()
    assert not result.unknown_mappings.empty
    assert "Mystery Galactic Fee XYZ" in set(result.unknown_mappings["raw_value"])


def test_auto_withdrawal_and_opposite_sign(tmp_path: Path):
    orders_dir, finance_dir, wallet_dir = _dirs(tmp_path)
    write_orders(orders_dir / "orders.xlsx", [["I1", "1001", "10.00", "confirmed", "01 Aug 2026"]])
    write_finance_overview(
        finance_dir / "finance.xlsx",
        [["1001", "01 Aug 2026", "I1", "SKU", "8.00", "0", "ยืนยันแล้ว", "p", "B1", "โอนเงินไปยังยอดของฉันแล้ว", "SC"]],
    )
    write_wallet(
        wallet_dir / "wallet.xlsx",
        [
            ["T1", "01 Aug 2026", "Deposit", "Settlement", "+80.00", "Statement"],
            ["T2", "02 Aug 2026", "Withdrawal", "Auto Withdrawal", "-50.00", "Paid"],
            ["T3", "03 Aug 2026", "Withdrawal", "Auto Withdrawal", "25.00", "Paid opposite sign"],
        ],
    )
    result = _run(orders_dir, finance_dir, wallet_dir)
    assert result.wallet_totals["opening_balance_found"] is False
    assert result.wallet_totals["calculated_closing_balance"] == "ข้อมูลไม่เพียงพอ"
    auto = result.wallet_details[result.wallet_details["is_auto_withdrawal"]]
    assert len(auto) == 2
    assert Decimal("-50.00") in set(auto["signed_amount"])
    assert Decimal("25.00") in set(auto["signed_amount"])


def test_net_settled_for_overview_profile(tmp_path: Path):
    orders_dir, finance_dir, wallet_dir = _dirs(tmp_path)
    write_orders(orders_dir / "orders.xlsx", [["I1", "1001", "100.00", "confirmed", "01 Aug 2026"]])
    write_finance_overview(
        finance_dir / "finance.xlsx",
        [["1001", "01 Aug 2026", "I1", "SKU", "75.00", "0", "ยืนยันแล้ว", "p", "B1", "โอนเงินไปยังยอดของฉันแล้ว", "SC"]],
    )
    _wallet_minimal(wallet_dir / "wallet.xlsx")
    result = _run(orders_dir, finance_dir, wallet_dir)
    row = result.order_vs_finance.iloc[0]
    assert row["match_status"] == "NET_SETTLED"
    assert row["order_to_finance_difference"] == Decimal("25.00")
    assert result.finance_profile == "finance_overview"


def test_full_outer_join_keeps_all_keys(tmp_path: Path):
    orders_dir, finance_dir, wallet_dir = _dirs(tmp_path)
    write_orders(
        orders_dir / "orders.xlsx",
        [
            ["I1", "A", "10.00", "confirmed", "01 Aug 2026"],
            ["I2", "B", "20.00", "confirmed", "01 Aug 2026"],
        ],
    )
    write_finance_overview(
        finance_dir / "finance.xlsx",
        [
            ["B", "01 Aug 2026", "I2", "SKU", "15.00", "0", "ยืนยันแล้ว", "p", "B1", "โอนเงินไปยังยอดของฉันแล้ว", "SC"],
            ["C", "01 Aug 2026", "I3", "SKU", "9.00", "0", "ยืนยันแล้ว", "p", "B1", "โอนเงินไปยังยอดของฉันแล้ว", "SC"],
        ],
    )
    _wallet_minimal(wallet_dir / "wallet.xlsx")
    result = _run(orders_dir, finance_dir, wallet_dir)
    keys = set(result.order_vs_finance["order_number"])
    assert keys == {"A", "B", "C"}
