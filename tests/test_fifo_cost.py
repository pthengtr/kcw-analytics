from datetime import date

from src.kcw.fifo_cost import (
    CLEANUP_MOVED,
    CLEANUP_ON_HAND,
    CLEANUP_ZERO,
    STATUS_OK,
    STATUS_PARTIAL,
    STATUS_UNCOSTED,
    WINDOW_5Y,
    WINDOW_FULL,
    ReceiptLine,
    SaleLine,
    cleanup_decision,
    is_stock_count_bill,
    is_transfer_bill,
    rebuild,
    sale_for_cost,
    unit_cost_ex_vat,
)


def _r(line_id, year, month, day, units, cost, billno="PI"):
    return ReceiptLine(line_id, billno, date(year, month, day), units, cost, "SUP")


def _s(line_id, year, month, day, units, billno="SI"):
    return SaleLine(line_id, billno, date(year, month, day), units)


def test_newest_receipts_cover_on_hand_and_older_cost_the_sale():
    # 10 @ 100, then 10 @ 200. On hand 10. One sale of 8. Shrink of 2 stays off the sale.
    result = rebuild(
        10,
        [_r(1, 2024, 1, 1, 10, 100), _r(2, 2025, 6, 1, 10, 200)],
        [_s(9, 2025, 8, 1, 8)],
        as_of=date(2026, 10, 4),
    )
    assert result.history_window == WINDOW_5Y
    assert result.cost_complete is True
    assert result.on_hand_covered_qty == 10
    assert result.on_hand_value == 2000
    assert result.unassigned_qty == 2
    assert len(result.sales) == 1
    sale = result.sales[0]
    assert sale.status == STATUS_OK
    assert sale.fifo_unit_cost == 100
    assert sale.fifo_ext_cost == 800
    assert sale.allocs[0].pi_billno == "PI"
    assert sale.allocs[0].pidet_id == 1


def test_one_sale_blends_two_receipts():
    result = rebuild(
        5,
        [_r(1, 2024, 1, 1, 10, 100, "P1"), _r(2, 2025, 1, 1, 10, 300, "P2")],
        [_s(9, 2025, 6, 1, 12)],
        as_of=date(2026, 10, 4),
    )
    sale = result.sales[0]
    assert sale.status == STATUS_OK
    assert abs(sale.fifo_ext_cost - (10 * 100 + 2 * 300)) < 1e-6
    assert [row.pi_billno for row in sale.allocs] == ["P1", "P2"]
    assert result.unassigned_qty == 3


def test_stock_count_is_not_a_sale_line():
    assert is_stock_count_bill("SA2608-00050")
    assert is_stock_count_bill("3SA2608-00001")
    assert sale_for_cost(
        line_id=1,
        billno="SA2608-00050",
        billdate=date(2026, 8, 1),
        units=4,
        jourmode="1",
    ) is None
    assert is_transfer_bill("3TF6910-0007")
    assert sale_for_cost(
        line_id=2,
        billno="3TF6910-0007",
        billdate=date(2026, 10, 3),
        units=4,
        jourmode="1",
    ) is None


def test_sale_without_a_receipt_stays_uncosted():
    result = rebuild(
        0,
        [],
        [_s(1, 2026, 1, 2, 3, "8K69-1")],
        as_of=date(2026, 10, 4),
    )
    assert result.sales[0].status == STATUS_UNCOSTED
    assert result.sales[0].fifo_unit_cost is None
    assert result.cost_complete is False


def test_partial_when_receipts_run_out_mid_sale():
    result = rebuild(
        0,
        [_r(1, 2026, 1, 1, 2, 50)],
        [_s(2, 2026, 2, 1, 5)],
        as_of=date(2026, 10, 4),
    )
    sale = result.sales[0]
    assert sale.status == STATUS_PARTIAL
    assert sale.fifo_ext_cost == 100
    assert sale.allocs[0].qty == 2
    assert result.cost_complete is False


def test_older_than_five_years_is_used_only_when_the_window_is_open():
    as_of = date(2026, 10, 4)
    old = _r(1, 2018, 1, 1, 10, 40, "OLD")
    recent = _r(2, 2024, 1, 1, 1, 90, "NEW")
    # On hand 10 cannot be covered by the single recent unit.
    result = rebuild(10, [old, recent], [], as_of=as_of)
    assert result.history_window == WINDOW_FULL
    assert result.on_hand_covered_qty == 10
    assert result.cost_complete is True
    assert abs(result.on_hand_value - (90 + 9 * 40)) < 1e-6

    closed = rebuild(
        1,
        [old, recent],
        [_s(9, 2024, 6, 1, 0)],
        as_of=as_of,
    )
    # Zero-unit sale is ignored; on hand is the recent unit. 5 years is enough.
    assert closed.history_window == WINDOW_5Y
    assert closed.on_hand_value == 90


def test_vat_inclusive_amount_is_stripped():
    assert abs(unit_cost_ex_vat(107, 1, "Y") - 100) < 1e-9
    assert unit_cost_ex_vat(107, 1, "N") == 107
    assert unit_cost_ex_vat(0, 1, "N") is None


def test_cleanup_flags():
    as_of = date(2026, 10, 4)
    assert cleanup_decision(2, date(2010, 1, 1), as_of=as_of) == (False, CLEANUP_ON_HAND)
    assert cleanup_decision(0, date(2024, 1, 1), as_of=as_of) == (False, CLEANUP_MOVED)
    assert cleanup_decision(0, date(2018, 1, 1), as_of=as_of) == (True, CLEANUP_ZERO)
    assert cleanup_decision(0, None, as_of=as_of) == (True, CLEANUP_ZERO)
