"""Backward FIFO cost for one product.

On-hand is covered by the newest costed receipts. Older receipts are assigned
to sale lines, oldest sale first. Stock-count and transfer bills are not
sales. Opening cost is not used. A 5-year window is kept only when that
window already explains on-hand and every sale in it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from typing import Iterable, Sequence

QTY_EPS = 1e-4
STALE_EPS = 0.01

STATUS_OK = "OK"
STATUS_PARTIAL = "PARTIAL"
STATUS_UNCOSTED = "UNCOSTED"

WINDOW_5Y = "5y"
WINDOW_FULL = "full"

KIND_SALE = "sale"
KIND_PURCHASE = "purchase"
KIND_STOCK_COUNT = "stock_count"

CLEANUP_ZERO = "zero_oh_no_movement_5y"
CLEANUP_ON_HAND = "still_on_hand"
CLEANUP_MOVED = "moved_within_5y"


@dataclass(frozen=True)
class ReceiptLine:
    line_id: int
    billno: str
    billdate: date
    units: float
    unit_cost: float
    supplier: str | None = None


@dataclass(frozen=True)
class SaleLine:
    line_id: int
    billno: str
    billdate: date
    units: float


@dataclass(frozen=True)
class Alloc:
    pidet_id: int
    pi_billno: str
    pi_billdate: date
    supplier: str | None
    qty: float
    unit_cost: float


@dataclass(frozen=True)
class SaleCost:
    line_id: int
    billno: str
    billdate: date
    units: float
    fifo_unit_cost: float | None
    fifo_ext_cost: float | None
    status: str
    allocs: tuple[Alloc, ...]


@dataclass(frozen=True)
class FifoResult:
    on_hand_qty: float
    on_hand_covered_qty: float
    on_hand_value: float
    unassigned_qty: float
    sales: tuple[SaleCost, ...]
    history_window: str
    cost_complete: bool


def five_year_cutoff(as_of: date) -> date:
    try:
        return as_of.replace(year=as_of.year - 5)
    except ValueError:
        return as_of.replace(year=as_of.year - 5, day=28)


def is_stock_count_bill(billno: str | None) -> bool:
    bill = (billno or "").strip().upper()
    return bill.startswith(("SA", "3SA"))


def is_transfer_bill(billno: str | None) -> bool:
    bill = (billno or "").strip().upper()
    return bill.startswith(("TF", "3TF", "CNTF", "3CNTF"))


def unit_cost_ex_vat(amount: float, units: float, taxic: str | None) -> float | None:
    """Net unit cost. TAXIC=Y means the line amount includes 7% VAT."""
    if units <= QTY_EPS or amount <= QTY_EPS:
        return None
    net = amount * 100.0 / 107.0 if (taxic or "").strip().upper() == "Y" else amount
    if net <= QTY_EPS:
        return None
    return net / units


def ledger_on_hand(qty_beg: float, purchase_units: float, issue_units: float) -> float:
    """Same identity as ICMAS.QTYOH2: begin + receipts − issues."""
    return float(qty_beg or 0.0) + float(purchase_units or 0.0) - float(issue_units or 0.0)


def _positive(units: float) -> bool:
    return units > QTY_EPS


def receipt_for_cost(
    *,
    line_id: int,
    billno: str,
    billdate: date | None,
    units: float,
    amount: float,
    taxic: str | None,
    supplier: str | None,
) -> ReceiptLine | None:
    """Costed supplier receipt. Transfers and zero amounts do not open a layer."""
    if billdate is None or is_transfer_bill(billno) or not _positive(units):
        return None
    cost = unit_cost_ex_vat(amount, units, taxic)
    if cost is None:
        return None
    acct = (supplier or "").strip() or None
    return ReceiptLine(
        line_id=int(line_id),
        billno=(billno or "").strip(),
        billdate=billdate,
        units=float(units),
        unit_cost=cost,
        supplier=acct,
    )


def sale_for_cost(
    *,
    line_id: int,
    billno: str,
    billdate: date | None,
    units: float,
    jourmode: str | None,
) -> SaleLine | None:
    """Customer sale with positive qty. Stock counts, transfers, and returns are out."""
    if billdate is None:
        return None
    if (jourmode or "").strip() == "0":
        return None
    if is_stock_count_bill(billno) or is_transfer_bill(billno):
        return None
    if not _positive(units):
        return None
    return SaleLine(
        line_id=int(line_id),
        billno=(billno or "").strip(),
        billdate=billdate,
        units=float(units),
    )


def issue_kind(billno: str | None, units: float, jourmode: str | None) -> str | None:
    """Last-movement kind for an issue, or None when it does not count."""
    if (jourmode or "").strip() == "0":
        return None
    if is_transfer_bill(billno):
        return None
    if is_stock_count_bill(billno):
        return KIND_STOCK_COUNT
    if abs(units) > QTY_EPS:
        return KIND_SALE
    return None


def _complete(on_hand: float, result: FifoResult) -> bool:
    if on_hand < -QTY_EPS:
        return False
    target = max(on_hand, 0.0)
    if result.on_hand_covered_qty + QTY_EPS < target:
        return False
    return all(row.status == STATUS_OK for row in result.sales)


def _in_window(billdate: date, cutoff: date) -> bool:
    return billdate >= cutoff


def allocate(
    on_hand: float,
    receipts: Sequence[ReceiptLine],
    sales: Sequence[SaleLine],
) -> FifoResult:
    """Cover `on_hand` from the newest receipts, then cost sales from what is left."""
    layers: list[dict] = [
        {
            "line_id": row.line_id,
            "billno": row.billno,
            "billdate": row.billdate,
            "units_left": row.units,
            "unit_cost": row.unit_cost,
            "supplier": row.supplier,
        }
        for row in receipts
        if _positive(row.units)
    ]
    layers.sort(key=lambda row: (row["billdate"], row["line_id"]), reverse=True)

    peel = max(float(on_hand or 0.0), 0.0)
    covered = 0.0
    stock_value = 0.0
    for layer in layers:
        if peel <= QTY_EPS:
            break
        take = min(layer["units_left"], peel)
        if take <= QTY_EPS:
            continue
        stock_value += take * layer["unit_cost"]
        covered += take
        layer["units_left"] -= take
        peel -= take

    consumed = [layer for layer in layers if layer["units_left"] > QTY_EPS]
    consumed.sort(key=lambda row: (row["billdate"], row["line_id"]))
    cursor = 0
    sale_rows: list[SaleCost] = []
    ordered = sorted(sales, key=lambda row: (row.billdate, row.line_id))
    for sale in ordered:
        if not _positive(sale.units):
            continue
        need = sale.units
        allocs: list[Alloc] = []
        ext = 0.0
        costed = 0.0
        while need > QTY_EPS and cursor < len(consumed):
            layer = consumed[cursor]
            take = min(need, layer["units_left"])
            if take <= QTY_EPS:
                cursor += 1
                continue
            allocs.append(
                Alloc(
                    pidet_id=int(layer["line_id"]),
                    pi_billno=layer["billno"],
                    pi_billdate=layer["billdate"],
                    supplier=layer["supplier"],
                    qty=take,
                    unit_cost=layer["unit_cost"],
                )
            )
            ext += take * layer["unit_cost"]
            costed += take
            need -= take
            layer["units_left"] -= take
            if layer["units_left"] <= QTY_EPS:
                cursor += 1
        if costed <= QTY_EPS:
            status = STATUS_UNCOSTED
            unit = None
            ext_out = None
        elif need > QTY_EPS:
            status = STATUS_PARTIAL
            unit = ext / costed
            ext_out = ext
        else:
            status = STATUS_OK
            unit = ext / sale.units
            ext_out = ext
        sale_rows.append(
            SaleCost(
                line_id=sale.line_id,
                billno=sale.billno,
                billdate=sale.billdate,
                units=sale.units,
                fifo_unit_cost=unit,
                fifo_ext_cost=ext_out,
                status=status,
                allocs=tuple(allocs),
            )
        )

    unassigned = sum(layer["units_left"] for layer in consumed if layer["units_left"] > QTY_EPS)
    return FifoResult(
        on_hand_qty=float(on_hand or 0.0),
        on_hand_covered_qty=covered,
        on_hand_value=stock_value,
        unassigned_qty=unassigned,
        sales=tuple(sale_rows),
        history_window=WINDOW_FULL,
        cost_complete=False,
    )


def rebuild(
    on_hand: float,
    receipts: Sequence[ReceiptLine],
    sales: Sequence[SaleLine],
    *,
    as_of: date,
) -> FifoResult:
    """Use 5 years when that window closes the cycle. Otherwise use all history."""
    cutoff = five_year_cutoff(as_of)
    recent_receipts = [row for row in receipts if _in_window(row.billdate, cutoff)]
    recent_sales = [row for row in sales if _in_window(row.billdate, cutoff)]
    dropped_history = len(recent_receipts) != len(receipts) or len(recent_sales) != len(sales)
    if not dropped_history:
        result = allocate(on_hand, receipts, sales)
        return replace(
            result,
            history_window=WINDOW_5Y,
            cost_complete=_complete(on_hand, result),
        )

    trial = allocate(on_hand, recent_receipts, recent_sales)
    if _complete(on_hand, trial):
        return replace(trial, history_window=WINDOW_5Y, cost_complete=True)

    full = allocate(on_hand, receipts, sales)
    return replace(full, history_window=WINDOW_FULL, cost_complete=_complete(on_hand, full))


def pick_last_movement(movements: Iterable[tuple[date, str]]) -> tuple[date | None, str | None]:
    rank = {KIND_STOCK_COUNT: 3, KIND_SALE: 2, KIND_PURCHASE: 1}
    best: tuple[date, str] | None = None
    for at, kind in movements:
        if best is None or (at, rank.get(kind, 0)) > (best[0], rank.get(best[1], 0)):
            best = (at, kind)
    if best is None:
        return None, None
    return best


def qtyoh2_is_stale(qtyoh2_icmas: float | None, qtyoh2_ledger: float) -> bool:
    if qtyoh2_icmas is None:
        return abs(qtyoh2_ledger) > STALE_EPS
    return abs(float(qtyoh2_icmas) - float(qtyoh2_ledger)) > STALE_EPS


def cleanup_decision(
    qtyoh2_ledger: float,
    last_movement_at: date | None,
    *,
    as_of: date,
) -> tuple[bool, str]:
    if qtyoh2_ledger > STALE_EPS:
        return False, CLEANUP_ON_HAND
    cutoff = five_year_cutoff(as_of)
    if last_movement_at is not None and last_movement_at >= cutoff:
        return False, CLEANUP_MOVED
    return True, CLEANUP_ZERO
