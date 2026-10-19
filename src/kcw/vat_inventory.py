"""VAT-only inventory. Same identity as QTYOH2, with a VAT document filter.

QTYOH2 = opening + purchases − issues.

VAT stock uses that identity on a filtered set:

- purchases are ISVAT=Y receipt lines, plus RV and 3RV
- issues are TR / TD / TAD and their credit notes (HQ and SYP), TAR / 3TAR /
  CNTAR / 3CNTAR, and SA / 3SA stock adjustments
- non-VAT shop bills stay out; TAR already stands for a non-VAT sale of VAT goods
- transfers stay in the walk back to 1 Jan and stay out of the VAT movement list

Opening VAT qty is today's company on-hand walked back to 1 Jan, then peeled
onto pre-2026 receipts, newest first. Only the slices that land on ISVAT=Y
receipts enter the VAT book. Remaining qty is peeled the same way onto the
VAT receipt pool (opening slices, 2026 VAT purchases, RV) for cost and origin.

Each 2026 movement keeps that bill's net purchase cost and, for a sale or TAR,
its net selling price. Both are per smallest unit and exclude VAT. A sale's
cost is the backward FIFO cost: the newest VAT receipts stay on hand, and the
older receipts are assigned to issues, oldest first. The purchase bill numbers
of those receipts are kept on the sale.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date

from src.kcw.fifo_cost import unit_cost_ex_vat

QTY_EPS = 1e-4
YEAR_START = date(2026, 1, 1)

KIND_OPEN = "OPEN"
KIND_PI = "PI"
KIND_RV = "RV"
KIND_SALE = "SALE"
KIND_TAR = "TAR"
KIND_SA = "SA"

_INFLOW = {KIND_OPEN, KIND_PI, KIND_RV}


@dataclass(frozen=True)
class Receipt:
    line_id: int
    billno: str
    billdate: date
    units: float
    unit_cost: float
    is_vat: bool
    supplier: str | None = None
    site: str = "hq"


@dataclass(frozen=True)
class Flow:
    """One 2026 line. `units` keeps the source sign. Sales and SA subtract it."""

    billdate: date
    billno: str
    units: float
    kind: str
    site: str = "hq"
    unit_cost: float | None = None
    sale_price: float | None = None
    cost_billno: str | None = None
    supplier: str | None = None


@dataclass(frozen=True)
class Layer:
    billno: str
    billdate: date
    source: str
    qty: float
    unit_cost: float
    supplier: str | None = None
    site: str = "hq"


@dataclass(frozen=True)
class Movement:
    billdate: date
    billno: str
    kind: str
    site: str
    qty: float
    unit_cost: float | None
    balance: float
    sale_price: float | None = None
    cost_billno: str | None = None
    supplier: str | None = None


@dataclass
class VatProduct:
    bcode: str
    descr: str
    qty: float
    value: float
    uncovered_qty: float
    open_qty: float
    purchase_qty: float
    rv_qty: float
    sale_qty: float
    tar_qty: float
    sa_qty: float
    actual_qty: float
    layers: list[Layer] = field(default_factory=list)
    movements: list[Movement] = field(default_factory=list)


def is_service_bcode(bcode: str | None) -> bool:
    code = (bcode or "").strip().upper()
    return code.startswith(("70", "91"))


def document_kind(billno: str | None) -> str | None:
    """SIDET bill role in the VAT book. TAR numbers are owned by billgen."""

    bill = (billno or "").strip().upper()
    if not bill:
        return None
    if bill.startswith(("3CNTAR", "CNTAR", "3TAR", "TAR")):
        return None
    if bill.startswith(("3SA", "SA")):
        return KIND_SA
    if bill.startswith(
        (
            "3CNTAD",
            "CNTAD",
            "3TAD",
            "TAD",
            "3TD",
            "TD",
            "3TR",
            "TR",
            "3CN",
            "CN",
        )
    ):
        return KIND_SALE
    return None


def sale_unit_price(
    *,
    amount: float,
    units: float,
    taxic: str | None,
    price: float | None = None,
    mtp: float | None = None,
) -> float | None:
    """Net selling price per smallest unit. TAXIC=Y means the amount includes VAT."""

    net = unit_cost_ex_vat(abs(float(amount or 0.0)), abs(float(units or 0.0)), taxic)
    if net is not None:
        return net
    pack = abs(float(mtp or 0.0)) or 1.0
    raw = abs(float(price or 0.0))
    if raw <= QTY_EPS:
        return None
    net_price = raw * 100.0 / 107.0 if (taxic or "").strip().upper() == "Y" else raw
    if net_price <= QTY_EPS:
        return None
    return net_price / pack


def trace_sale_costs(
    pool: list[Layer],
    on_hand: float,
    sales: list[Flow],
    tars: list[Flow],
    adjustments: list[Flow],
) -> tuple[list[Flow], list[Flow]]:
    """Assign leftover VAT receipts to sales after the newest receipts cover on-hand.

    Same backward FIFO as the company cost book: newest layers stay in stock,
    older layers are consumed by issues from oldest to newest. A sale keeps the
    purchase bill numbers it consumed.
    """

    layers = [
        {
            "billno": row.billno,
            "billdate": row.billdate,
            "left": row.qty,
            "unit_cost": row.unit_cost,
        }
        for row in pool
        if _positive(row.qty)
    ]
    layers.sort(key=lambda row: (row["billdate"], row["billno"]), reverse=True)
    reserve = max(float(on_hand or 0.0), 0.0)
    for layer in layers:
        if reserve <= QTY_EPS:
            break
        take = min(layer["left"], reserve)
        layer["left"] -= take
        reserve -= take
    layers.sort(key=lambda row: (row["billdate"], row["billno"]))

    cursor = 0
    slots = (
        [("sales", index, row) for index, row in enumerate(sales)]
        + [("tars", index, row) for index, row in enumerate(tars)]
        + [("adjustments", index, row) for index, row in enumerate(adjustments)]
    )
    slots.sort(
        key=lambda item: (item[2].billdate, _kind_rank(item[2].kind), item[2].billno, item[2].site)
    )
    sales_out = list(sales)
    tars_out = list(tars)
    for group, index, row in slots:
        if row.kind not in (KIND_SALE, KIND_TAR, KIND_SA) or not _positive(row.units):
            continue
        need = float(row.units)
        allocs: list[tuple[str, float, float]] = []
        extended = 0.0
        costed = 0.0
        while need > QTY_EPS and cursor < len(layers):
            layer = layers[cursor]
            if layer["left"] <= QTY_EPS:
                cursor += 1
                continue
            take = min(need, layer["left"])
            allocs.append((layer["billno"], layer["unit_cost"], take))
            extended += take * layer["unit_cost"]
            costed += take
            need -= take
            layer["left"] -= take
            if layer["left"] <= QTY_EPS:
                cursor += 1
        if group == "adjustments" or costed <= QTY_EPS:
            continue
        bills: list[str] = []
        for billno, _cost, _qty in allocs:
            if billno and billno not in bills:
                bills.append(billno)
        stamped = replace(
            row,
            unit_cost=extended / costed,
            cost_billno=", ".join(bills) or None,
        )
        if group == "sales":
            sales_out[index] = stamped
        else:
            tars_out[index] = stamped
    return sales_out, tars_out


def stock_effect(kind: str, units: float) -> float:
    """Signed change in VAT qty. Inflows add the source qty; issues subtract it."""

    if kind in _INFLOW:
        return float(units)
    return -float(units)


def _positive(units: float) -> bool:
    return units > QTY_EPS


def peel_opening(on_hand: float, receipts: list[Receipt]) -> tuple[list[Layer], float]:
    """Cover 1 Jan qty from newest older receipts. Keep only VAT slices."""

    left = max(float(on_hand or 0.0), 0.0)
    ordered = sorted(
        (row for row in receipts if _positive(row.units) and row.billdate < YEAR_START),
        key=lambda row: (row.billdate, row.site, row.line_id),
        reverse=True,
    )
    layers: list[Layer] = []
    for row in ordered:
        if left <= QTY_EPS:
            break
        take = min(row.units, left)
        left -= take
        if row.is_vat and take > QTY_EPS:
            layers.append(
                Layer(
                    billno=row.billno,
                    billdate=row.billdate,
                    source=KIND_PI,
                    qty=take,
                    unit_cost=row.unit_cost,
                    supplier=row.supplier,
                    site=row.site,
                )
            )
    return layers, left


def peel_on_hand(qty: float, pool: list[Layer]) -> tuple[list[Layer], float, float]:
    """Newest VAT receipts still on hand, with extended cost."""

    left = max(float(qty or 0.0), 0.0)
    ordered = sorted(pool, key=lambda row: (row.billdate, row.billno), reverse=True)
    kept: list[Layer] = []
    covered = 0.0
    value = 0.0
    for row in ordered:
        if left <= QTY_EPS:
            break
        if not _positive(row.qty):
            continue
        take = min(row.qty, left)
        left -= take
        covered += take
        value += take * row.unit_cost
        kept.append(
            Layer(
                billno=row.billno,
                billdate=row.billdate,
                source=row.source,
                qty=take,
                unit_cost=row.unit_cost,
                supplier=row.supplier,
                site=row.site,
            )
        )
    kept.sort(key=lambda row: (row.billdate, row.billno))
    return kept, covered, value


def _sum_units(rows: list[Flow]) -> float:
    return float(sum(row.units for row in rows))


def build_vat_product(
    *,
    bcode: str,
    descr: str,
    actual_qty: float,
    pre_receipts: list[Receipt],
    ytd_purchase_units: float,
    ytd_issue_units: float,
    purchases: list[Flow],
    sales: list[Flow],
    rvs: list[Flow],
    tars: list[Flow],
    adjustments: list[Flow],
) -> VatProduct | None:
    if is_service_bcode(bcode):
        return None

    actual_1_jan = float(actual_qty) - float(ytd_purchase_units) + float(ytd_issue_units)
    open_layers, _uncovered_open = peel_opening(actual_1_jan, pre_receipts)
    open_qty = sum(layer.qty for layer in open_layers)
    purchase_qty = _sum_units(purchases)
    rv_qty = _sum_units(rvs)
    sale_qty = _sum_units(sales)
    tar_qty = _sum_units(tars)
    sa_qty = _sum_units(adjustments)
    qty = open_qty + purchase_qty + rv_qty - sale_qty - tar_qty - sa_qty

    active = any(
        (
            abs(open_qty) > QTY_EPS,
            purchases,
            sales,
            rvs,
            tars,
        )
    )
    if not active:
        return None

    pool = list(open_layers)
    for row in purchases:
        if _positive(row.units):
            pool.append(
                Layer(
                    billno=row.billno,
                    billdate=row.billdate,
                    source=KIND_PI,
                    qty=row.units,
                    unit_cost=row.unit_cost or 0.0,
                    supplier=row.supplier,
                    site=row.site,
                )
            )
    for row in rvs:
        if _positive(row.units):
            pool.append(
                Layer(
                    billno=row.billno,
                    billdate=row.billdate,
                    source=KIND_RV,
                    qty=row.units,
                    unit_cost=row.unit_cost or 0.0,
                    supplier=None,
                    site=row.site,
                )
            )
    layers, _covered, value = peel_on_hand(qty, pool)
    uncovered = max(qty, 0.0) - sum(layer.qty for layer in layers)
    sales, tars = trace_sale_costs(pool, qty, sales, tars, adjustments)

    flows: list[tuple[int, Flow]] = []
    for row in purchases + rvs + sales + tars + adjustments:
        if row.billdate < YEAR_START:
            continue
        flows.append((_kind_rank(row.kind), row))
    flows.sort(key=lambda item: (item[1].billdate, item[0], item[1].billno, item[1].site))

    balance = open_qty
    movements = [
        Movement(
            billdate=YEAR_START,
            billno="",
            kind=KIND_OPEN,
            site="",
            qty=open_qty,
            unit_cost=None,
            balance=open_qty,
        )
    ]
    for _rank, row in flows:
        balance += stock_effect(row.kind, row.units)
        movements.append(
            Movement(
                billdate=row.billdate,
                billno=row.billno,
                kind=row.kind,
                site=row.site,
                qty=stock_effect(row.kind, row.units),
                unit_cost=row.unit_cost,
                balance=balance,
                sale_price=row.sale_price,
                cost_billno=row.cost_billno,
                supplier=row.supplier,
            )
        )

    return VatProduct(
        bcode=bcode.strip(),
        descr=(descr or "").strip(),
        qty=qty,
        value=value if qty > QTY_EPS else 0.0,
        uncovered_qty=uncovered if qty > QTY_EPS else 0.0,
        open_qty=open_qty,
        purchase_qty=purchase_qty,
        rv_qty=rv_qty,
        sale_qty=sale_qty,
        tar_qty=tar_qty,
        sa_qty=sa_qty,
        actual_qty=float(actual_qty),
        layers=layers if qty > QTY_EPS else [],
        movements=movements,
    )


def _kind_rank(kind: str) -> int:
    return {KIND_PI: 1, KIND_RV: 2, KIND_SALE: 3, KIND_TAR: 4, KIND_SA: 5}.get(kind, 9)
