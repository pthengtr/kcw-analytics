"""VAT-only inventory. Same identity as QTYOH2, with a VAT document filter.

QTYOH2 = opening + purchases − issues.

VAT stock uses that identity on a filtered set:

- purchases are ISVAT=Y receipt lines, plus RV and 3RV
- issues are TR / TD / TAD and their credit notes (HQ and SYP), TAR / 3TAR /
  CNTAR / 3CNTAR, and SA / 3SA stock adjustments
- non-VAT shop bills stay out; TAR already stands for a non-VAT sale of VAT goods
- a transfer is an issue at the sending branch and a receipt at the other
- only the portion backward FIFO can cost from VAT layers moves between books

Opening VAT qty is today's company on-hand walked back to 1 Jan, then peeled
onto pre-2026 supplier receipts, newest first. Only the slices that land on
ISVAT=Y receipts enter the VAT book. The oldest of those slices, up to the
quantity sitting on SYP's pre-2026 transfers, open the SYP book. The rest open
HQ. Remaining qty is peeled onto each branch's VAT receipt pool for cost.

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
KIND_TF = "TF"

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
    inbound: bool = False
    ref_billno: str | None = None


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
    ref_billno: str | None = None


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
    tf_qty: float
    actual_qty: float
    site: str = "hq"
    layers: list[Layer] = field(default_factory=list)
    movements: list[Movement] = field(default_factory=list)


@dataclass
class SiteBook:
    """One branch. Transfers on SIDET are issues. PIDET transfers only explain stock."""

    actual: float = 0.0
    ytd_pi: float = 0.0
    ytd_si: float = 0.0
    pre_supplier: list[Receipt] = field(default_factory=list)
    pre_transfer: list[Receipt] = field(default_factory=list)
    purchases: list[Flow] = field(default_factory=list)
    sales: list[Flow] = field(default_factory=list)
    rvs: list[Flow] = field(default_factory=list)
    tars: list[Flow] = field(default_factory=list)
    adjustments: list[Flow] = field(default_factory=list)
    transfer_out: list[Flow] = field(default_factory=list)


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
    transfers: list[Flow] | None = None,
) -> tuple[list[Flow], list[Flow], list[Flow]]:
    """Assign leftover VAT receipts to issues after the newest receipts cover on-hand.

    Same backward FIFO as the company cost book: newest layers stay in stock,
    older layers are consumed by issues from oldest to newest. A sale or transfer
    keeps the purchase bill numbers it consumed. A transfer keeps only the qty
    those VAT layers could cover.
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
    transfer_rows = list(transfers or [])
    slots = (
        [("sales", index, row) for index, row in enumerate(sales)]
        + [("tars", index, row) for index, row in enumerate(tars)]
        + [("adjustments", index, row) for index, row in enumerate(adjustments)]
        + [("transfers", index, row) for index, row in enumerate(transfer_rows)]
    )
    slots.sort(
        key=lambda item: (item[2].billdate, _kind_rank(item[2].kind), item[2].billno, item[2].site)
    )
    sales_out = list(sales)
    tars_out = list(tars)
    transfers_out: list[Flow] = []
    for group, index, row in slots:
        if row.kind not in (KIND_SALE, KIND_TAR, KIND_SA, KIND_TF) or not _positive(row.units):
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
            units=costed if group == "transfers" else row.units,
            unit_cost=extended / costed,
            cost_billno=", ".join(bills) or None,
        )
        if group == "sales":
            sales_out[index] = stamped
        elif group == "tars":
            tars_out[index] = stamped
        else:
            transfers_out.append(stamped)
    return sales_out, tars_out, transfers_out


def attach_return_costs(
    sales: list[Flow], tars: list[Flow]
) -> tuple[list[Flow], list[Flow]]:
    """A credit note keeps the purchase cost of the sale it reverses.

    The returned unit is not put back into the stock layers. Later sales stay
    uncosted when no receipt is left. The credit-note line shows the original
    sale's cost so the link is visible.
    """

    priced: dict[tuple[str, str], Flow] = {}
    for row in list(sales) + list(tars):
        if row.unit_cost is None or not _positive(row.units) or not row.billno:
            continue
        priced.setdefault((row.site, row.billno), row)

    def stamp(rows: list[Flow]) -> list[Flow]:
        out: list[Flow] = []
        for row in rows:
            source = priced.get((row.site, row.ref_billno or ""))
            if source is None or row.units >= -QTY_EPS:
                out.append(row)
                continue
            out.append(
                replace(row, unit_cost=source.unit_cost, cost_billno=source.cost_billno)
            )
        return out

    return stamp(sales), stamp(tars)


def is_credit_note_bill(billno: str | None) -> bool:
    bill = (billno or "").strip().upper()
    return bill.startswith(("3CNTAR", "CNTAR", "3CNTAD", "CNTAD", "3CN", "CN"))


def attribute_return_sources(
    sales: list[Flow], tars: list[Flow]
) -> tuple[list[Flow], list[Flow]]:
    """Name the credit notes on a later sale that had no purchase left.

    Returned units are sold again with no new receipt. The oldest uncosted
    sale takes the oldest credit note.
    """

    returns: list[dict] = []
    for row in list(sales) + list(tars):
        if row.units >= -QTY_EPS or not is_credit_note_bill(row.billno):
            continue
        returns.append(
            {"billno": row.billno, "billdate": row.billdate, "left": abs(float(row.units))}
        )
    returns.sort(key=lambda row: (row["billdate"], row["billno"]))

    slots = [("sales", index, row) for index, row in enumerate(sales)] + [
        ("tars", index, row) for index, row in enumerate(tars)
    ]
    slots.sort(key=lambda item: (item[2].billdate, item[2].billno, item[2].site))
    sales_out = list(sales)
    tars_out = list(tars)
    cursor = 0
    for group, index, row in slots:
        if row.unit_cost is not None or not _positive(row.units):
            continue
        need = float(row.units)
        bills: list[str] = []
        while need > QTY_EPS and cursor < len(returns):
            src = returns[cursor]
            if src["left"] <= QTY_EPS:
                cursor += 1
                continue
            take = min(need, src["left"])
            if src["billno"] not in bills:
                bills.append(src["billno"])
            need -= take
            src["left"] -= take
            if src["left"] <= QTY_EPS:
                cursor += 1
        if not bills:
            continue
        stamped = replace(row, ref_billno=", ".join(bills))
        if group == "sales":
            sales_out[index] = stamped
        else:
            tars_out[index] = stamped
    return sales_out, tars_out


def stock_effect(kind: str, units: float, *, inbound: bool = False) -> float:
    """Signed change in VAT qty. Inflows add the source qty; issues subtract it."""

    if kind in _INFLOW or (kind == KIND_TF and inbound):
        return float(units)
    return -float(units)


def _positive(units: float) -> bool:
    return units > QTY_EPS


def peel_opening(
    on_hand: float, receipts: list[Receipt], *, year_start: date = YEAR_START
) -> tuple[list[Layer], float]:
    """Cover 1 Jan qty from newest older receipts. Keep only VAT slices."""

    left = max(float(on_hand or 0.0), 0.0)
    ordered = sorted(
        (row for row in receipts if _positive(row.units) and row.billdate < year_start),
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


def transfer_cover(on_hand: float, receipts: list[Receipt], *, year_start: date = YEAR_START) -> float:
    """How much of `on_hand` sits on transfer receipts from before the year, newest first."""

    left = max(float(on_hand or 0.0), 0.0)
    covered = 0.0
    ordered = sorted(
        (row for row in receipts if _positive(row.units) and row.billdate < year_start),
        key=lambda row: (row.billdate, row.site, row.line_id),
        reverse=True,
    )
    for row in ordered:
        if left <= QTY_EPS:
            break
        take = min(row.units, left)
        left -= take
        covered += take
    return covered


def split_opening_layers(layers: list[Layer], syp_qty: float) -> tuple[list[Layer], list[Layer]]:
    """Give SYP the oldest opening VAT layers, up to the qty its old transfers hold."""

    left = max(float(syp_qty or 0.0), 0.0)
    ordered = sorted(layers, key=lambda row: (row.billdate, row.billno))
    syp: list[Layer] = []
    hq: list[Layer] = []
    for row in ordered:
        if not _positive(row.qty) or left <= QTY_EPS:
            hq.append(replace(row, site="hq"))
            continue
        take = min(row.qty, left)
        left -= take
        syp.append(replace(row, qty=take, site="syp"))
        rest = row.qty - take
        if rest > QTY_EPS:
            hq.append(replace(row, qty=rest, site="hq"))
    return syp, hq


def _receipt_layers(rows: list[Flow], source: str) -> list[Layer]:
    layers: list[Layer] = []
    for row in rows:
        if not _positive(row.units):
            continue
        layers.append(
            Layer(
                billno=row.billno,
                billdate=row.billdate,
                source=source,
                qty=row.units,
                unit_cost=row.unit_cost or 0.0,
                supplier=row.supplier,
                site=row.site,
            )
        )
    return layers


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
    transfers_out: list[Flow] | None = None,
    transfers_in: list[Flow] | None = None,
    opening_layers: list[Layer] | None = None,
    site: str = "hq",
    year_start: date = YEAR_START,
) -> VatProduct | None:
    if is_service_bcode(bcode):
        return None

    outbound = list(transfers_out or [])
    inbound = [replace(row, inbound=True, site=site) for row in (transfers_in or [])]
    if opening_layers is None:
        actual_1_jan = float(actual_qty) - float(ytd_purchase_units) + float(ytd_issue_units)
        open_layers, _uncovered_open = peel_opening(actual_1_jan, pre_receipts, year_start=year_start)
    else:
        open_layers = [replace(row, site=site) for row in opening_layers]
    open_qty = sum(layer.qty for layer in open_layers)
    purchase_qty = _sum_units(purchases)
    rv_qty = _sum_units(rvs)
    sale_qty = _sum_units(sales)
    tar_qty = _sum_units(tars)
    sa_qty = _sum_units(adjustments)
    tf_in_qty = _sum_units(inbound)
    tf_out_qty = _sum_units(outbound)
    qty_if_all = open_qty + purchase_qty + rv_qty + tf_in_qty - sale_qty - tar_qty - sa_qty - tf_out_qty

    pool = list(open_layers)
    pool.extend(_receipt_layers(purchases, KIND_PI))
    pool.extend(_receipt_layers(rvs, KIND_RV))
    pool.extend(_receipt_layers(inbound, KIND_TF))
    sales, tars, outbound = trace_sale_costs(
        pool,
        max(qty_if_all, 0.0),
        sales,
        tars,
        adjustments,
        outbound,
    )
    sales, tars = attach_return_costs(sales, tars)
    sales, tars = attribute_return_sources(sales, tars)
    tf_out_costed = _sum_units(outbound)
    tf_qty = tf_in_qty - tf_out_costed
    qty = qty_if_all + (tf_out_qty - tf_out_costed)

    active = any(
        (
            abs(open_qty) > QTY_EPS,
            abs(tf_qty) > QTY_EPS,
            purchases,
            sales,
            rvs,
            tars,
        )
    )
    if not active:
        return None

    layers, _covered, value = peel_on_hand(qty, pool)
    uncovered = max(qty, 0.0) - sum(layer.qty for layer in layers)

    flows: list[tuple[int, Flow]] = []
    for row in purchases + rvs + sales + tars + adjustments + inbound + outbound:
        if row.billdate < year_start:
            continue
        flows.append((_kind_rank(row.kind), row))
    flows.sort(key=lambda item: (item[1].billdate, item[0], item[1].billno, item[1].site))

    balance = open_qty
    movements = [
        Movement(
            billdate=year_start,
            billno="",
            kind=KIND_OPEN,
            site=site,
            qty=open_qty,
            unit_cost=None,
            balance=open_qty,
        )
    ]
    for _rank, row in flows:
        signed = stock_effect(row.kind, row.units, inbound=row.inbound)
        balance += signed
        movements.append(
            Movement(
                billdate=row.billdate,
                billno=row.billno,
                kind=row.kind,
                site=site,
                qty=signed,
                unit_cost=row.unit_cost,
                balance=balance,
                sale_price=row.sale_price,
                cost_billno=row.cost_billno,
                supplier=row.supplier,
                ref_billno=row.ref_billno,
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
        tf_qty=tf_qty,
        actual_qty=float(actual_qty),
        site=site,
        layers=layers if qty > QTY_EPS else [],
        movements=movements,
    )


def _inbound_from_outbound(product: VatProduct | None, site: str) -> list[Flow]:
    if product is None:
        return []
    rows: list[Flow] = []
    for row in product.movements:
        if row.kind != KIND_TF or row.qty >= -QTY_EPS:
            continue
        rows.append(
            Flow(
                billdate=row.billdate,
                billno=row.billno,
                units=abs(row.qty),
                kind=KIND_TF,
                site=site,
                unit_cost=row.unit_cost,
                cost_billno=row.cost_billno,
                inbound=True,
            )
        )
    return rows


def _same_transfers(left: list[Flow], right: list[Flow]) -> bool:
    def key(row: Flow) -> tuple:
        return (row.billno, row.billdate, round(row.units, 4), round(row.unit_cost or 0.0, 4))

    return sorted(map(key, left)) == sorted(map(key, right))


def build_hq_and_syp(
    *,
    bcode: str,
    descr: str,
    hq: SiteBook,
    syp: SiteBook,
    year_start: date = YEAR_START,
) -> list[VatProduct]:
    """Two VAT books. A costed HQ transfer arrives at SYP with the same FIFO cost."""

    if is_service_bcode(bcode):
        return []

    company_1_jan = (
        float(hq.actual)
        + float(syp.actual)
        - float(hq.ytd_pi)
        - float(syp.ytd_pi)
        + float(hq.ytd_si)
        + float(syp.ytd_si)
    )
    company_layers, _uncovered = peel_opening(
        company_1_jan,
        list(hq.pre_supplier) + list(syp.pre_supplier),
        year_start=year_start,
    )
    syp_1_jan = float(syp.actual) - float(syp.ytd_pi) + float(syp.ytd_si)
    syp_open, hq_open = split_opening_layers(
        company_layers, transfer_cover(syp_1_jan, syp.pre_transfer, year_start=year_start)
    )

    def build_site(book: SiteBook, site: str, opening: list[Layer], inbound: list[Flow]) -> VatProduct | None:
        return build_vat_product(
            bcode=bcode,
            descr=descr,
            actual_qty=book.actual,
            pre_receipts=[],
            ytd_purchase_units=0.0,
            ytd_issue_units=0.0,
            purchases=book.purchases,
            sales=book.sales,
            rvs=book.rvs,
            tars=book.tars,
            adjustments=book.adjustments,
            transfers_out=book.transfer_out,
            transfers_in=inbound,
            opening_layers=opening,
            site=site,
            year_start=year_start,
        )

    hq_inbound: list[Flow] = []
    hq_product: VatProduct | None = None
    syp_product: VatProduct | None = None
    for _pass in range(2):
        hq_product = build_site(hq, "hq", hq_open, hq_inbound)
        syp_inbound = _inbound_from_outbound(hq_product, "syp")
        syp_product = build_site(syp, "syp", syp_open, syp_inbound)
        returned = _inbound_from_outbound(syp_product, "hq")
        if _same_transfers(returned, hq_inbound):
            break
        hq_inbound = returned

    return [row for row in (hq_product, syp_product) if row is not None]


def _kind_rank(kind: str) -> int:
    return {KIND_PI: 1, KIND_RV: 2, KIND_SALE: 3, KIND_TAR: 4, KIND_TF: 5, KIND_SA: 6}.get(kind, 9)
