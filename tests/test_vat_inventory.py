from datetime import date

from src.kcw.vat_inventory import (
    KIND_PI,
    KIND_RV,
    KIND_SA,
    KIND_SALE,
    KIND_TAR,
    Flow,
    Receipt,
    build_vat_product,
    document_kind,
    peel_opening,
    sale_unit_price,
)


def _receipt(billdate, units, is_vat, billno="PI", line_id=1, cost=10.0):
    return Receipt(
        line_id=line_id,
        billno=billno,
        billdate=billdate,
        units=units,
        unit_cost=cost,
        is_vat=is_vat,
    )


def _flow(kind, units, billno, day=15, cost=None, sale_price=None):
    return Flow(
        billdate=date(2026, 3, day),
        billno=billno,
        units=units,
        kind=kind,
        unit_cost=cost,
        sale_price=sale_price,
    )


def test_document_kind_keeps_tar_out_of_sidet_sales():
    assert document_kind("TR6901-001") == KIND_SALE
    assert document_kind("3TAD6903-002") == KIND_SALE
    assert document_kind("CNTAD6902-001") == KIND_SALE
    assert document_kind("CN6901-004") == KIND_SALE
    assert document_kind("SA2608-00001") == KIND_SA
    assert document_kind("3SA6901-001") == KIND_SA
    assert document_kind("TAR6903-001") is None
    assert document_kind("CNTAR6902-001") is None
    assert document_kind("8K69-0001") is None
    assert document_kind("TF6901-001") is None


def test_opening_peel_skips_non_vat_layers():
    layers, left = peel_opening(
        10,
        [
            _receipt(date(2025, 12, 1), 6, False, "N1", 1),
            _receipt(date(2025, 11, 1), 8, True, "V1", 2, cost=4),
        ],
    )
    assert left == 0
    assert len(layers) == 1
    assert layers[0].billno == "V1"
    assert layers[0].qty == 4
    assert layers[0].unit_cost == 4


def test_vat_purchase_and_vat_sale():
    product = build_vat_product(
        bcode="A",
        descr="bolt",
        actual_qty=4,
        pre_receipts=[],
        ytd_purchase_units=10,
        ytd_issue_units=6,
        purchases=[_flow(KIND_PI, 10, "PI1", cost=5)],
        sales=[_flow(KIND_SALE, 6, "TR1")],
        rvs=[],
        tars=[],
        adjustments=[],
    )
    assert product is not None
    assert product.open_qty == 0
    assert product.qty == 4
    assert product.layers[0].billno == "PI1"
    assert product.layers[0].qty == 4
    assert product.value == 20


def test_non_vat_purchase_and_sale_stay_out():
    product = build_vat_product(
        bcode="B",
        descr="nut",
        actual_qty=3,
        pre_receipts=[_receipt(date(2025, 6, 1), 3, False)],
        ytd_purchase_units=0,
        ytd_issue_units=0,
        purchases=[],
        sales=[],
        rvs=[],
        tars=[],
        adjustments=[],
    )
    assert product is None


def test_vat_purchase_and_tar_sale():
    product = build_vat_product(
        bcode="C",
        descr="vat buy",
        actual_qty=0,
        pre_receipts=[],
        ytd_purchase_units=5,
        ytd_issue_units=5,
        purchases=[_flow(KIND_PI, 5, "PI9", cost=8)],
        sales=[],
        rvs=[],
        tars=[_flow(KIND_TAR, 5, "TAR6903-001")],
        adjustments=[],
    )
    assert product is not None
    assert product.qty == 0
    assert product.layers == []


def test_rv_replaces_non_vat_purchase_for_a_vat_sale():
    product = build_vat_product(
        bcode="D",
        descr="rv",
        actual_qty=0,
        pre_receipts=[_receipt(date(2025, 1, 1), 9, False, "NPI")],
        ytd_purchase_units=0,
        ytd_issue_units=4,
        purchases=[],
        sales=[_flow(KIND_SALE, 4, "TR9")],
        rvs=[_flow(KIND_RV, 4, "RV6903-001", cost=3)],
        tars=[],
        adjustments=[],
    )
    assert product is not None
    assert product.open_qty == 0
    assert product.qty == 0


def test_stock_adjustment_changes_vat_qty():
    product = build_vat_product(
        bcode="E",
        descr="counted",
        actual_qty=7,
        pre_receipts=[],
        ytd_purchase_units=10,
        ytd_issue_units=3,
        purchases=[_flow(KIND_PI, 10, "PI2", cost=2)],
        sales=[],
        rvs=[],
        tars=[],
        adjustments=[_flow(KIND_SA, 3, "SA2608-00001")],
    )
    assert product is not None
    assert product.qty == 7
    assert product.sa_qty == 3
    assert product.movements[-1].kind == KIND_SA
    assert product.movements[-1].qty == -3
    assert abs(product.movements[-1].balance - 7) < 1e-9


def test_stock_adjustment_without_vat_flow_stays_out():
    assert (
        build_vat_product(
            bcode="Z",
            descr="bad count",
            actual_qty=100,
            pre_receipts=[],
            ytd_purchase_units=0,
            ytd_issue_units=-100,
            purchases=[],
            sales=[],
            rvs=[],
            tars=[],
            adjustments=[_flow(KIND_SA, -100, "SA2610-00265")],
        )
        is None
    )
    assert (
        build_vat_product(
            bcode="7001",
            descr="labor",
            actual_qty=1,
            pre_receipts=[],
            ytd_purchase_units=1,
            ytd_issue_units=0,
            purchases=[_flow(KIND_PI, 1, "PI", cost=1)],
            sales=[],
            rvs=[],
            tars=[],
            adjustments=[],
        )
        is None
    )


def test_sale_unit_price_is_net_per_smallest_unit():
    assert abs(sale_unit_price(amount=214, units=2, taxic="Y") - 100) < 1e-6
    assert abs(sale_unit_price(amount=0, units=2, taxic="Y", price=214, mtp=2) - 100) < 1e-6
    assert sale_unit_price(amount=0, units=1, taxic="N", price=0) is None


def test_sale_cost_traces_to_the_older_purchase_bill():
    product = build_vat_product(
        bcode="F",
        descr="fifo",
        actual_qty=14,
        pre_receipts=[],
        ytd_purchase_units=20,
        ytd_issue_units=6,
        purchases=[
            _flow(KIND_PI, 10, "PI-OLD", day=1, cost=4),
            _flow(KIND_PI, 10, "PI-NEW", day=10, cost=9),
        ],
        sales=[_flow(KIND_SALE, 6, "TR1", day=15, sale_price=20)],
        rvs=[],
        tars=[],
        adjustments=[],
    )
    assert product is not None
    assert product.qty == 14
    assert product.layers[0].billno == "PI-OLD"
    assert product.layers[0].qty == 4
    assert product.layers[1].billno == "PI-NEW"
    sale = next(row for row in product.movements if row.kind == KIND_SALE)
    assert sale.unit_cost == 4
    assert sale.cost_billno == "PI-OLD"
    assert sale.sale_price == 20


def test_sale_cost_lists_every_purchase_bill_it_consumes():
    product = build_vat_product(
        bcode="G",
        descr="split",
        actual_qty=3,
        pre_receipts=[],
        ytd_purchase_units=10,
        ytd_issue_units=7,
        purchases=[
            _flow(KIND_PI, 5, "PI-A", day=1, cost=4),
            _flow(KIND_PI, 5, "PI-B", day=5, cost=6),
        ],
        sales=[_flow(KIND_SALE, 7, "TR2", day=10)],
        rvs=[],
        tars=[],
        adjustments=[],
    )
    assert product is not None
    sale = next(row for row in product.movements if row.kind == KIND_SALE)
    assert abs(sale.unit_cost - ((5 * 4 + 2 * 6) / 7)) < 1e-9
    assert sale.cost_billno == "PI-A, PI-B"


def test_movement_keeps_the_transaction_sale_price():
    product = build_vat_product(
        bcode="F",
        descr="priced",
        actual_qty=1,
        pre_receipts=[],
        ytd_purchase_units=2,
        ytd_issue_units=1,
        purchases=[_flow(KIND_PI, 2, "PI3", cost=6)],
        sales=[_flow(KIND_SALE, 1, "TR3", sale_price=15)],
        rvs=[],
        tars=[],
        adjustments=[],
    )
    assert product is not None
    sale = next(row for row in product.movements if row.kind == KIND_SALE)
    assert sale.sale_price == 15
    assert sale.unit_cost == 6
    assert sale.cost_billno == "PI3"
    purchase = next(row for row in product.movements if row.kind == KIND_PI)
    assert purchase.unit_cost == 6
    assert purchase.sale_price is None
