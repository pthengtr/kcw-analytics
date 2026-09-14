"""Unit tests for product insight helpers (no PARTS9 / Spark required)."""

from src.kcw.product_insight_channels import billtype_std, channel_of
from src.kcw.product_insight_derived import (
    build_derived,
    classify_margin,
    classify_trend,
    flatten_insight_columns,
    pack_order,
    safe_holding_qty,
    suggested_cover_weeks,
)
from src.kcw.product_insight_generate import parse_window, build_fact_pack, _is_vendor_purchase
from datetime import date
import sqlite3


def test_channel_online_and_transfer():
    assert channel_of("TAD6901-001", "1") == "online"
    assert channel_of("TFV6808-012", "1") == "transfer"
    assert channel_of("TF6808-001", "2") == "transfer"
    assert channel_of("8K69-001", "2") == "hq_store"
    assert channel_of("ANY", "0") == "excluded"
    assert channel_of("3K68-0000001", "2") == "syp_store"
    assert channel_of("K68-0001", "2", src_site="syp") == "syp_store"


def test_billtype_std_prefixes():
    assert billtype_std("TAD1") == "TAD"
    assert billtype_std("CNTAD1") == "CNTAD"
    assert billtype_std("TFV1") == "TFV"


def test_vendor_purchase_excludes_transfers():
    assert _is_vendor_purchase("IV26080195") is True
    assert _is_vendor_purchase("TFV6908-091") is False
    assert _is_vendor_purchase("TF6808-001") is False
    assert _is_vendor_purchase("3TF6901-1") is False


def test_parse_window():
    base = date(2026, 9, 13)
    assert parse_window("5y", as_of=base).isoformat() == "2021-09-14"
    assert parse_window("14d", as_of=base).isoformat() == "2026-08-30"
    assert parse_window("2w", as_of=base).isoformat() == "2026-08-30"


def test_holding_and_pack():
    assert suggested_cover_weeks(30) == 4.0
    assert suggested_cover_weeks(8) == 6.0
    hold = safe_holding_qty(57, 4)
    assert hold is not None and abs(hold - 52.47) < 0.1
    packed = pack_order(15, 10)
    assert packed["order_qty_large"] == 2
    assert packed["order_qty"] == 20


def test_classify_trend_and_margin():
    assert classify_trend(qty_window=0, monthly_ref=0, window_days=30) == "dead"
    assert classify_trend(qty_window=40, monthly_ref=10, window_days=30) == "hot"
    assert classify_trend(qty_window=10, monthly_ref=10, window_days=30) in ("flat", "growing")
    assert classify_margin(
        list_pct=30, realized_pct=28, prior_pct=30, cost_change_pct=12, price_change_pct=1
    ) == "cost_up_price_lag"
    assert classify_margin(
        list_pct=-5, realized_pct=-5, prior_pct=10, cost_change_pct=0, price_change_pct=0
    ) == "negative"


def test_flatten_prefers_derived_numbers():
    derived = build_derived(
        {
            "recent": {
                "last_30d": {"customer_qty": 10, "typical_monthly_qty": 10, "channel_qty": {"hq_store": 10}},
                "last_90d": {"customer_qty": 30, "typical_monthly_qty": 10, "channel_qty": {"hq_store": 30}},
                "last_12m": {
                    "customer_qty": 120,
                    "typical_monthly_qty": 10,
                    "channel_qty": {"hq_store": 80, "syp_store": 30, "online": 10},
                },
            },
            "master": {"UI1": "ชิ้น", "UI2": "กล่อง", "MTP2": 10, "PRICE1": 100, "COSTAVG": 70},
            "stock": {"hq": {"qtyoh2": 5, "qtymin": 0}, "syp": {"qtyoh2": 2, "qtymin": 0}},
            "purchase_summary": {
                "last_supplier": "ACME",
                "last_price": 69,
                "last_date": "2026-08-01",
                "last_src_site": "hq",
            },
            "margin": {
                "realized_pct_12m": 28.0,
                "realized_pct_prior_12m": 32.0,
                "buy_cost_change_pct_12m": 4.0,
                "sell_price_change_pct_12m": 1.0,
            },
        }
    )
    cols = flatten_insight_columns({}, derived)
    assert cols["sales_qty_12m"] == 120
    assert cols["safe_holding_qty"] is not None
    assert cols["order_ok"] == "yes"  # demand policy, not snap overstock
    assert cols["suggested_order_qty"] == 20  # pack-round hold 13.8 → 2×10, not gap vs QTYOH 7
    assert cols["rec_qtymin"] is not None and cols["rec_qtymin"] < cols["safe_holding_qty"]
    assert cols["rec_transfer_qty_to_syp"] == 10  # SYP target pack, not 74-147 gap
    assert cols["margin_pct_list"] == 30.0
    assert cols["trend_12m"] in ("flat", "growing", "declining", "hot", "dead", "lumpy", "unknown")
    assert cols["last_supplier"] == "ACME"


def test_build_fact_pack_old_snap_shape():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE icmas (bcode TEXT PRIMARY KEY, descr TEXT, brand TEXT, model TEXT,
          pcode TEXT, mcode TEXT, acode TEXT, size1 TEXT, ui1 TEXT, ui2 TEXT, mtp2 REAL,
          status TEXT, location1 TEXT, costavg REAL, costlast REAL, price1 REAL);
        CREATE TABLE sidet (bcode TEXT, billno TEXT, billdate TEXT, qty REAL, ui TEXT,
          price REAL, amount REAL, jourmode TEXT);
        CREATE TABLE pidet (bcode TEXT, billno TEXT, billdate TEXT, qty REAL, ui TEXT,
          price REAL, amount REAL, billtype TEXT);
        INSERT INTO meta VALUES ('site','hq'), ('facts_as_of','2026-09-14T00:00:00+07:00');
        INSERT INTO icmas (bcode, descr, ui1, mtp2, costavg, price1)
          VALUES ('X1', 'Bolt', 'ชิ้น', 10, 70, 100);
        INSERT INTO sidet VALUES ('X1','8K69-1','2026-08-01',5,'ชิ้น',100,500,'2');
        INSERT INTO pidet VALUES ('X1','P1','2026-07-01',10,'ชิ้น',70,700,'PI');
        """
    )
    facts = build_fact_pack(conn, site="hq", bcode="X1", facts_as_of="2026-09-14T00:00:00+07:00")
    assert "derived" in facts
    assert facts["recent"]["last_30d"]["customer_qty"] >= 0
    assert facts["derived"]["demand"]["qty_12m"] == 5
    conn.close()
