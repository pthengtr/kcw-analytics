"""QTYMIN recommendation rules. No Spark and no PARTS9."""

from src.kcw.product_insight_derived import build_derived
from src.kcw.product_insight_qtymin import (
    normalize_model_decision,
    pick_idle_kind,
    review_is_due,
    screen_qtymin,
)


def _facts(**stock):
    return {
        "recent": {
            "last_30d": {"customer_qty": 4, "typical_monthly_qty": 4, "channel_qty": {"syp_store": 4}},
            "last_90d": {"customer_qty": 12, "typical_monthly_qty": 4, "channel_qty": {"syp_store": 12}},
            "last_12m": {
                "customer_qty": 24,
                "typical_monthly_qty": 2,
                "channel_qty": {"syp_store": 24},
            },
        },
        "master": {"MTP2": 1},
        "stock": stock,
        "purchase_summary": {},
        "margin": {},
    }


def test_company_insight_still_zeros_a_negative_hq_minimum():
    derived = build_derived(
        _facts(hq={"qtyoh2": 0, "qtymin": -1}, syp={"qtyoh2": 3, "qtymin": 0})
    )
    assert derived["flags"]["rec_qtymin"] == 0.0


def test_syp_branch_insight_keeps_a_holding_when_hq_is_blocked():
    derived = build_derived(
        _facts(hq={"qtyoh2": 0, "qtymin": -1}, syp={"qtyoh2": 3, "qtymin": -1}),
        policy_branch="syp",
    )
    assert derived["flags"]["rec_qtymin"] > 0


def test_screen_skips_positive_and_sends_sales_to_the_model():
    assert screen_qtymin(4, 10) == "skip"
    assert screen_qtymin(0, 0) == "rule"
    assert screen_qtymin(-1, 0) == "rule"
    assert screen_qtymin(0, 3) == "model"
    assert screen_qtymin(-1, 3) == "model"
    assert screen_qtymin(None, 3) == "skip"


def test_positive_recommendation_is_not_reopened_when_the_insight_changes():
    existing = {"decision": "set_positive", "insight_generated_at": "2026-10-01T00:00:00+07:00"}
    assert review_is_due(existing, insight_generated_at="2026-10-22T00:00:00+07:00", screen_kind="model") is False


def test_do_not_order_is_due_again_on_a_newer_insight():
    existing = {"decision": "set_do_not_order", "insight_generated_at": "2026-10-01T00:00:00+07:00"}
    assert review_is_due(existing, insight_generated_at="2026-10-01T00:00:00+07:00", screen_kind="rule") is False
    assert review_is_due(existing, insight_generated_at="2026-10-22T00:00:00+07:00", screen_kind="model") is True


def test_model_must_choose_minus_one_or_a_positive():
    stay = normalize_model_decision(
        {"decision": "set_do_not_order", "suggested_qtymin": 9, "why": "คง -1"},
        current_qtymin=-1,
    )
    assert stay["suggested_qtymin"] == -1.0
    move = normalize_model_decision(
        {"decision": "set_positive", "suggested_qtymin": 4, "why": "ยังขาย", "watch": ["still_selling", "nope"]},
        current_qtymin=0,
    )
    assert move["suggested_qtymin"] == 4
    assert move["watch"] == ["still_selling"]
    try:
        normalize_model_decision(
            {"decision": "set_positive", "suggested_qtymin": 0, "why": "ศูนย์"},
            current_qtymin=0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("zero is not a positive minimum")


def test_idle_alternates_and_falls_through():
    kind, nxt = pick_idle_kind("syp", syp_pending=True, qtymin_pending=True)
    assert (kind, nxt) == ("syp", "qtymin")
    kind, nxt = pick_idle_kind("qtymin", syp_pending=True, qtymin_pending=True)
    assert (kind, nxt) == ("qtymin", "syp")
    kind, nxt = pick_idle_kind("syp", syp_pending=False, qtymin_pending=True)
    assert kind == "qtymin"
    assert pick_idle_kind("syp", syp_pending=False, qtymin_pending=False)[0] is None
