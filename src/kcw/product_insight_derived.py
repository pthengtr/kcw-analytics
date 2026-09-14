"""Deterministic demand / holding / trend / margin facts for product insights.

The model must cite these as origin; SQLite query columns prefer these numbers
over free-text so ops can GROUP BY enums.
"""

from __future__ import annotations

import math
from typing import Any

CUSTOMER_CHANNELS = ("hq_store", "syp_store", "online")
WEEKS_PER_MONTH = 4.345

TREND_LABELS = ("hot", "growing", "flat", "declining", "dead", "lumpy", "seasonal", "unknown")
MARGIN_FLAGS = ("healthy", "thin", "weak", "negative", "cost_up_price_lag", "unknown")
ORDER_OK = ("yes", "caution", "no")
DEAD_STOCK = ("yes", "no", "maybe")
STOCK_ANOMALY = ("none", "negative", "do_not_restock", "other")


def _f(v: Any, default: float | None = None) -> float | None:
    if v is None or v == "":
        return default
    try:
        n = float(v)
    except (TypeError, ValueError):
        return default
    if n != n:  # NaN
        return default
    return n


def _round(v: float | None, nd: int = 4) -> float | None:
    if v is None:
        return None
    return round(float(v), nd)


def suggested_cover_weeks(monthly_qty: float | None) -> float | None:
    """Policy cover from company customer demand (small units / month)."""
    m = _f(monthly_qty)
    if m is None:
        return None
    if m <= 0:
        return None
    if m >= 20:
        return 4.0
    if m >= 5:
        return 6.0
    if m >= 1:
        return 8.0
    return 12.0


def safe_holding_qty(monthly_qty: float | None, cover_weeks: float | None) -> float | None:
    m = _f(monthly_qty)
    w = _f(cover_weeks)
    if m is None or w is None or m < 0 or w <= 0:
        return None
    return _round(m * (w / WEEKS_PER_MONTH), 2)


def pack_order(need_small: float | None, mtp2: float | None) -> dict[str, Any]:
    """Round a small-unit need up to whole large packs when MTP2 > 1."""
    need = _f(need_small, 0.0) or 0.0
    mtp = _f(mtp2)
    if need <= 0:
        return {
            "order_qty": 0.0,
            "order_qty_large": 0.0 if (mtp and mtp > 1) else None,
            "rounded_from": 0.0,
            "mtp2": mtp,
        }
    if mtp is None or mtp <= 1:
        q = math.ceil(need - 1e-9)
        return {
            "order_qty": float(q),
            "order_qty_large": None,
            "rounded_from": _round(need, 2),
            "mtp2": mtp,
        }
    packs = int(math.ceil(need / mtp - 1e-9))
    return {
        "order_qty": float(packs * mtp),
        "order_qty_large": float(packs),
        "rounded_from": _round(need, 2),
        "mtp2": mtp,
    }


def classify_trend(*, qty_window: float, monthly_ref: float | None, window_days: int) -> str:
    """Compare a short window's monthly equivalent to a 12m monthly baseline."""
    q = _f(qty_window, 0.0) or 0.0
    ref = _f(monthly_ref)
    if q <= 0 and (ref is None or ref <= 0):
        return "dead"
    months = max(window_days / 30.4375, 1 / 30.4375)
    monthly_eq = q / months
    if ref is None or ref <= 0:
        return "lumpy" if q > 0 else "dead"
    ratio = monthly_eq / ref
    if ratio >= 1.8:
        return "hot"
    if ratio >= 1.25:
        return "growing"
    if ratio >= 0.75:
        return "flat"
    if q <= 0:
        return "dead"
    return "declining"


def classify_margin(
    *,
    list_pct: float | None,
    realized_pct: float | None,
    prior_pct: float | None,
    cost_change_pct: float | None,
    price_change_pct: float | None,
) -> str:
    cost_up = (_f(cost_change_pct) or 0) >= 10
    price_lag = (_f(price_change_pct) or 0) < 3
    if cost_up and price_lag:
        return "cost_up_price_lag"
    pct = _f(realized_pct)
    if pct is None:
        pct = _f(list_pct)
    if pct is None:
        return "unknown"
    if pct < 0:
        return "negative"
    if pct < 10:
        return "weak"
    if pct < 25:
        return "thin"
    return "healthy"


def _avg(vals: list[float]) -> float | None:
    if not vals:
        return None
    return sum(vals) / len(vals)


def _pct_change(new: float | None, old: float | None) -> float | None:
    n, o = _f(new), _f(old)
    if n is None or o is None or o == 0:
        return None
    return _round((n - o) / abs(o) * 100.0, 2)


def _margin_pct(price: float | None, cost: float | None) -> float | None:
    p, c = _f(price), _f(cost)
    if p is None or c is None or p == 0:
        return None
    return _round((p - c) / p * 100.0, 2)


def build_derived(facts: dict[str, Any]) -> dict[str, Any]:
    """Assemble origin-traced numbers the prompt and SQLite columns share."""
    recent = facts.get("recent") or {}
    last_30 = recent.get("last_30d") or {}
    last_90 = recent.get("last_90d") or {}
    last_12 = recent.get("last_12m") or {}
    master = facts.get("master") or {}
    stock = facts.get("stock") or {}
    purchase = facts.get("purchase_summary") or {}
    margin_in = facts.get("margin") or {}

    qty_30 = _f(last_30.get("customer_qty"), 0.0) or 0.0
    qty_90 = _f(last_90.get("customer_qty"), 0.0) or 0.0
    qty_12 = _f(last_12.get("customer_qty"), 0.0) or 0.0
    monthly_12 = _f(last_12.get("typical_monthly_qty"))
    monthly_90 = _f(last_90.get("typical_monthly_qty"))
    monthly_30 = _f(last_30.get("typical_monthly_qty"))
    ch12 = last_12.get("channel_qty") or {}
    ch90 = last_90.get("channel_qty") or {}

    cover = suggested_cover_weeks(monthly_12)
    hold = safe_holding_qty(monthly_12, cover)

    hq = stock.get("hq") or {}
    syp = stock.get("syp") or {}
    qtyoh_hq = _f(hq.get("qtyoh2"))
    qtyoh_syp = _f(syp.get("qtyoh2"))
    qtymin_hq = _f(hq.get("qtymin"))
    qtymin_syp = _f(syp.get("qtymin"))
    company_oh = None
    if qtyoh_hq is not None or qtyoh_syp is not None:
        company_oh = (qtyoh_hq or 0.0) + (qtyoh_syp or 0.0)

    mtp2 = _f(master.get("MTP2"))
    # Standing replenishment (valid 14–30d): typical PO size to refill to target
    # when live QTYOH hits rec_qtymin. Do NOT subtract snapshot QTYOH.
    packed = pack_order(hold, mtp2)

    syp_monthly = _f((ch12.get("syp_store")), 0.0)
    if syp_monthly is not None:
        syp_monthly = syp_monthly / 12.0
    syp_cover = suggested_cover_weeks(syp_monthly) if (syp_monthly or 0) > 0 else None
    syp_hold = safe_holding_qty(syp_monthly, syp_cover)
    syp_packed = pack_order(syp_hold, mtp2)
    do_not_syp = qtymin_syp is not None and qtymin_syp < 0
    xfer_qty = None
    if do_not_syp or (syp_monthly or 0) <= 0:
        xfer_qty = 0.0
    elif syp_hold is not None:
        # Typical HQ→SYP batch to bring SYP up to its target (not snap-gap).
        xfer_qty = syp_packed["order_qty"] if syp_packed["order_qty"] else syp_hold

    list_pct = _margin_pct(_f(master.get("PRICE1")), _f(master.get("COSTAVG")))
    realized = _f(margin_in.get("realized_pct_12m"))
    prior = _f(margin_in.get("realized_pct_prior_12m"))
    cost_chg = _f(margin_in.get("buy_cost_change_pct_12m"))
    price_chg = _f(margin_in.get("sell_price_change_pct_12m"))
    margin_flag = classify_margin(
        list_pct=list_pct,
        realized_pct=realized,
        prior_pct=prior,
        cost_change_pct=cost_chg,
        price_change_pct=price_chg,
    )
    delta_pp = None
    if realized is not None and prior is not None:
        delta_pp = _round(realized - prior, 2)

    trend_30 = classify_trend(qty_window=qty_30, monthly_ref=monthly_12, window_days=30)
    trend_90 = classify_trend(qty_window=qty_90, monthly_ref=monthly_12, window_days=90)
    if qty_12 <= 0:
        trend_12 = "dead"
    elif monthly_90 is not None and monthly_12 and monthly_12 > 0:
        r = monthly_90 / monthly_12
        if r >= 1.25:
            trend_12 = "growing"
        elif r >= 0.75:
            trend_12 = "flat"
        else:
            trend_12 = "declining"
    else:
        trend_12 = "unknown"

    do_not_hq = qtymin_hq is not None and qtymin_hq < 0
    neg_hq = qtyoh_hq is not None and qtyoh_hq < 0
    neg_syp = qtyoh_syp is not None and qtyoh_syp < 0
    if neg_hq or neg_syp:
        stock_anom = "negative"
    elif do_not_hq or (qtymin_syp is not None and qtymin_syp < 0):
        stock_anom = "do_not_restock"
    else:
        stock_anom = "none"

    dead = "yes" if qty_12 <= 0 and qty_90 <= 0 else ("maybe" if qty_12 <= 0 else "no")
    # order_ok = restock *policy* for the insight lifetime (14–30d), not "PO today vs QTYOH".
    if do_not_hq or dead == "yes":
        order_ok = "no"
    elif dead == "maybe" or trend_90 == "declining" or margin_flag in ("negative", "weak"):
        order_ok = "caution"
    elif (monthly_12 or 0) > 0:
        order_ok = "yes"
    else:
        order_ok = "caution"

    if do_not_hq or dead == "yes":
        rec_qtymin = 0.0
        packed = pack_order(0.0, mtp2)
        xfer_qty = 0.0
    else:
        trigger_weeks = None
        if cover is not None:
            trigger_weeks = 2.0 if cover >= 4 else max(cover * 0.5, 1.0)
        rec_qtymin = safe_holding_qty(monthly_12, trigger_weeks) if trigger_weeks else hold
    check_stock = "yes" if stock_anom in ("negative", "do_not_restock") else "no"

    return {
        "demand": {
            "customer_channels": list(CUSTOMER_CHANNELS),
            "formula": (
                "customer_qty = hq_store + syp_store + online; "
                "transfer and excluded (JOURMODE=0) are not demand. "
                "typical_monthly_qty = customer_qty / (days/30.4375)."
            ),
            "qty_30d": _round(qty_30, 2),
            "qty_90d": _round(qty_90, 2),
            "qty_12m": _round(qty_12, 2),
            "monthly_from_30d": _round(monthly_30, 4),
            "monthly_from_90d": _round(monthly_90, 4),
            "monthly_from_12m": _round(monthly_12, 4),
            "by_channel_12m": {k: _round(v, 2) for k, v in sorted((ch12 or {}).items())},
            "by_channel_90d": {k: _round(v, 2) for k, v in sorted((ch90 or {}).items())},
        },
        "holding": {
            "formula": (
                f"cover_weeks from monthly_12m policy; "
                f"safe_holding_qty = monthly_12m * cover_weeks / {WEEKS_PER_MONTH} "
                "(weeks, not months)."
            ),
            "cover_weeks": cover,
            "safe_holding_qty": hold,
            "typical_monthly_qty": _round(monthly_12, 4),
        },
        "stock": {
            "qtyoh_hq": qtyoh_hq,
            "qtyoh_syp": qtyoh_syp,
            "qtymin_hq": qtymin_hq,
            "qtymin_syp": qtymin_syp,
            "company_qtyoh": _round(company_oh, 2) if company_oh is not None else None,
            "as_of_note": (
                "QTYOH2/QTYMIN are snapshot-only (facts_as_of). "
                "Insight is a 14–30 day policy — do not treat snap QTYOH as live or as order-now."
            ),
            "anomaly": stock_anom,
        },
        "order": {
            "formula": (
                "Standing replenishment for 14–30 days: when *live* QTYOH2 ≤ rec_qtymin, "
                "order pack-rounded safe_holding_qty (MTP2). Not (safe_holding - snap QTYOH)."
            ),
            "order_qty": packed["order_qty"],
            "order_qty_large": packed["order_qty_large"],
            "mtp2": packed["mtp2"],
            "ui1": master.get("UI1"),
            "ui2": master.get("UI2"),
            "last_supplier": purchase.get("last_supplier"),
            "last_buy_price": purchase.get("last_price"),
            "last_buy_date": purchase.get("last_date"),
            "last_buy_src": purchase.get("last_src_site"),
        },
        "transfer": {
            "formula": (
                "SYP target = syp_store last_12m/12 × SYP cover weeks / 4.345, pack-rounded. "
                "Typical HQ→SYP batch when live SYP QTYOH2 is below that target. "
                "0 if SYP QTYMIN<0 or no syp_store sales. Do not use snap QTYOH gap."
            ),
            "syp_monthly": _round(syp_monthly, 4),
            "syp_safe_holding": syp_hold,
            "rec_transfer_qty_to_syp": _round(xfer_qty, 2) if xfer_qty is not None else None,
        },
        "trend": {
            "30d": trend_30,
            "90d": trend_90,
            "12m": trend_12,
            "formula": (
                "30d/90d label compares window monthly-equivalent to 12m monthly. "
                "hot>=1.8x growing>=1.25x flat 0.75–1.25 declining<0.75 dead=no customer qty."
            ),
        },
        "margin": {
            "formula": (
                "list_pct = (PRICE1-COSTAVG)/PRICE1*100; "
                "realized_12m = (avg customer sell - avg PI buy) / avg sell * 100 "
                "over last 12 months vs prior 12 months."
            ),
            "list_pct": list_pct,
            "realized_pct_12m": realized,
            "realized_pct_prior_12m": prior,
            "delta_pp": delta_pp,
            "cost_change_pct_12m": cost_chg,
            "price_change_pct_12m": price_chg,
            "flag": margin_flag,
        },
        "flags": {
            "order_ok": order_ok,
            "dead_stock": dead,
            "check_stock": check_stock,
            "stock_anomaly": stock_anom,
            "rec_qtymin": rec_qtymin,
        },
    }


def _num(v: Any, fallback: Any = None) -> float | None:
    n = _f(v)
    if n is None:
        return _f(fallback)
    return n


def _enum(v: Any, fallback: Any, allowed: tuple[str, ...]) -> str | None:
    s = str(v).strip().lower() if v is not None and str(v).strip() else ""
    if s in allowed:
        return s
    fb = str(fallback).strip().lower() if fallback is not None else ""
    return fb if fb in allowed else None


def flatten_insight_columns(insight: dict[str, Any] | None, derived: dict[str, Any]) -> dict[str, Any]:
    """Map model JSON + derived facts onto product_insights query columns."""
    i = insight if isinstance(insight, dict) else {}
    d = derived or {}
    dem = d.get("demand") or {}
    hold = d.get("holding") or {}
    st = d.get("stock") or {}
    od = d.get("order") or {}
    trn = d.get("transfer") or {}
    td = d.get("trend") or {}
    mg = d.get("margin") or {}
    fl = d.get("flags") or {}

    purch = i.get("purchase") if isinstance(i.get("purchase"), dict) else {}
    icmas = i.get("icmas") if isinstance(i.get("icmas"), dict) else {}
    xfer = i.get("transfer") if isinstance(i.get("transfer"), dict) else {}
    trends = i.get("trends") if isinstance(i.get("trends"), dict) else {}
    margin = i.get("margin") if isinstance(i.get("margin"), dict) else {}

    return {
        "typical_monthly_qty": _num(i.get("typical_monthly_qty"), hold.get("typical_monthly_qty")),
        "suggested_cover_weeks": _num(i.get("suggested_cover_weeks"), hold.get("cover_weeks")),
        "safe_holding_qty": _num(i.get("safe_holding_qty"), hold.get("safe_holding_qty")),
        "safe_holding_reason": (purch.get("safe_holding_reason") or i.get("demand_hint") or "")[:500] or None,
        "order_ok": _enum(purch.get("order_ok"), fl.get("order_ok"), ORDER_OK),
        "order_ok_reason": (purch.get("order_ok_reason") or "")[:500] or None,
        "dead_stock": _enum(i.get("dead_stock"), fl.get("dead_stock"), DEAD_STOCK),
        "dead_stock_reason": (i.get("dead_stock_reason") or "")[:500] or None,
        "suggested_order_qty": _num(purch.get("suggested_order_qty"), od.get("order_qty")),
        "suggested_order_qty_large": _num(purch.get("suggested_order_qty_large"), od.get("order_qty_large")),
        "order_unit": purch.get("order_unit") or od.get("ui1"),
        "order_unit_large": purch.get("order_unit_large") or od.get("ui2"),
        "last_supplier": purch.get("last_supplier") or od.get("last_supplier"),
        "last_buy_price": _num(purch.get("last_buy_price"), od.get("last_buy_price")),
        "last_buy_date": purch.get("last_buy_date") or od.get("last_buy_date"),
        "rec_qtymin": _num(icmas.get("rec_qtymin"), fl.get("rec_qtymin")),
        "rec_qtymin_reason": (icmas.get("rec_qtymin_reason") or "")[:500] or None,
        "check_stock": _enum(icmas.get("check_stock"), fl.get("check_stock"), ("yes", "no")),
        "stock_anomaly": _enum(icmas.get("stock_anomaly"), fl.get("stock_anomaly"), STOCK_ANOMALY),
        "qtyoh_hq": _num(st.get("qtyoh_hq")),
        "qtyoh_syp": _num(st.get("qtyoh_syp")),
        "qtymin_hq": _num(st.get("qtymin_hq")),
        "qtymin_syp": _num(st.get("qtymin_syp")),
        "rec_transfer_qty_to_syp": _num(xfer.get("qty"), trn.get("rec_transfer_qty_to_syp")),
        "rec_transfer_reason": (xfer.get("reason") or "")[:500] or None,
        "sales_qty_30d": _num(dem.get("qty_30d")),
        "sales_qty_90d": _num(dem.get("qty_90d")),
        "sales_qty_12m": _num(dem.get("qty_12m")),
        "trend_30d": _enum(trends.get("d30"), td.get("30d"), TREND_LABELS),
        "trend_90d": _enum(trends.get("d90"), td.get("90d"), TREND_LABELS),
        "trend_12m": _enum(trends.get("m12"), td.get("12m"), TREND_LABELS),
        "trend_label": _enum(i.get("trend_label"), td.get("12m"), TREND_LABELS),
        "margin_pct_list": _num(margin.get("list_pct"), mg.get("list_pct")),
        "margin_pct_12m": _num(mg.get("realized_pct_12m")),
        "margin_pct_prior_12m": _num(mg.get("realized_pct_prior_12m")),
        "margin_delta_pp": _num(margin.get("delta_pp"), mg.get("delta_pp")),
        "margin_flag": _enum(margin.get("flag"), mg.get("flag"), MARGIN_FLAGS),
        "cost_change_pct_12m": _num(mg.get("cost_change_pct_12m")),
        "price_change_pct_12m": _num(mg.get("price_change_pct_12m")),
    }
