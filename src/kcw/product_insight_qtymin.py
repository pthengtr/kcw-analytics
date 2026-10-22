"""Recommend QTYMIN -1 or a positive minimum from a stored insight.

Nothing here writes ICMAS. A current positive QTYMIN is left for the user.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from src.kcw.product_insight_store import (
    connect_insights,
    init_insights_schema,
    insight_qtymin_slice,
    iter_insight_qtymin_slices,
    qtymin_review_marks,
    upsert_qtymin_review,
    utc_now_iso,
)

QTYMIN_WINDOW = "qtymin"
WATCH = (
    "still_selling",
    "one_off_sale",
    "dead_with_stock",
    "weak_margin",
    "other_branch_disagrees",
    "insight_min_was_forced_zero",
)


def _prompt_path() -> Path:
    return Path(__file__).resolve().parents[2] / "prompts" / "qtymin_review_v1.yaml"


def load_qtymin_prompt() -> dict[str, Any]:
    data = yaml.safe_load(_prompt_path().read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or not data.get("system"):
        raise RuntimeError(f"invalid prompt file: {_prompt_path()}")
    return data


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def current_meaning(qtymin: float | None) -> str | None:
    if qtymin is None:
        return None
    if qtymin < 0:
        return "do_not_order"
    if qtymin == 0:
        return "unset"
    return "positive"


def screen_qtymin(current_qtymin: float | None, branch_qty_12m: float | None) -> str:
    """skip | rule | model. Only 0 and negative values are analyzed."""
    if current_qtymin is None or current_qtymin > 0:
        return "skip"
    if (_num(branch_qty_12m) or 0) <= 0:
        return "rule"
    return "model"


def review_is_due(
    existing: dict[str, Any] | None,
    *,
    insight_generated_at: str | None,
    screen_kind: str,
) -> bool:
    """A positive recommendation stays closed. -1 is due again on a newer insight."""
    if screen_kind == "skip":
        return False
    if not existing:
        return True
    if existing.get("decision") == "set_positive":
        return False
    return existing.get("insight_generated_at") != insight_generated_at


def pick_idle_kind(
    prefer: str,
    *,
    syp_pending: bool,
    qtymin_pending: bool,
) -> tuple[str | None, str]:
    """Choose syp insight or qtymin, then flip the preference for the next claim."""
    nxt = "qtymin" if prefer != "qtymin" else "syp"
    order = ("syp", "qtymin") if prefer != "qtymin" else ("qtymin", "syp")
    open_kind = {"syp": syp_pending, "qtymin": qtymin_pending}
    for kind in order:
        if open_kind[kind]:
            return kind, nxt
    return None, nxt


def rule_why(current_qtymin: float | None) -> str:
    if current_qtymin == 0:
        return "ไม่มียอดขายสาขานี้ใน 12 เดือน ค่า 0 คือยังไม่ตั้งจุดสั่ง จึงแนะนำเป็น -1 ไม่สั่งซื้อ"
    return "ไม่มียอดขายสาขานี้ใน 12 เดือน ควรคง -1 ไม่สั่งซื้อ"


def rule_review(
    *,
    current_qtymin: float | None,
    qtyoh: float | None,
    insight_generated_at: str | None,
) -> dict[str, Any]:
    watch: list[str] = []
    if qtyoh is not None and qtyoh > 0:
        watch.append("dead_with_stock")
    return {
        "current_qtymin": current_qtymin,
        "current_meaning": current_meaning(current_qtymin) or "unset",
        "decision": "set_do_not_order",
        "suggested_qtymin": -1.0,
        "why": rule_why(current_qtymin),
        "watch": watch,
        "insight_generated_at": insight_generated_at,
        "source": "rule",
    }


def normalize_model_decision(payload: dict[str, Any], *, current_qtymin: float | None) -> dict[str, Any]:
    decision = str(payload.get("decision") or "").strip()
    if decision not in ("set_do_not_order", "set_positive"):
        raise ValueError(f"decision must be set_do_not_order or set_positive, got {decision!r}")
    if decision == "set_do_not_order":
        suggested = -1.0
    else:
        suggested = _num(payload.get("suggested_qtymin"))
        if suggested is None or suggested <= 0:
            raise ValueError("set_positive requires suggested_qtymin > 0")
    watch_raw = payload.get("watch") or []
    if not isinstance(watch_raw, list):
        watch_raw = []
    watch = [str(item) for item in watch_raw if str(item) in WATCH]
    why = str(payload.get("why") or "").strip()
    if not why:
        raise ValueError("why is required")
    return {
        "current_qtymin": current_qtymin,
        "current_meaning": current_meaning(current_qtymin) or "unset",
        "decision": decision,
        "suggested_qtymin": suggested,
        "why": why[:500],
        "watch": watch,
        "source": "model",
    }


def slice_from_insight_row(row: dict[str, Any]) -> dict[str, Any]:
    qtymin = _num(row.get("current_qtymin"))
    rec = _num(row.get("rec_qtymin"))
    forced = bool(qtymin is not None and qtymin < 0 and rec == 0)
    return {
        "bcode": str(row.get("bcode") or "").strip(),
        "insight_generated_at": str(row.get("generated_at") or "") or None,
        "current_qtymin": qtymin,
        "qtyoh": _num(row.get("qtyoh")),
        "branch_qty_12m": _num(row.get("branch_qty_12m")) or 0.0,
        "dead_stock": row.get("dead_stock"),
        "order_ok": row.get("order_ok"),
        "margin_flag": row.get("margin_flag"),
        "rec_qtymin": rec,
        "insight_min_was_forced_zero": forced,
    }


def slice_from_facts(
    *,
    bcode: str,
    insight_generated_at: str,
    facts: dict[str, Any],
    branch: str,
) -> dict[str, Any]:
    derived = facts.get("derived") or {}
    stock = derived.get("stock") or {}
    demand = derived.get("demand") or {}
    flags = derived.get("flags") or {}
    margin = derived.get("margin") or {}
    key = "syp" if branch == "syp" else "hq"
    channel = "syp_store" if branch == "syp" else "hq_store"
    by_channel = demand.get("by_channel_12m") or {}
    qtymin = _num(stock.get(f"qtymin_{key}"))
    rec = _num(flags.get("rec_qtymin"))
    return slice_from_insight_row(
        {
            "bcode": bcode,
            "generated_at": insight_generated_at,
            "current_qtymin": qtymin,
            "qtyoh": stock.get(f"qtyoh_{key}"),
            "branch_qty_12m": by_channel.get(channel, demand.get("qty_12m")),
            "dead_stock": flags.get("dead_stock"),
            "order_ok": flags.get("order_ok"),
            "margin_flag": margin.get("flag"),
            "rec_qtymin": rec,
        }
    )


def _save_review(site: str, bcode: str, review: dict[str, Any], conn: Any = None) -> None:
    upsert_qtymin_review(
        site=site,
        bcode=bcode,
        current_qtymin=review["current_qtymin"],
        current_meaning=review["current_meaning"],
        decision=review["decision"],
        suggested_qtymin=review["suggested_qtymin"],
        why=review["why"],
        watch=review["watch"],
        insight_generated_at=review.get("insight_generated_at"),
        reviewed_at=utc_now_iso(),
        source=review["source"],
        conn=conn,
    )


def _enqueue(
    *,
    site: str,
    bcode: str,
    snap_id: str,
    insight_generated_at: str | None,
    score: int,
) -> None:
    now = utc_now_iso()
    conn = connect_insights()
    try:
        init_insights_schema(conn)
        conn.execute(
            """
            INSERT INTO insight_queue (
              site, bcode, snap_id, facts_as_of, window, movement_score,
              status, updated_at, leased_at, retry_count, last_error, due_tier
            ) VALUES (?,?,?,?,?,?, 'pending', ?, NULL, 0, NULL, 1)
            ON CONFLICT(site, bcode, snap_id, window) DO UPDATE SET
              facts_as_of=excluded.facts_as_of,
              movement_score=excluded.movement_score,
              status=CASE
                WHEN insight_queue.status='running' THEN insight_queue.status
                ELSE 'pending'
              END,
              updated_at=CASE
                WHEN insight_queue.status='running' THEN insight_queue.updated_at
                ELSE excluded.updated_at
              END,
              retry_count=CASE
                WHEN insight_queue.status='running' THEN insight_queue.retry_count
                ELSE 0
              END,
              last_error=NULL
            """,
            (site, bcode, snap_id, insight_generated_at or now, QTYMIN_WINDOW, score, now),
        )
        conn.commit()
    finally:
        conn.close()


def _close_queue(site: str, bcode: str, snap_id: str) -> None:
    conn = connect_insights()
    try:
        conn.execute(
            """
            UPDATE insight_queue
            SET status='done', leased_at=NULL, updated_at=?
            WHERE site=? AND bcode=? AND snap_id=? AND window=?
              AND status IN ('pending', 'running')
            """,
            (utc_now_iso(), site, bcode, snap_id, QTYMIN_WINDOW),
        )
        conn.commit()
    finally:
        conn.close()


def apply_screen(
    *,
    site: str,
    snap_id: str,
    item: dict[str, Any],
    existing: dict[str, Any] | None,
    conn: Any = None,
) -> str:
    """Write an obvious -1, enqueue a model job, or skip. Returns the action taken."""
    kind = screen_qtymin(item.get("current_qtymin"), item.get("branch_qty_12m"))
    bcode = item["bcode"]
    if not review_is_due(
        existing,
        insight_generated_at=item.get("insight_generated_at"),
        screen_kind=kind,
    ):
        if kind == "skip" or (existing and existing.get("decision") == "set_positive"):
            _close_queue(site, bcode, snap_id)
        return "skip"
    if kind == "rule":
        review = rule_review(
            current_qtymin=item.get("current_qtymin"),
            qtyoh=item.get("qtyoh"),
            insight_generated_at=item.get("insight_generated_at"),
        )
        _save_review(site, bcode, review, conn=conn)
        _close_queue(site, bcode, snap_id)
        return "rule"
    _enqueue(
        site=site,
        bcode=bcode,
        snap_id=snap_id,
        insight_generated_at=item.get("insight_generated_at"),
        score=int(item.get("branch_qty_12m") or 0),
    )
    return "model"


def note_insight_for_review(
    *,
    site: str,
    bcode: str,
    snap_id: str,
    insight_generated_at: str,
    facts: dict[str, Any],
) -> str:
    """After a branch insight is saved, screen that product's QTYMIN."""
    item = slice_from_facts(
        bcode=bcode,
        insight_generated_at=insight_generated_at,
        facts=facts,
        branch=site,
    )
    existing = qtymin_review_marks(site).get(bcode)
    return apply_screen(site=site, snap_id=snap_id, item=item, existing=existing)


def sync_qtymin_queue(*, site: str, snap_id: str) -> dict[str, int]:
    """Fill the model queue from stored insights. Obvious no-sales rows skip the Spark."""
    from src.kcw.product_insight_store import _pg_connect

    marks = qtymin_review_marks(site)
    counts = {"skip": 0, "rule": 0, "model": 0}
    pending_commit = 0
    with _pg_connect() as conn:
        for raw in iter_insight_qtymin_slices(site):
            item = slice_from_insight_row(raw)
            if not item["bcode"]:
                continue
            action = apply_screen(
                site=site,
                snap_id=snap_id,
                item=item,
                existing=marks.get(item["bcode"]),
                conn=conn,
            )
            counts[action] = counts.get(action, 0) + 1
            if action == "rule":
                pending_commit += 1
                if pending_commit >= 200:
                    conn.commit()
                    pending_commit = 0
        if pending_commit:
            conn.commit()
    return counts


def run_qtymin_model(
    *,
    site: str,
    bcode: str,
    snap_id: str,
    insight_generated_at: str | None,
    model: str,
) -> dict[str, Any]:
    """One Spark recommendation for a queued 0 or -1."""
    from src.kcw.product_insight_generate import _parse_insight_json, _spark_chat

    raw = insight_qtymin_slice(site, bcode)
    match = slice_from_insight_row(raw) if raw else None
    if match is None:
        raise RuntimeError(f"no {site} insight for {bcode}")
    kind = screen_qtymin(match.get("current_qtymin"), match.get("branch_qty_12m"))
    if kind == "rule":
        review = rule_review(
            current_qtymin=match.get("current_qtymin"),
            qtyoh=match.get("qtyoh"),
            insight_generated_at=match.get("insight_generated_at"),
        )
        _save_review(site, bcode, review)
        _close_queue(site, bcode, snap_id)
        return {"bcode": bcode, "elapsed_s": 0, "summary": review["why"]}
    if kind == "skip":
        _close_queue(site, bcode, snap_id)
        return {"bcode": bcode, "elapsed_s": 0, "summary": "skip"}

    prompt = load_qtymin_prompt()
    payload = {
        "site": site,
        "bcode": bcode,
        "current_qtymin": match["current_qtymin"],
        "current_meaning": current_meaning(match["current_qtymin"]),
        "qtyoh": match["qtyoh"],
        "branch_qty_12m": match["branch_qty_12m"],
        "dead_stock": match["dead_stock"],
        "order_ok": match["order_ok"],
        "margin_flag": match["margin_flag"],
        "rec_qtymin": match["rec_qtymin"],
        "insight_min_was_forced_zero": match["insight_min_was_forced_zero"],
        "insight_generated_at": match["insight_generated_at"] or insight_generated_at,
    }
    user = (
        "Branch slice from the stored insight:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "Recommend set_do_not_order or set_positive. JSON only."
    )
    content, _usage, elapsed = _spark_chat(
        model=model,
        system=str(prompt.get("system") or ""),
        user=user,
        temperature=float(prompt.get("temperature", 0.2)),
    )
    parsed = _parse_insight_json(content)
    if not isinstance(parsed, dict):
        raise ValueError("qtymin model output was not an object")
    review = normalize_model_decision(parsed, current_qtymin=match["current_qtymin"])
    review["insight_generated_at"] = payload["insight_generated_at"]
    _save_review(site, bcode, review)
    _close_queue(site, bcode, snap_id)
    return {
        "bcode": bcode,
        "elapsed_s": round(elapsed, 2),
        "summary": review["why"],
    }
