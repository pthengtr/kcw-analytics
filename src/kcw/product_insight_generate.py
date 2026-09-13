"""Generate product insights from a local snapshot via Spark vLLM."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from src.kcw.product_insight_channels import CHANNEL_LEGEND, channel_of
from src.kcw.product_insight_store import (
    connect_insights,
    init_insights_schema,
    resolve_snap_id,
    snap_path,
    upsert_insight,
    utc_now_iso,
)

MAX_LINES_PER_KIND = 800  # soft cap for pathological SKUs
SPARK_BASE = os.getenv("SPARK_VLLM_BASE_URL", "http://spark-3583:8000/v1").rstrip("/")


def parse_window(window: str, *, as_of: date | None = None) -> date:
    """Return cutoff date = as_of - window."""
    w = (window or "5y").strip().lower()
    base = as_of or date.today()
    m = re.fullmatch(r"(\d+)\s*([dwmy])", w)
    if not m:
        raise ValueError(f"invalid window {window!r}; use e.g. 5y, 14d, 2w")
    n, unit = int(m.group(1)), m.group(2)
    if unit == "d":
        return base - timedelta(days=n)
    if unit == "w":
        return base - timedelta(weeks=n)
    if unit == "m":
        return base - timedelta(days=30 * n)
    if unit == "y":
        return base - timedelta(days=365 * n)
    raise ValueError(f"invalid window unit in {window!r}")


def _prompt_path() -> Path:
    return Path(__file__).resolve().parents[2] / "prompts" / "product_insight_v1.yaml"


def load_prompt() -> dict[str, Any]:
    path = _prompt_path()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict) or not data.get("system"):
        raise RuntimeError(f"invalid prompt file: {path}")
    return data


def _snap_meta(conn: sqlite3.Connection) -> dict[str, str]:
    return {r[0]: r[1] for r in conn.execute("SELECT key, value FROM meta")}


def _open_snap(snap_id: str) -> sqlite3.Connection:
    path = snap_path(snap_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def build_queue(
    *,
    site: str,
    snap_id: str,
    window: str,
    limit: int | None,
    resume: bool,
) -> list[dict[str, Any]]:
    snap = _open_snap(snap_id)
    try:
        meta = _snap_meta(snap)
        facts_as_of = meta.get("facts_as_of") or utc_now_iso()
        # parse facts_as_of date for window
        as_of_d = date.fromisoformat(facts_as_of[:10])
        cutoff = parse_window(window, as_of=as_of_d).isoformat()

        rows = snap.execute(
            """
            SELECT bcode, SUM(n) AS score FROM (
              SELECT bcode, COUNT(*) AS n FROM sidet
              WHERE billdate >= ? GROUP BY bcode
              UNION ALL
              SELECT bcode, COUNT(*) AS n FROM pidet
              WHERE billdate >= ? GROUP BY bcode
            ) GROUP BY bcode
            ORDER BY score DESC
            """,
            (cutoff, cutoff),
        ).fetchall()
        items = [
            {
                "site": site.lower(),
                "bcode": r["bcode"],
                "movement_score": int(r["score"] or 0),
                "facts_as_of": facts_as_of,
                "snap_id": snap_id,
                "window": window,
            }
            for r in rows
            if r["bcode"]
        ]
        if limit is not None and limit > 0:
            items = items[: int(limit)]
    finally:
        snap.close()

    insight = connect_insights()
    try:
        init_insights_schema(insight)
        now = utc_now_iso()
        if not resume:
            insight.execute(
                "DELETE FROM insight_queue WHERE site=? AND snap_id=? AND window=?",
                (site.lower(), snap_id, window),
            )
        for it in items:
            insight.execute(
                """
                INSERT INTO insight_queue (
                  site, bcode, snap_id, facts_as_of, window, movement_score, status, updated_at
                ) VALUES (?,?,?,?,?,?, 'pending', ?)
                ON CONFLICT(site, bcode, snap_id, window) DO UPDATE SET
                  movement_score=excluded.movement_score,
                  facts_as_of=excluded.facts_as_of,
                  updated_at=excluded.updated_at
                """,
                (
                    it["site"],
                    it["bcode"],
                    snap_id,
                    it["facts_as_of"],
                    window,
                    it["movement_score"],
                    now,
                ),
            )
        insight.commit()

        if resume:
            pending = insight.execute(
                """
                SELECT site, bcode, snap_id, facts_as_of, window, movement_score, status
                FROM insight_queue
                WHERE site=? AND snap_id=? AND window=? AND status='pending'
                ORDER BY movement_score DESC
                """,
                (site.lower(), snap_id, window),
            ).fetchall()
            return [dict(r) for r in pending]
        return items
    finally:
        insight.close()


def _by_year(lines: list[dict[str, Any]]) -> dict[str, float]:
    y: dict[str, float] = defaultdict(float)
    for r in lines:
        d = str(r.get("billdate") or "")[:4]
        try:
            y[d] += float(r.get("qty") or 0)
        except Exception:
            pass
    return {k: round(v, 4) for k, v in sorted(y.items())}


def build_fact_pack(snap: sqlite3.Connection, *, site: str, bcode: str, facts_as_of: str) -> dict[str, Any]:
    ic = snap.execute("SELECT * FROM icmas WHERE bcode=?", (bcode,)).fetchone()
    master = {}
    if ic:
        master = {
            "BCODE": ic["bcode"],
            "DESCR": ic["descr"],
            "BRAND": ic["brand"],
            "MODEL": ic["model"],
            "PCODE": ic["pcode"],
            "MCODE": ic["mcode"],
            "ACODE": ic["acode"],
            "SIZE1": ic["size1"],
            "UI1": ic["ui1"],
            "UI2": ic["ui2"],
            "MTP2": ic["mtp2"],
            "STATUS": ic["status"],
            "LOCATION1": ic["location1"],
            "COSTAVG": ic["costavg"],
            "COSTLAST": ic["costlast"],
            "PRICE1": ic["price1"],
        }
        master = {k: v for k, v in master.items() if v is not None and v != ""}

    si_rows = snap.execute(
        """
        SELECT billno, billdate, qty, ui, price, amount, jourmode
        FROM sidet WHERE bcode=? ORDER BY billdate
        """,
        (bcode,),
    ).fetchall()
    pi_rows = snap.execute(
        """
        SELECT billno, billdate, qty, ui, price, amount, billtype
        FROM pidet WHERE bcode=? ORDER BY billdate
        """,
        (bcode,),
    ).fetchall()

    mix: dict[str, float] = defaultdict(float)
    si_c: list[dict[str, Any]] = []
    for r in si_rows:
        ch = channel_of(r["billno"], r["jourmode"])
        try:
            mix[ch] += float(r["qty"] or 0)
        except Exception:
            pass
        si_c.append(
            {
                "BILLNO": r["billno"],
                "BILLDATE": r["billdate"],
                "QTY": r["qty"],
                "UI": r["ui"],
                "PRICE": r["price"],
                "AMOUNT": r["amount"],
                "JOURMODE": r["jourmode"],
                "channel": ch,
            }
        )
    if len(si_c) > MAX_LINES_PER_KIND:
        # keep newest
        si_c = si_c[-MAX_LINES_PER_KIND:]

    pi_c = [
        {
            "BILLNO": r["billno"],
            "BILLDATE": r["billdate"],
            "QTY": r["qty"],
            "UI": r["ui"],
            "PRICE": r["price"],
            "AMOUNT": r["amount"],
            "BILLTYPE": r["billtype"],
        }
        for r in pi_rows
    ]
    if len(pi_c) > MAX_LINES_PER_KIND:
        pi_c = pi_c[-MAX_LINES_PER_KIND:]

    return {
        "site": site,
        "bcode": bcode,
        "facts_as_of": facts_as_of,
        "window_years": 5,
        "master": master,
        "channel_qty_5y": {k: round(v, 4) for k, v in sorted(mix.items())},
        "sales_5y": {
            "line_count": len(si_rows),
            "qty_by_year": _by_year(si_c),
            "lines": si_c,
        },
        "purchase_5y": {
            "line_count": len(pi_rows),
            "qty_by_year": _by_year(pi_c),
            "lines": pi_c,
        },
    }


def _facts_hash(facts: dict[str, Any]) -> str:
    blob = json.dumps(facts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def resolve_model(model: str | None = None) -> str:
    """Pick Spark model id: explicit arg → SPARK_MODEL env → listed id → first loaded."""
    preferred = (model or os.getenv("SPARK_MODEL") or os.getenv("PRODUCT_INSIGHT_MODEL") or "").strip()
    with urllib.request.urlopen(f"{SPARK_BASE}/models", timeout=30) as r:
        data = json.load(r)
    ids = [m["id"] for m in (data.get("data") or []) if m.get("id")]
    if not ids:
        raise RuntimeError(f"no models at {SPARK_BASE}/models")
    if preferred:
        if preferred in ids:
            return preferred
        raise RuntimeError(f"model {preferred!r} not loaded; available={ids}")
    return ids[0]


def _list_model() -> str:
    return resolve_model(None)


def _spark_chat(*, model: str, system: str, user: str, temperature: float) -> tuple[str, dict | None, float]:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": 2048,
        }
    ).encode()
    req = urllib.request.Request(
        f"{SPARK_BASE}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        body = json.load(r)
    elapsed = time.time() - t0
    msg = body["choices"][0]["message"]
    content = _message_text(msg)
    return content, body.get("usage"), elapsed


def _message_text(msg: dict[str, Any] | None) -> str:
    """Nemotron reasoning servers often put the answer in `reasoning` with content=null."""
    if not isinstance(msg, dict):
        return ""
    for key in ("content", "reasoning_content", "reasoning"):
        val = msg.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _parse_insight_json(content: str) -> dict[str, Any]:
    raw = (content or "").strip()
    if "</think>" in raw:
        raw = raw.split("</think>", 1)[1].strip()
    if "```" in raw:
        raw = raw.split("```", 1)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.split("```", 1)[0].strip()
    return json.loads(raw)


def _mark_queue_done(*, site: str, bcode: str, snap_id: str, window: str) -> None:
    conn = connect_insights()
    try:
        conn.execute(
            """
            UPDATE insight_queue
            SET status='done', updated_at=?
            WHERE site=? AND bcode=? AND snap_id=? AND window=?
            """,
            (utc_now_iso(), site.lower(), bcode, snap_id, window),
        )
        conn.commit()
    finally:
        conn.close()


def generate_one(
    *,
    site: str,
    bcode: str,
    snap_id: str,
    window: str,
    facts_as_of: str,
    prompt: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    snap = _open_snap(snap_id)
    try:
        facts = build_fact_pack(snap, site=site, bcode=bcode, facts_as_of=facts_as_of)
    finally:
        snap.close()

    generated_at = utc_now_iso()
    system = (prompt.get("system") or "").strip()
    if CHANNEL_LEGEND not in system:
        system = system + "\n\n" + CHANNEL_LEGEND
    user = (
        f"Site: {site}\nBCODE: {bcode}\n"
        f"facts_as_of: {facts_as_of}\ngenerated_at: {generated_at}\n"
        f"Facts:\n{json.dumps(facts, ensure_ascii=False, indent=2)}\n\n"
        "Produce analytical insight JSON only."
    )
    temp = float(prompt.get("temperature", 0.2))
    content, usage, elapsed = _spark_chat(model=model, system=system, user=user, temperature=temp)
    try:
        insight = _parse_insight_json(content)
    except Exception as exc:
        insight = {
            "summary": "parse_error",
            "sales_trend": "",
            "trend_label": "unknown",
            "channel_mix": "",
            "demand_hint": "",
            "typical_monthly_qty": None,
            "suggested_cover_weeks": None,
            "anomalies": [f"json_parse_error: {exc}"],
            "dead_stock": "maybe",
            "dead_stock_reason": "model output was not valid JSON",
            "raw": (content or "")[:2000],
        }

    summary = insight.get("summary") if isinstance(insight, dict) else None
    upsert_insight(
        site=site,
        bcode=bcode,
        generated_at=generated_at,
        facts_as_of=facts_as_of,
        prompt_version=str(prompt.get("version") or "v1"),
        model_id=model,
        facts_hash=_facts_hash(facts),
        summary=summary if isinstance(summary, str) else None,
        insight_json=json.dumps(insight, ensure_ascii=False),
    )
    _mark_queue_done(site=site, bcode=bcode, snap_id=snap_id, window=window)
    return {
        "bcode": bcode,
        "elapsed_s": round(elapsed, 2),
        "usage": usage,
        "summary": summary,
    }


def run_generate(
    *,
    site: str = "hq",
    snap: str = "latest",
    window: str = "5y",
    limit: int | None = None,
    concurrency: int = 1,
    resume: bool = True,
    model: str | None = None,
) -> int:
    snap_id = resolve_snap_id(snap)
    site = site.lower()
    concurrency = max(1, min(int(concurrency), 8))
    prompt = load_prompt()
    model_id = resolve_model(model)
    print(f"snap={snap_id} site={site} window={window} limit={limit} concurrency={concurrency} resume={resume}")
    print(f"model={model_id} prompt={prompt.get('version')}")

    queue = build_queue(site=site, snap_id=snap_id, window=window, limit=limit, resume=resume)
    print(f"queue_pending={len(queue)}")
    if not queue:
        print("nothing to do")
        return 0

    # facts_as_of from first item / meta
    facts_as_of = queue[0].get("facts_as_of") or utc_now_iso()
    latencies: list[float] = []
    errors = 0

    def _job(item: dict[str, Any]) -> dict[str, Any]:
        return generate_one(
            site=site,
            bcode=item["bcode"],
            snap_id=snap_id,
            window=window,
            facts_as_of=item.get("facts_as_of") or facts_as_of,
            prompt=prompt,
            model=model_id,
        )

    if concurrency == 1:
        for i, item in enumerate(queue, 1):
            try:
                res = _job(item)
                latencies.append(float(res["elapsed_s"]))
                print(
                    f"[{i}/{len(queue)}] {res['bcode']} {res['elapsed_s']}s :: {(res.get('summary') or '')[:80]}",
                    flush=True,
                )
            except Exception as exc:
                errors += 1
                print(f"[{i}/{len(queue)}] ERROR {item['bcode']}: {exc}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {ex.submit(_job, item): item for item in queue}
            done_n = 0
            for fut in as_completed(futs):
                done_n += 1
                item = futs[fut]
                try:
                    res = fut.result()
                    latencies.append(float(res["elapsed_s"]))
                    print(
                        f"[{done_n}/{len(queue)}] {res['bcode']} {res['elapsed_s']}s :: {(res.get('summary') or '')[:80]}",
                        flush=True,
                    )
                except Exception as exc:
                    errors += 1
                    print(f"[{done_n}/{len(queue)}] ERROR {item['bcode']}: {exc}", flush=True)

    if latencies:
        latencies.sort()
        avg = sum(latencies) / len(latencies)
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        print(f"bench n={len(latencies)} avg={avg:.1f}s p50={p50:.1f}s p95={p95:.1f}s errors={errors}")
        # extrapolate if this was a limit run — print hint using remaining unknown
        if limit and avg > 0:
            # rough: if full 5y ~31k
            print(f"extrapolate_31k_hours_conc{concurrency}={(31000 * avg / concurrency) / 3600:.1f}")
    return 1 if errors and not latencies else 0
