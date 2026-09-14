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
from src.kcw.product_insight_derived import build_derived, flatten_insight_columns
from src.kcw.product_insight_store import (
    connect_insights,
    init_insights_schema,
    resolve_snap_id,
    snap_path,
    upsert_insight,
    utc_now_iso,
)

MAX_LINES_PER_KIND = 40  # prompt line dump; rollups still use full history
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
        d = str(r.get("billdate") or r.get("BILLDATE") or "")[:4]
        try:
            y[d] += float(r.get("qty") if "qty" in r else r.get("QTY") or 0)
        except Exception:
            pass
    return {k: round(v, 4) for k, v in sorted(y.items())}


def _line_qty(r: dict[str, Any]) -> float:
    try:
        return float(r.get("qty") if "qty" in r else r.get("QTY") or 0)
    except Exception:
        return 0.0


def _line_date(r: dict[str, Any]) -> str:
    return str(r.get("billdate") or r.get("BILLDATE") or "")[:10]


def _recent_sales_rollup(
    lines: list[dict[str, Any]],
    *,
    as_of: str,
    customer_channels: set[str] | None = None,
) -> dict[str, Any]:
    """Absolute recent sales for ops (not 5y %-of-total)."""
    customer_channels = customer_channels or {"hq_store", "online", "syp_store"}
    try:
        as_of_d = date.fromisoformat(as_of[:10])
    except ValueError:
        as_of_d = date.today()

    def window_stats(days: int) -> dict[str, Any]:
        start = (as_of_d - timedelta(days=days)).isoformat()
        by_ch: dict[str, float] = defaultdict(float)
        customer = 0.0
        lines_n = 0
        for r in lines:
            d = _line_date(r)
            if not d or d < start or d > as_of_d.isoformat():
                continue
            q = _line_qty(r)
            ch = str(r.get("channel") or "unknown")
            by_ch[ch] += q
            lines_n += 1
            if ch in customer_channels:
                customer += q
        months = max(days / 30.4375, 1 / 30.4375)
        return {
            "days": days,
            "from": start,
            "to": as_of_d.isoformat(),
            "customer_qty": round(customer, 4),
            "typical_monthly_qty": round(customer / months, 4),
            "channel_qty": {k: round(v, 4) for k, v in sorted(by_ch.items())},
            "line_count": lines_n,
        }

    return {
        "note": "Use recent_* for rates/mix/holding. sales_5y is trend context only.",
        "last_30d": window_stats(30),
        "last_90d": window_stats(90),
        "last_12m": window_stats(365),
    }


def _table_cols(snap: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in snap.execute(f"PRAGMA table_info({table})")}


def _has_table(snap: sqlite3.Connection, table: str) -> bool:
    row = snap.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return row is not None


def _fnum(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        n = float(v)
    except (TypeError, ValueError):
        return None
    if n != n:
        return None
    return n


def _stock_block(snap: sqlite3.Connection, bcode: str, master: dict[str, Any]) -> dict[str, Any]:
    hq: dict[str, Any] = {}
    syp: dict[str, Any] = {}
    if _has_table(snap, "icmas_stock"):
        for r in snap.execute(
            "SELECT * FROM icmas_stock WHERE bcode=?",
            (bcode,),
        ):
            src = str(r["src_site"] or "").lower()
            block = {
                "qtyoh2": _fnum(r["qtyoh2"] if "qtyoh2" in r.keys() else None),
                "qtymin": _fnum(r["qtymin"] if "qtymin" in r.keys() else None),
                "qtymax": _fnum(r["qtymax"] if "qtymax" in r.keys() else None),
                "costavg": _fnum(r["costavg"] if "costavg" in r.keys() else None),
                "costlast": _fnum(r["costlast"] if "costlast" in r.keys() else None),
                "price1": _fnum(r["price1"] if "price1" in r.keys() else None),
            }
            if src == "syp":
                syp = block
            else:
                hq = block
    if not hq and master:
        hq = {
            "qtyoh2": _fnum(master.get("QTYOH2")),
            "qtymin": _fnum(master.get("QTYMIN")),
            "qtymax": _fnum(master.get("QTYMAX")),
        }
    return {"hq": hq, "syp": syp}


def _is_vendor_purchase(billno: str | None) -> bool:
    """True for supplier invoices; false for HQ↔SYP transfer receipts on PIDET."""
    b = (billno or "").strip().upper()
    if b.startswith(("TFV", "3TFV", "CNTF", "3CNTF")):
        return False
    if b.startswith(("TF", "3TF")):
        return False
    return True


def _purchase_summary(pi_rows: list[dict[str, Any]]) -> dict[str, Any]:
    vendor_rows = [r for r in pi_rows if _is_vendor_purchase(r.get("BILLNO"))]
    by_sup: dict[str, dict[str, Any]] = defaultdict(lambda: {"qty": 0.0, "amount": 0.0, "lines": 0})
    last = None
    for r in vendor_rows:
        d = str(r.get("BILLDATE") or "")[:10]
        q = _fnum(r.get("QTY")) or 0.0
        amt = _fnum(r.get("AMOUNT"))
        if amt is None:
            p = _fnum(r.get("PRICE"))
            amt = (p or 0.0) * q
        name = (r.get("ACCTNAME") or r.get("ACCTNO") or "").strip() or "(unknown)"
        slot = by_sup[name]
        slot["qty"] += q
        slot["amount"] += amt
        slot["lines"] += 1
        slot["acctno"] = r.get("ACCTNO")
        if last is None or d >= str(last.get("BILLDATE") or ""):
            last = r
    top = sorted(by_sup.items(), key=lambda kv: kv[1]["qty"], reverse=True)[:5]
    return {
        "last_date": (last or {}).get("BILLDATE"),
        "last_price": _fnum((last or {}).get("PRICE")),
        "last_qty": _fnum((last or {}).get("QTY")),
        "last_ui": (last or {}).get("UI"),
        "last_supplier": (last or {}).get("ACCTNAME") or (last or {}).get("ACCTNO"),
        "last_supplier_acct": (last or {}).get("ACCTNO"),
        "last_src_site": (last or {}).get("SRC_SITE"),
        "suppliers": [
            {
                "name": k,
                "acctno": v.get("acctno"),
                "qty": round(v["qty"], 4),
                "amount": round(v["amount"], 2),
                "lines": v["lines"],
            }
            for k, v in top
        ],
    }


def _avg_unit(rows: list[dict[str, Any]], *, start: str, end: str, qty_key: str, price_key: str) -> float | None:
    qsum = 0.0
    psum = 0.0
    for r in rows:
        d = str(r.get("billdate") or r.get("BILLDATE") or "")[:10]
        if not d or d < start or d > end:
            continue
        q = _fnum(r.get(qty_key) if qty_key in r else r.get("QTY")) or 0.0
        p = _fnum(r.get(price_key) if price_key in r else r.get("PRICE"))
        if q <= 0 or p is None:
            continue
        qsum += q
        psum += p * q
    if qsum <= 0:
        return None
    return psum / qsum


def _margin_block(
    si_all: list[dict[str, Any]],
    pi_rows: list[dict[str, Any]],
    *,
    as_of: str,
    customer_channels: set[str],
) -> dict[str, Any]:
    try:
        as_of_d = date.fromisoformat(as_of[:10])
    except ValueError:
        as_of_d = date.today()
    end = as_of_d.isoformat()
    start_12 = (as_of_d - timedelta(days=365)).isoformat()
    start_prior = (as_of_d - timedelta(days=730)).isoformat()
    end_prior = start_12
    cust = [r for r in si_all if str(r.get("channel")) in customer_channels]
    vendor_pi = [r for r in pi_rows if _is_vendor_purchase(r.get("BILLNO") or r.get("billno"))]
    sell_12 = _avg_unit(cust, start=start_12, end=end, qty_key="qty", price_key="price")
    sell_prior = _avg_unit(cust, start=start_prior, end=end_prior, qty_key="qty", price_key="price")
    buy_12 = _avg_unit(vendor_pi, start=start_12, end=end, qty_key="QTY", price_key="PRICE")
    buy_prior = _avg_unit(vendor_pi, start=start_prior, end=end_prior, qty_key="QTY", price_key="PRICE")

    def realized(sell: float | None, buy: float | None) -> float | None:
        if sell is None or buy is None or sell == 0:
            return None
        return round((sell - buy) / sell * 100.0, 2)

    def chg(new: float | None, old: float | None) -> float | None:
        if new is None or old is None or old == 0:
            return None
        return round((new - old) / abs(old) * 100.0, 2)

    return {
        "avg_sell_12m": round(sell_12, 4) if sell_12 is not None else None,
        "avg_sell_prior_12m": round(sell_prior, 4) if sell_prior is not None else None,
        "avg_buy_12m": round(buy_12, 4) if buy_12 is not None else None,
        "avg_buy_prior_12m": round(buy_prior, 4) if buy_prior is not None else None,
        "realized_pct_12m": realized(sell_12, buy_12),
        "realized_pct_prior_12m": realized(sell_prior, buy_prior),
        "sell_price_change_pct_12m": chg(sell_12, sell_prior),
        "buy_cost_change_pct_12m": chg(buy_12, buy_prior),
    }


def _row_src_site(row: sqlite3.Row | dict[str, Any], default: str = "hq") -> str:
    try:
        keys = row.keys() if hasattr(row, "keys") else row
        if "src_site" in keys:
            val = row["src_site"]
            if val:
                return str(val).strip().lower()
    except Exception:
        pass
    return default


def build_fact_pack(snap: sqlite3.Connection, *, site: str, bcode: str, facts_as_of: str) -> dict[str, Any]:
    meta = _snap_meta(snap)
    sources = [s.strip() for s in (meta.get("sites") or meta.get("site") or site).split(",") if s.strip()]
    has_si_src = "src_site" in _table_cols(snap, "sidet")
    has_pi_src = "src_site" in _table_cols(snap, "pidet")

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
        keys = ic.keys()
        for src_key, dest in (
            ("pricem1", "PRICEM1"),
            ("qtyoh2", "QTYOH2"),
            ("qtymin", "QTYMIN"),
            ("qtymax", "QTYMAX"),
        ):
            if src_key in keys:
                master[dest] = ic[src_key]
        src = _row_src_site(ic, default=site)
        if src:
            master["SRC_SITE"] = src
        master = {k: v for k, v in master.items() if v is not None and v != ""}

    if has_si_src:
        si_sql = """
            SELECT billno, billdate, qty, ui, price, amount, jourmode,
                   COALESCE(src_site, 'hq') AS src_site
            FROM sidet WHERE bcode=? ORDER BY billdate, src_site
        """
    else:
        si_sql = """
            SELECT billno, billdate, qty, ui, price, amount, jourmode,
                   'hq' AS src_site
            FROM sidet WHERE bcode=? ORDER BY billdate
        """
    if has_pi_src:
        pi_sql = """
            SELECT billno, billdate, qty, ui, price, amount, billtype,
                   COALESCE(src_site, 'hq') AS src_site
            FROM pidet WHERE bcode=? ORDER BY billdate, src_site
        """
    else:
        pi_sql = """
            SELECT billno, billdate, qty, ui, price, amount, billtype,
                   'hq' AS src_site
            FROM pidet WHERE bcode=? ORDER BY billdate
        """
    si_rows = snap.execute(si_sql, (bcode,)).fetchall()
    pi_rows = snap.execute(pi_sql, (bcode,)).fetchall()
    pimas_map: dict[tuple[str, str], sqlite3.Row] = {}
    if _has_table(snap, "pimas"):
        for r in snap.execute(
            """
            SELECT src_site, billno, acctno, acctname, billdate
            FROM pimas
            WHERE billno IN (SELECT DISTINCT billno FROM pidet WHERE bcode=?)
            """,
            (bcode,),
        ):
            pimas_map[(str(r["src_site"] or "hq"), str(r["billno"] or ""))] = r

    mix: dict[str, float] = defaultdict(float)
    by_src_qty: dict[str, float] = defaultdict(float)
    si_c: list[dict[str, Any]] = []
    for r in si_rows:
        src = _row_src_site(r)
        ch = channel_of(r["billno"], r["jourmode"], src_site=src)
        try:
            q = float(r["qty"] or 0)
            mix[ch] += q
            by_src_qty[src] += q
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
                "SRC_SITE": src,
                "channel": ch,
            }
        )
    if len(si_c) > MAX_LINES_PER_KIND:
        # keep newest
        si_c = si_c[-MAX_LINES_PER_KIND:]

    # Recent rollups from full SI history (before line soft-cap) for operational rates.
    si_all: list[dict[str, Any]] = []
    for r in si_rows:
        src = _row_src_site(r)
        ch = channel_of(r["billno"], r["jourmode"], src_site=src)
        si_all.append(
            {
                "billdate": r["billdate"],
                "qty": r["qty"],
                "price": r["price"],
                "amount": r["amount"],
                "channel": ch,
            }
        )
    recent = _recent_sales_rollup(si_all, as_of=facts_as_of)

    pi_c = []
    for r in pi_rows:
        src = _row_src_site(r)
        hdr = pimas_map.get((src, str(r["billno"] or "")))
        pi_c.append(
            {
                "BILLNO": r["billno"],
                "BILLDATE": r["billdate"],
                "QTY": r["qty"],
                "UI": r["ui"],
                "PRICE": r["price"],
                "AMOUNT": r["amount"],
                "BILLTYPE": r["billtype"],
                "SRC_SITE": src,
                "ACCTNO": (hdr["acctno"] if hdr else None),
                "ACCTNAME": (hdr["acctname"] if hdr else None),
            }
        )
    purchase_summary = _purchase_summary(pi_c)
    if len(pi_c) > MAX_LINES_PER_KIND:
        pi_c = pi_c[-MAX_LINES_PER_KIND:]

    stock = _stock_block(snap, bcode, master)
    margin = _margin_block(
        si_all,
        pi_c if len(pi_rows) <= MAX_LINES_PER_KIND else [
            {
                "BILLDATE": r["billdate"],
                "QTY": r["qty"],
                "PRICE": r["price"],
                "AMOUNT": r["amount"],
            }
            for r in pi_rows
        ],
        as_of=facts_as_of,
        customer_channels={"hq_store", "online", "syp_store"},
    )

    facts = {
        "site": site,
        "sources": sources,
        "bcode": bcode,
        "facts_as_of": facts_as_of,
        "window_years": 5,
        "master": master,
        "stock": stock,
        "recent": recent,
        "channel_qty_5y": {k: round(v, 4) for k, v in sorted(mix.items())},
        "sales_qty_by_src_5y": {k: round(v, 4) for k, v in sorted(by_src_qty.items())},
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
        "purchase_summary": purchase_summary,
        "margin": margin,
    }
    facts["derived"] = build_derived(facts)
    return facts


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
            "max_tokens": 4096,
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
        "Use Facts.derived as the origin for numbers. Override only with a stated reason. "
        "This insight will be reused 14–30 days: standing policy, not a live PO vs snapshot QTYOH. "
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
            "safe_holding_qty": None,
            "anomalies": [f"json_parse_error: {exc}"],
            "dead_stock": "maybe",
            "dead_stock_reason": "model output was not valid JSON",
            "raw": (content or "")[:2000],
        }

    if isinstance(insight, dict):
        insight["derived"] = facts.get("derived") or {}
    extras = flatten_insight_columns(
        insight if isinstance(insight, dict) else {},
        facts.get("derived") or {},
    )

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
        extras=extras,
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
