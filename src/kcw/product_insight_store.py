"""Paths and SQLite store for product insights (shared HQ analytic + Explorer)."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _expand(p: str | Path) -> Path:
    return Path(p).expanduser()


def insights_data_dir() -> Path:
    env = os.getenv("PRODUCT_INSIGHTS_DATA_DIR", "").strip()
    if env:
        d = _expand(env)
    else:
        d = _expand("~/kcw-data/product_insights")
    d.mkdir(parents=True, exist_ok=True)
    return d


def insights_db_path() -> Path:
    env = os.getenv("PRODUCT_INSIGHTS_DB", "").strip()
    if env:
        path = _expand(env)
    else:
        path = insights_data_dir() / "insights.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def snaps_dir() -> Path:
    env = os.getenv("PRODUCT_INSIGHTS_SNAP_DIR", "").strip()
    if env:
        d = _expand(env)
    else:
        d = insights_data_dir() / "snaps"
    d.mkdir(parents=True, exist_ok=True)
    return d


def snap_path(snap_id: str) -> Path:
    return snaps_dir() / snap_id / "snapshot.sqlite"


def resolve_snap_id(snap: str) -> str:
    snap = (snap or "latest").strip()
    if snap != "latest":
        if not snap_path(snap).is_file():
            raise FileNotFoundError(f"snapshot not found: {snap_path(snap)}")
        return snap
    # Prefer newest by mtime (name sort wrongly ranks "smoke5y" above timestamps).
    paths = list(snaps_dir().glob("*/snapshot.sqlite"))
    if not paths:
        raise FileNotFoundError(f"no snapshots under {snaps_dir()}")
    newest = max(paths, key=lambda p: p.stat().st_mtime)
    return newest.parent.name


def connect_insights(*, readonly: bool = False) -> sqlite3.Connection:
    path = insights_db_path()
    if readonly:
        uri = f"file:{path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise


INSIGHT_QUERY_COLUMNS: tuple[tuple[str, str], ...] = (
    ("typical_monthly_qty", "REAL"),
    ("suggested_cover_weeks", "REAL"),
    ("safe_holding_qty", "REAL"),
    ("safe_holding_reason", "TEXT"),
    ("order_ok", "TEXT"),
    ("order_ok_reason", "TEXT"),
    ("dead_stock", "TEXT"),
    ("dead_stock_reason", "TEXT"),
    ("suggested_order_qty", "REAL"),
    ("suggested_order_qty_large", "REAL"),
    ("order_unit", "TEXT"),
    ("order_unit_large", "TEXT"),
    ("last_supplier", "TEXT"),
    ("last_supplier_acct", "TEXT"),
    ("last_buy_price", "REAL"),
    ("last_buy_date", "TEXT"),
    ("rec_qtymin", "REAL"),
    ("rec_qtymin_reason", "TEXT"),
    ("check_stock", "TEXT"),
    ("stock_anomaly", "TEXT"),
    ("qtyoh_hq", "REAL"),
    ("qtyoh_syp", "REAL"),
    ("qtymin_hq", "REAL"),
    ("qtymin_syp", "REAL"),
    ("rec_transfer_qty_to_syp", "REAL"),
    ("rec_transfer_reason", "TEXT"),
    ("sales_qty_30d", "REAL"),
    ("sales_qty_90d", "REAL"),
    ("sales_qty_12m", "REAL"),
    ("trend_30d", "TEXT"),
    ("trend_90d", "TEXT"),
    ("trend_12m", "TEXT"),
    ("trend_label", "TEXT"),
    ("margin_pct_list", "REAL"),
    ("margin_pct_12m", "REAL"),
    ("margin_pct_prior_12m", "REAL"),
    ("margin_delta_pp", "REAL"),
    ("margin_flag", "TEXT"),
    ("cost_change_pct_12m", "REAL"),
    ("price_change_pct_12m", "REAL"),
)


def init_insights_schema(conn: sqlite3.Connection | None = None) -> None:
    own = conn is None
    if own:
        conn = connect_insights()
    assert conn is not None
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS product_insights (
          site TEXT NOT NULL,
          bcode TEXT NOT NULL,
          generated_at TEXT NOT NULL,
          facts_as_of TEXT NOT NULL,
          prompt_version TEXT NOT NULL,
          model_id TEXT,
          facts_hash TEXT,
          summary TEXT,
          insight_json TEXT NOT NULL,
          PRIMARY KEY (site, bcode)
        );
        CREATE TABLE IF NOT EXISTS insight_queue (
          site TEXT NOT NULL,
          bcode TEXT NOT NULL,
          snap_id TEXT NOT NULL,
          facts_as_of TEXT NOT NULL,
          window TEXT NOT NULL,
          movement_score INTEGER NOT NULL DEFAULT 0,
          status TEXT NOT NULL DEFAULT 'pending',
          updated_at TEXT NOT NULL,
          leased_at TEXT,
          retry_count INTEGER NOT NULL DEFAULT 0,
          last_error TEXT,
          PRIMARY KEY (site, bcode, snap_id, window)
        );
        CREATE TABLE IF NOT EXISTS insight_worker_state (
          site TEXT PRIMARY KEY,
          mover_window TEXT NOT NULL,
          last_snap_id TEXT,
          last_snap_at TEXT,
          updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS idle_pref (
          id INTEGER PRIMARY KEY CHECK (id = 1),
          kind TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS idle_sync (
          id INTEGER PRIMARY KEY CHECK (id = 1),
          synced_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS insight_queue_status_idx
          ON insight_queue (site, snap_id, window, status, movement_score DESC);
        """
    )
    # Migrate older DBs created before lease columns existed.
    _ensure_column(conn, "insight_queue", "leased_at", "TEXT")
    _ensure_column(conn, "insight_queue", "retry_count", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "insight_queue", "last_error", "TEXT")
    _ensure_column(conn, "insight_queue", "due_tier", "INTEGER NOT NULL DEFAULT 1")
    for col, decl in INSIGHT_QUERY_COLUMNS:
        _ensure_column(conn, "product_insights", col, decl)
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS pi_trend12_idx ON product_insights (trend_12m);
        CREATE INDEX IF NOT EXISTS pi_trend90_idx ON product_insights (trend_90d);
        CREATE INDEX IF NOT EXISTS pi_margin_flag_idx ON product_insights (margin_flag);
        CREATE INDEX IF NOT EXISTS pi_order_ok_idx ON product_insights (order_ok);
        CREATE INDEX IF NOT EXISTS pi_dead_idx ON product_insights (dead_stock);
        CREATE INDEX IF NOT EXISTS pi_stock_anom_idx ON product_insights (stock_anomaly);
        CREATE INDEX IF NOT EXISTS pi_last_sup_acct_idx
          ON product_insights (site, last_supplier_acct);
        """
    )
    conn.commit()
    if own:
        conn.close()


INSIGHT_TABLE = "product_insight.product_insights"
QTYMIN_REVIEW_TABLE = "product_insight.qtymin_reviews"


def _pg_connect():
    import psycopg

    from src.kcw.tar import supabase_db_url

    return psycopg.connect(supabase_db_url(), connect_timeout=15)


def _ensure_remote_data_through(conn) -> bool:
    """Bill date the insight was generated against. Missing on older rows."""
    try:
        conn.execute(
            f"ALTER TABLE {INSIGHT_TABLE} ADD COLUMN IF NOT EXISTS data_through text"
        )
        return True
    except Exception as exc:
        print(f"[insight] data_through column skipped: {exc}", flush=True)
        return False


def remote_generated_at(site: str) -> dict[str, str]:
    """bcode -> generated_at for insights already stored in Supabase."""
    with _pg_connect() as conn:
        rows = conn.execute(
            f"SELECT bcode, generated_at FROM {INSIGHT_TABLE} WHERE site = %s",
            (site.lower(),),
        ).fetchall()
    return {str(bcode): str(generated_at) for bcode, generated_at in rows if bcode and generated_at}


def remote_insight_marks(site: str) -> dict[str, dict[str, str | None]]:
    """bcode -> generated_at, facts_as_of, data_through from Supabase."""
    with _pg_connect() as conn:
        has_through = _ensure_remote_data_through(conn)
        if has_through:
            conn.commit()
            rows = conn.execute(
                f"""
                SELECT bcode, generated_at, facts_as_of, data_through
                FROM {INSIGHT_TABLE} WHERE site = %s
                """,
                (site.lower(),),
            ).fetchall()
        else:
            conn.rollback()
            rows = conn.execute(
                f"""
                SELECT bcode, generated_at, facts_as_of
                FROM {INSIGHT_TABLE} WHERE site = %s
                """,
                (site.lower(),),
            ).fetchall()
    out: dict[str, dict[str, str | None]] = {}
    for row in rows:
        bcode = str(row[0] or "").strip()
        if not bcode:
            continue
        out[bcode] = {
            "generated_at": str(row[1]) if row[1] else None,
            "facts_as_of": str(row[2]) if row[2] else None,
            "data_through": (str(row[3])[:10] if has_through and row[3] else None),
        }
    return out


def upsert_insight(
    *,
    site: str,
    bcode: str,
    generated_at: str,
    facts_as_of: str,
    prompt_version: str,
    model_id: str | None,
    facts_hash: str | None,
    summary: str | None,
    insight_json: str,
    extras: dict[str, Any] | None = None,
    data_through: str | None = None,
) -> None:
    """Insert/update one insight in Supabase, including SKUs that already exist locally."""
    site_l = site.lower()
    code = bcode.strip()

    import json

    from psycopg.types.json import Json

    extra = extras or {}
    payload: dict[str, Any] = {
        "site": site_l,
        "bcode": code,
        "generated_at": generated_at,
        "facts_as_of": facts_as_of,
        "prompt_version": prompt_version,
        "model_id": model_id,
        "facts_hash": facts_hash,
        "summary": summary,
        "insight_json": Json(json.loads(insight_json) if isinstance(insight_json, str) else insight_json),
    }
    for col, _decl in INSIGHT_QUERY_COLUMNS:
        if col in extra:
            payload[col] = extra.get(col)

    through = (data_through or "")[:10]
    with _pg_connect() as conn:
        if through and _ensure_remote_data_through(conn):
            payload["data_through"] = through
        elif through:
            conn.rollback()
        cols = list(payload.keys())
        col_sql = ", ".join(cols)
        placeholders = ", ".join(["%s"] * len(cols))
        update_sql = ", ".join(
            f"{col}=excluded.{col}" for col in cols if col not in ("site", "bcode")
        )
        values = [payload[c] for c in cols]
        conn.execute(
            f"""
            INSERT INTO {INSIGHT_TABLE} ({col_sql})
            VALUES ({placeholders})
            ON CONFLICT (site, bcode) DO UPDATE SET {update_sql}
            """,
            values,
        )
        conn.commit()


def _ensure_qtymin_reviews(conn) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {QTYMIN_REVIEW_TABLE} (
          site text NOT NULL,
          bcode text NOT NULL,
          current_qtymin double precision,
          current_meaning text NOT NULL,
          decision text NOT NULL,
          suggested_qtymin double precision,
          why text,
          watch jsonb,
          insight_generated_at text,
          reviewed_at text NOT NULL,
          source text NOT NULL,
          PRIMARY KEY (site, bcode)
        )
        """
    )


def upsert_qtymin_review(
    *,
    site: str,
    bcode: str,
    current_qtymin: float | None,
    current_meaning: str,
    decision: str,
    suggested_qtymin: float,
    why: str | None,
    watch: list[str] | None,
    insight_generated_at: str | None,
    reviewed_at: str,
    source: str,
    conn: Any = None,
) -> None:
    """Store one QTYMIN recommendation. Does not write PARTS9."""
    from psycopg.types.json import Json

    own = conn is None
    if own:
        conn = _pg_connect()
    try:
        _ensure_qtymin_reviews(conn)
        conn.execute(
            f"""
            INSERT INTO {QTYMIN_REVIEW_TABLE} (
              site, bcode, current_qtymin, current_meaning, decision,
              suggested_qtymin, why, watch, insight_generated_at, reviewed_at, source
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (site, bcode) DO UPDATE SET
              current_qtymin=excluded.current_qtymin,
              current_meaning=excluded.current_meaning,
              decision=excluded.decision,
              suggested_qtymin=excluded.suggested_qtymin,
              why=excluded.why,
              watch=excluded.watch,
              insight_generated_at=excluded.insight_generated_at,
              reviewed_at=excluded.reviewed_at,
              source=excluded.source
            """,
            (
                site.lower(),
                bcode.strip(),
                current_qtymin,
                current_meaning,
                decision,
                suggested_qtymin,
                (why or "")[:500] or None,
                Json(watch or []),
                insight_generated_at,
                reviewed_at,
                source,
            ),
        )
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def qtymin_review_marks(site: str) -> dict[str, dict[str, str | None]]:
    """bcode -> decision and the insight timestamp the review sat on."""
    with _pg_connect() as conn:
        _ensure_qtymin_reviews(conn)
        conn.commit()
        rows = conn.execute(
            f"""
            SELECT bcode, decision, insight_generated_at
            FROM {QTYMIN_REVIEW_TABLE}
            WHERE site = %s
            """,
            (site.lower(),),
        ).fetchall()
    out: dict[str, dict[str, str | None]] = {}
    for bcode, decision, insight_at in rows:
        code = str(bcode or "").strip()
        if not code:
            continue
        out[code] = {
            "decision": str(decision) if decision else None,
            "insight_generated_at": str(insight_at) if insight_at else None,
        }
    return out


def insight_qtymin_slice(site: str, bcode: str) -> dict[str, Any] | None:
    """One branch slice for a stored insight, or None."""
    site_l = site.lower()
    channel = "syp" if site_l == "syp" else "hq"
    qtymin_col = "qtymin_syp" if site_l == "syp" else "qtymin_hq"
    qtyoh_col = "qtyoh_syp" if site_l == "syp" else "qtyoh_hq"
    with _pg_connect() as conn:
        row = conn.execute(
            f"""
            SELECT bcode, generated_at, dead_stock, order_ok, margin_flag, rec_qtymin,
                   {qtymin_col}, {qtyoh_col},
                   insight_json #>> ARRAY['dashboard','channels_12m',%s]
            FROM {INSIGHT_TABLE}
            WHERE site = %s AND bcode = %s
            """,
            (channel, site_l, bcode.strip()),
        ).fetchone()
    if row is None:
        return None
    cols = (
        "bcode",
        "generated_at",
        "dead_stock",
        "order_ok",
        "margin_flag",
        "rec_qtymin",
        "current_qtymin",
        "qtyoh",
        "branch_qty_12m",
    )
    return dict(zip(cols, row))


def iter_insight_qtymin_slices(site: str):
    """Yield branch slices from stored insights. No PARTS9 read."""
    site_l = site.lower()
    channel = "syp" if site_l == "syp" else "hq"
    qtymin_col = "qtymin_syp" if site_l == "syp" else "qtymin_hq"
    qtyoh_col = "qtyoh_syp" if site_l == "syp" else "qtyoh_hq"
    with _pg_connect() as conn:
        cur = conn.execute(
            f"""
            SELECT bcode, generated_at, dead_stock, order_ok, margin_flag, rec_qtymin,
                   {qtymin_col}, {qtyoh_col},
                   insight_json #>> ARRAY['dashboard','channels_12m',%s]
            FROM {INSIGHT_TABLE}
            WHERE site = %s
            """,
            (channel, site_l),
        )
        cols = (
            "bcode",
            "generated_at",
            "dead_stock",
            "order_ok",
            "margin_flag",
            "rec_qtymin",
            "current_qtymin",
            "qtyoh",
            "branch_qty_12m",
        )
        while True:
            batch = cur.fetchmany(500)
            if not batch:
                break
            for raw in batch:
                yield dict(zip(cols, raw))


def get_insight(site: str, bcode: str) -> dict[str, Any] | None:
    path = insights_db_path()
    if not path.is_file():
        return None
    try:
        conn = connect_insights(readonly=True)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute(
            """
            SELECT site, bcode, generated_at, facts_as_of, prompt_version,
                   model_id, summary, insight_json
            FROM product_insights
            WHERE site = ? AND bcode = ?
            """,
            (site.lower(), bcode.strip()),
        ).fetchone()
        return dict(row) if row else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def queue_status(site: str, bcode: str) -> dict[str, Any] | None:
    """Latest queue row for bcode (any snap), preferring pending then done."""
    path = insights_db_path()
    if not path.is_file():
        return None
    try:
        conn = connect_insights(readonly=True)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute(
            """
            SELECT site, bcode, snap_id, facts_as_of, window, movement_score, status, updated_at
            FROM insight_queue
            WHERE site = ? AND bcode = ?
            ORDER BY
              CASE status
                WHEN 'running' THEN 0
                WHEN 'pending' THEN 1
                WHEN 'done' THEN 2
                ELSE 3
              END,
              updated_at DESC
            LIMIT 1
            """,
            (site.lower(), bcode.strip()),
        ).fetchone()
        return dict(row) if row else None
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_iso_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None
