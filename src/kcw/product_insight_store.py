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
    return conn


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


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
        CREATE INDEX IF NOT EXISTS insight_queue_status_idx
          ON insight_queue (site, snap_id, window, status, movement_score DESC);
        """
    )
    # Migrate older DBs created before lease columns existed.
    _ensure_column(conn, "insight_queue", "leased_at", "TEXT")
    _ensure_column(conn, "insight_queue", "retry_count", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "insight_queue", "last_error", "TEXT")
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
        """
    )
    conn.commit()
    if own:
        conn.close()


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
) -> None:
    conn = connect_insights()
    try:
        init_insights_schema(conn)
        extra = extras or {}
        extra_cols = [c for c, _ in INSIGHT_QUERY_COLUMNS if c in extra]
        col_sql = (
            "site, bcode, generated_at, facts_as_of, prompt_version, "
            "model_id, facts_hash, summary, insight_json"
        )
        placeholders = "?, ?, ?, ?, ?, ?, ?, ?, ?"
        values: list[Any] = [
            site.lower(),
            bcode.strip(),
            generated_at,
            facts_as_of,
            prompt_version,
            model_id,
            facts_hash,
            summary,
            insight_json,
        ]
        update_sql = (
            "generated_at=excluded.generated_at, "
            "facts_as_of=excluded.facts_as_of, "
            "prompt_version=excluded.prompt_version, "
            "model_id=excluded.model_id, "
            "facts_hash=excluded.facts_hash, "
            "summary=excluded.summary, "
            "insight_json=excluded.insight_json"
        )
        for col in extra_cols:
            col_sql += f", {col}"
            placeholders += ", ?"
            values.append(extra.get(col))
            update_sql += f", {col}=excluded.{col}"
        conn.execute(
            f"""
            INSERT INTO product_insights ({col_sql})
            VALUES ({placeholders})
            ON CONFLICT(site, bcode) DO UPDATE SET {update_sql}
            """,
            values,
        )
        conn.commit()
    finally:
        conn.close()


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
