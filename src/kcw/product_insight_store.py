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
    snaps = sorted(
        (p.parent.name for p in snaps_dir().glob("*/snapshot.sqlite")),
        reverse=True,
    )
    if not snaps:
        raise FileNotFoundError(f"no snapshots under {snaps_dir()}")
    return snaps[0]


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
          PRIMARY KEY (site, bcode, snap_id, window)
        );
        CREATE INDEX IF NOT EXISTS insight_queue_status_idx
          ON insight_queue (site, snap_id, window, status, movement_score DESC);
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
) -> None:
    conn = connect_insights()
    try:
        init_insights_schema(conn)
        conn.execute(
            """
            INSERT INTO product_insights (
              site, bcode, generated_at, facts_as_of, prompt_version,
              model_id, facts_hash, summary, insight_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(site, bcode) DO UPDATE SET
              generated_at=excluded.generated_at,
              facts_as_of=excluded.facts_as_of,
              prompt_version=excluded.prompt_version,
              model_id=excluded.model_id,
              facts_hash=excluded.facts_hash,
              summary=excluded.summary,
              insight_json=excluded.insight_json
            """,
            (
                site.lower(),
                bcode.strip(),
                generated_at,
                facts_as_of,
                prompt_version,
                model_id,
                facts_hash,
                summary,
                insight_json,
            ),
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
              CASE status WHEN 'pending' THEN 0 WHEN 'done' THEN 1 ELSE 2 END,
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
