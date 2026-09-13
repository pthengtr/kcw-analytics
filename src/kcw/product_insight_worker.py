"""Continuous product-insight worker: catch-up then weekly movers (auto-steady).

Priority: never analyzed → age >= fresh_days → optional soft refresh.
Runs forever; idles when nothing is due. Default fresh_days=14, auto-steady 5y→7d.
"""

from __future__ import annotations

import sqlite3
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.kcw.product_insight_generate import (
    generate_one,
    load_prompt,
    parse_window,
    _list_model,
    _open_snap,
    _snap_meta,
)
from src.kcw.product_insight_store import (
    connect_insights,
    init_insights_schema,
    parse_iso_dt,
    resolve_snap_id,
    snap_path,
    utc_now_iso,
)

BACKFILL_WINDOW = "5y"
STEADY_WINDOW = "7d"
DEFAULT_FRESH_DAYS = 14
DEFAULT_SOFT_MIN_DAYS = 7  # soft tier only when enable_soft_refresh
DEFAULT_SNAP_EVERY_DAYS = 7
DEFAULT_IDLE_SECONDS = 120
DEFAULT_LEASE_MINUTES = 90
DEFAULT_MAX_RETRIES = 5
DEFAULT_RESYNC_EVERY = 50  # jobs between eligibility resync


def _age_days(generated_at: str | None, *, now: datetime) -> float | None:
    dt = parse_iso_dt(generated_at)
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (now - dt.astimezone(now.tzinfo)).total_seconds() / 86400.0


def _load_worker_state(conn: sqlite3.Connection, site: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT site, mover_window, last_snap_id, last_snap_at, updated_at FROM insight_worker_state WHERE site=?",
        (site,),
    ).fetchone()
    return dict(row) if row else {}


def _save_worker_state(
    conn: sqlite3.Connection,
    *,
    site: str,
    mover_window: str,
    last_snap_id: str | None,
    last_snap_at: str | None,
) -> None:
    now = utc_now_iso()
    conn.execute(
        """
        INSERT INTO insight_worker_state (site, mover_window, last_snap_id, last_snap_at, updated_at)
        VALUES (?,?,?,?,?)
        ON CONFLICT(site) DO UPDATE SET
          mover_window=excluded.mover_window,
          last_snap_id=excluded.last_snap_id,
          last_snap_at=excluded.last_snap_at,
          updated_at=excluded.updated_at
        """,
        (site, mover_window, last_snap_id, last_snap_at, now),
    )
    conn.commit()


def _mover_rows(snap: sqlite3.Connection, *, window: str, facts_as_of: str) -> list[tuple[str, int]]:
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
        """,
        (cutoff, cutoff),
    ).fetchall()
    return [(r["bcode"], int(r["score"] or 0)) for r in rows if r["bcode"]]


def sync_eligibility(
    *,
    site: str,
    snap_id: str,
    window: str,
    fresh_days: int,
    soft_min_days: int,
    enable_soft_refresh: bool,
) -> dict[str, int]:
    """Upsert movers for (site,snap,window) and set pending/done from insight age."""
    snap = _open_snap(snap_id)
    try:
        meta = _snap_meta(snap)
        facts_as_of = meta.get("facts_as_of") or utc_now_iso()
        movers = _mover_rows(snap, window=window, facts_as_of=facts_as_of)
    finally:
        snap.close()

    now = datetime.now().astimezone()
    now_iso = utc_now_iso()
    conn = connect_insights()
    try:
        init_insights_schema(conn)
        # Keep only current snap+window as the active eligibility set for this site/window.
        # Older snap rows remain for audit but are not claimed.
        for bcode, score in movers:
            conn.execute(
                """
                INSERT INTO insight_queue (
                  site, bcode, snap_id, facts_as_of, window, movement_score,
                  status, updated_at, leased_at, retry_count, last_error
                ) VALUES (?,?,?,?,?,?, 'pending', ?, NULL, 0, NULL)
                ON CONFLICT(site, bcode, snap_id, window) DO UPDATE SET
                  movement_score=excluded.movement_score,
                  facts_as_of=excluded.facts_as_of,
                  updated_at=excluded.updated_at
                """,
                (site, bcode, snap_id, facts_as_of, window, score, now_iso),
            )

        # Recompute due status from product_insights age (skip in-flight running).
        rows = conn.execute(
            """
            SELECT q.bcode, q.status, q.retry_count, pi.generated_at
            FROM insight_queue q
            LEFT JOIN product_insights pi ON pi.site=q.site AND pi.bcode=q.bcode
            WHERE q.site=? AND q.snap_id=? AND q.window=?
            """,
            (site, snap_id, window),
        ).fetchall()

        counts = {"movers": len(movers), "pending": 0, "done_fresh": 0, "never": 0, "stale": 0, "soft": 0, "error": 0}
        for r in rows:
            status = r["status"] or "pending"
            if status == "running":
                continue
            if status == "error" and int(r["retry_count"] or 0) >= DEFAULT_MAX_RETRIES:
                counts["error"] += 1
                continue

            gen = r["generated_at"]
            age = _age_days(gen, now=now)
            due = False
            if gen is None:
                due = True
                counts["never"] += 1
            elif age is not None and age >= fresh_days:
                due = True
                counts["stale"] += 1
            elif enable_soft_refresh and age is not None and age >= soft_min_days:
                due = True
                counts["soft"] += 1
            else:
                counts["done_fresh"] += 1

            new_status = "pending" if due else "done"
            if status != new_status or (due and status == "error"):
                conn.execute(
                    """
                    UPDATE insight_queue
                    SET status=?, updated_at=?, leased_at=NULL,
                        last_error=CASE WHEN ?='pending' THEN NULL ELSE last_error END
                    WHERE site=? AND bcode=? AND snap_id=? AND window=?
                    """,
                    (new_status, now_iso, new_status, site, r["bcode"], snap_id, window),
                )
            if due:
                counts["pending"] += 1

        conn.commit()
        return counts
    finally:
        conn.close()


def reclaim_stale_leases(*, site: str, snap_id: str, window: str, lease_minutes: int) -> int:
    cutoff = (datetime.now().astimezone() - timedelta(minutes=lease_minutes)).isoformat(timespec="seconds")
    conn = connect_insights()
    try:
        init_insights_schema(conn)
        cur = conn.execute(
            """
            UPDATE insight_queue
            SET status='pending', leased_at=NULL, updated_at=?
            WHERE site=? AND snap_id=? AND window=? AND status='running'
              AND (leased_at IS NULL OR leased_at < ?)
            """,
            (utc_now_iso(), site, snap_id, window, cutoff),
        )
        conn.commit()
        return int(cur.rowcount or 0)
    finally:
        conn.close()


def count_never_analyzed(*, site: str, snap_id: str, window: str) -> int:
    conn = connect_insights()
    try:
        n = conn.execute(
            """
            SELECT COUNT(*) FROM insight_queue q
            LEFT JOIN product_insights pi ON pi.site=q.site AND pi.bcode=q.bcode
            WHERE q.site=? AND q.snap_id=? AND q.window=?
              AND pi.bcode IS NULL
            """,
            (site, snap_id, window),
        ).fetchone()[0]
        return int(n)
    finally:
        conn.close()


def claim_next(
    *,
    site: str,
    snap_id: str,
    window: str,
    fresh_days: int,
    soft_min_days: int,
    enable_soft_refresh: bool,
    max_retries: int,
) -> dict[str, Any] | None:
    """Claim highest-priority due BCODE (tier 0 never, 1 stale, 2 soft)."""
    now = datetime.now().astimezone()
    now_iso = utc_now_iso()
    conn = connect_insights()
    try:
        init_insights_schema(conn)
        rows = conn.execute(
            """
            SELECT q.site, q.bcode, q.snap_id, q.facts_as_of, q.window, q.movement_score,
                   q.status, q.retry_count, pi.generated_at
            FROM insight_queue q
            LEFT JOIN product_insights pi ON pi.site=q.site AND pi.bcode=q.bcode
            WHERE q.site=? AND q.snap_id=? AND q.window=?
              AND q.status IN ('pending', 'error')
              AND COALESCE(q.retry_count, 0) < ?
            """,
            (site, snap_id, window, max_retries),
        ).fetchall()

        best: tuple[int, int, str] | None = None  # (tier, -score, bcode)
        best_row: sqlite3.Row | None = None
        for r in rows:
            gen = r["generated_at"]
            age = _age_days(gen, now=now)
            if gen is None:
                tier = 0
            elif age is not None and age >= fresh_days:
                tier = 1
            elif enable_soft_refresh and age is not None and age >= soft_min_days:
                tier = 2
            else:
                continue
            key = (tier, -int(r["movement_score"] or 0), str(r["bcode"]))
            if best is None or key < best:
                best = key
                best_row = r

        if best_row is None:
            return None

        cur = conn.execute(
            """
            UPDATE insight_queue
            SET status='running', leased_at=?, updated_at=?
            WHERE site=? AND bcode=? AND snap_id=? AND window=?
              AND status IN ('pending', 'error')
            """,
            (now_iso, now_iso, site, best_row["bcode"], snap_id, window),
        )
        if cur.rowcount != 1:
            conn.rollback()
            return None
        conn.commit()
        out = dict(best_row)
        out["tier"] = best[0] if best else 0
        return out
    finally:
        conn.close()


def _mark_error(*, site: str, bcode: str, snap_id: str, window: str, error: str, max_retries: int) -> None:
    conn = connect_insights()
    try:
        row = conn.execute(
            """
            SELECT retry_count FROM insight_queue
            WHERE site=? AND bcode=? AND snap_id=? AND window=?
            """,
            (site, bcode, snap_id, window),
        ).fetchone()
        retries = int(row["retry_count"] or 0) + 1 if row else 1
        status = "error" if retries >= max_retries else "pending"
        conn.execute(
            """
            UPDATE insight_queue
            SET status=?, retry_count=?, last_error=?, leased_at=NULL, updated_at=?
            WHERE site=? AND bcode=? AND snap_id=? AND window=?
            """,
            (status, retries, error[:2000], utc_now_iso(), site, bcode, snap_id, window),
        )
        conn.commit()
    finally:
        conn.close()


def _maybe_refresh_snap(
    *,
    site: str,
    snap_every_days: int,
    years: int,
    last_snap_at: str | None,
    force: bool = False,
) -> tuple[str, str]:
    """Return (snap_id, last_snap_at_iso). Creates a new snap when due."""
    now = datetime.now().astimezone()
    need = force
    if not need:
        prev = parse_iso_dt(last_snap_at)
        if prev is None:
            # Use existing latest if present; only force new snap when none exists.
            try:
                sid = resolve_snap_id("latest")
                if snap_path(sid).is_file():
                    return sid, last_snap_at or utc_now_iso()
            except FileNotFoundError:
                need = True
        else:
            if prev.tzinfo is None:
                prev = prev.replace(tzinfo=timezone.utc)
            need = (now - prev.astimezone(now.tzinfo)) >= timedelta(days=snap_every_days)

    if need:
        from src.kcw.product_insight_snapshot import run_snapshot

        print(f"worker: refreshing snapshot site={site} years={years}", flush=True)
        out = run_snapshot(site=site, years=years, snap_id=None)
        if isinstance(out, dict):
            snap_id = out.get("snap_id") or resolve_snap_id("latest")
        else:
            # run_snapshot returns Path to snapshot.sqlite
            snap_id = Path(out).parent.name if out is not None else resolve_snap_id("latest")
        return snap_id, utc_now_iso()

    return resolve_snap_id("latest"), last_snap_at or utc_now_iso()


def run_worker(
    *,
    site: str = "hq",
    mover_window: str = BACKFILL_WINDOW,
    fresh_days: int = DEFAULT_FRESH_DAYS,
    soft_min_days: int = DEFAULT_SOFT_MIN_DAYS,
    enable_soft_refresh: bool = False,
    auto_steady: bool = True,
    snap_every_days: int = DEFAULT_SNAP_EVERY_DAYS,
    idle_seconds: int = DEFAULT_IDLE_SECONDS,
    lease_minutes: int = DEFAULT_LEASE_MINUTES,
    max_retries: int = DEFAULT_MAX_RETRIES,
    concurrency: int = 1,
    years: int = 5,
    max_jobs: int | None = None,
) -> int:
    """Run forever (or until max_jobs). concurrency is forced to 1 on this path."""
    site = site.lower()
    concurrency = 1  # GB10: serial only for this worker
    fresh_days = max(1, int(fresh_days))
    soft_min_days = max(0, int(soft_min_days))
    if soft_min_days >= fresh_days:
        soft_min_days = max(0, fresh_days - 1)
    idle_seconds = max(15, int(idle_seconds))
    snap_every_days = max(1, int(snap_every_days))

    conn = connect_insights()
    try:
        init_insights_schema(conn)
        state = _load_worker_state(conn, site)
    finally:
        conn.close()

    if state.get("mover_window"):
        # Resume prior mode unless caller forced a different starting window via empty state.
        mover_window = str(state["mover_window"])
    last_snap_at = state.get("last_snap_at")
    last_snap_id = state.get("last_snap_id")

    prompt = load_prompt()
    model = _list_model()
    print(
        f"worker start site={site} window={mover_window} fresh_days={fresh_days} "
        f"auto_steady={auto_steady} soft={enable_soft_refresh} snap_every={snap_every_days}d "
        f"model={model} prompt={prompt.get('version')}",
        flush=True,
    )

    jobs_done = 0
    since_sync = DEFAULT_RESYNC_EVERY  # force sync on first loop

    while True:
        if max_jobs is not None and jobs_done >= max_jobs:
            print(f"worker: max_jobs={max_jobs} reached, exiting", flush=True)
            return 0

        try:
            snap_id, last_snap_at = _maybe_refresh_snap(
                site=site,
                snap_every_days=snap_every_days,
                years=years,
                last_snap_at=last_snap_at,
            )
        except Exception as exc:
            print(f"worker: snap refresh failed: {exc}", flush=True)
            time.sleep(idle_seconds)
            continue

        if snap_id != last_snap_id:
            print(f"worker: using snap={snap_id}", flush=True)
            last_snap_id = snap_id
            since_sync = DEFAULT_RESYNC_EVERY

        # Persist state early so restarts resume correctly.
        conn = connect_insights()
        try:
            init_insights_schema(conn)
            _save_worker_state(
                conn,
                site=site,
                mover_window=mover_window,
                last_snap_id=last_snap_id,
                last_snap_at=last_snap_at,
            )
        finally:
            conn.close()

        if since_sync >= DEFAULT_RESYNC_EVERY:
            counts = sync_eligibility(
                site=site,
                snap_id=snap_id,
                window=mover_window,
                fresh_days=fresh_days,
                soft_min_days=soft_min_days,
                enable_soft_refresh=enable_soft_refresh,
            )
            print(
                f"worker: sync window={mover_window} snap={snap_id} "
                f"movers={counts['movers']} pending={counts['pending']} "
                f"never={counts['never']} stale={counts['stale']} fresh={counts['done_fresh']}",
                flush=True,
            )
            since_sync = 0

            if auto_steady and mover_window == BACKFILL_WINDOW:
                # Confirm never-analyzed across full backfill universe on this snap.
                never_left = count_never_analyzed(site=site, snap_id=snap_id, window=BACKFILL_WINDOW)
                if never_left == 0:
                    print(
                        f"worker: auto-steady — no never-analyzed left on {BACKFILL_WINDOW}; "
                        f"switching to {STEADY_WINDOW}",
                        flush=True,
                    )
                    mover_window = STEADY_WINDOW
                    conn = connect_insights()
                    try:
                        _save_worker_state(
                            conn,
                            site=site,
                            mover_window=mover_window,
                            last_snap_id=last_snap_id,
                            last_snap_at=last_snap_at,
                        )
                    finally:
                        conn.close()
                    counts = sync_eligibility(
                        site=site,
                        snap_id=snap_id,
                        window=mover_window,
                        fresh_days=fresh_days,
                        soft_min_days=soft_min_days,
                        enable_soft_refresh=enable_soft_refresh,
                    )
                    print(
                        f"worker: steady sync movers={counts['movers']} pending={counts['pending']}",
                        flush=True,
                    )

        reclaimed = reclaim_stale_leases(
            site=site, snap_id=snap_id, window=mover_window, lease_minutes=lease_minutes
        )
        if reclaimed:
            print(f"worker: reclaimed {reclaimed} stale lease(s)", flush=True)

        item = claim_next(
            site=site,
            snap_id=snap_id,
            window=mover_window,
            fresh_days=fresh_days,
            soft_min_days=soft_min_days,
            enable_soft_refresh=enable_soft_refresh,
            max_retries=max_retries,
        )
        if item is None:
            print(
                f"worker: idle — no due work (window={mover_window} fresh_days={fresh_days}); "
                f"sleep {idle_seconds}s",
                flush=True,
            )
            # Resync next wake so newly aged insights become due.
            since_sync = DEFAULT_RESYNC_EVERY
            time.sleep(idle_seconds)
            continue

        bcode = item["bcode"]
        tier = item.get("tier", "?")
        print(
            f"worker: claim {bcode} tier={tier} score={item.get('movement_score')} "
            f"window={mover_window}",
            flush=True,
        )
        try:
            res = generate_one(
                site=site,
                bcode=bcode,
                snap_id=snap_id,
                window=mover_window,
                facts_as_of=item.get("facts_as_of") or utc_now_iso(),
                prompt=prompt,
                model=model,
            )
            jobs_done += 1
            since_sync += 1
            print(
                f"worker: [{jobs_done}] {bcode} {res['elapsed_s']}s :: {(res.get('summary') or '')[:80]}",
                flush=True,
            )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            print(f"worker: ERROR {bcode}: {err}", flush=True)
            traceback.print_exc()
            _mark_error(
                site=site,
                bcode=bcode,
                snap_id=snap_id,
                window=mover_window,
                error=err,
                max_retries=max_retries,
            )
            since_sync += 1
            time.sleep(min(30, idle_seconds))
