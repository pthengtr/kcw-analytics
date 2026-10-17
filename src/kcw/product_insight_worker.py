"""Continuous product-insight worker.

Default age mode: never analyzed, then age >= fresh_days, then optional soft refresh.
Delta mode (--delta): incremental snap refresh, then a finite queue of never-analyzed,
previous errors, and products whose snap bills are newer than the insight already stored.
When that queue drains, the next refresh builds a shorter one.
"""

from __future__ import annotations

import fcntl
import sqlite3
import time
import traceback
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from src.kcw.product_insight_generate import (
    SPARK_BASE,
    generate_one,
    load_prompt,
    parse_window,
    resolve_model,
    _open_snap,
    _snap_meta,
)
from src.kcw.product_insight_store import (
    connect_insights,
    init_insights_schema,
    insights_data_dir,
    parse_iso_dt,
    remote_generated_at,
    remote_insight_marks,
    resolve_snap_id,
    snap_path,
    utc_now_iso,
)

BACKFILL_WINDOW = "5y"
STEADY_WINDOW = "7d"
DELTA_WINDOW = "delta"
DEFAULT_FRESH_DAYS = 14
DEFAULT_DELTA_EVERY_HOURS = 6
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


def due_kind(
    *,
    local_generated_at: str | None,
    remote_generated_at: str | None,
    fresh_days: int,
    now: datetime,
    enable_soft_refresh: bool,
    soft_min_days: int,
) -> str:
    """Why a BCODE is or isn't due.

    The live timestamp is the Supabase row when one exists, otherwise the local
    SQLite row. Local rows are not frozen: a newer snap rewrites them once they
    are never / stale / soft under fresh_days.
    """
    generated_at = remote_generated_at or local_generated_at
    if generated_at is None:
        return "never"
    age = _age_days(generated_at, now=now)
    if age is not None and age >= fresh_days:
        return "stale"
    if enable_soft_refresh and age is not None and age >= soft_min_days:
        return "soft"
    return "fresh"


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

        # Age comes from Supabase, or from local SQLite when Supabase has no row.
        remote = remote_generated_at(site)
        rows = conn.execute(
            """
            SELECT q.bcode, q.status, q.retry_count, pi.generated_at
            FROM insight_queue q
            LEFT JOIN product_insights pi ON pi.site=q.site AND pi.bcode=q.bcode
            WHERE q.site=? AND q.snap_id=? AND q.window=?
            """,
            (site, snap_id, window),
        ).fetchall()

        counts = {
            "movers": len(movers),
            "pending": 0,
            "done_fresh": 0,
            "never": 0,
            "stale": 0,
            "soft": 0,
            "error": 0,
        }
        for r in rows:
            status = r["status"] or "pending"
            if status == "running":
                continue
            if status == "error" and int(r["retry_count"] or 0) >= DEFAULT_MAX_RETRIES:
                counts["error"] += 1
                continue

            kind = due_kind(
                local_generated_at=r["generated_at"],
                remote_generated_at=remote.get(str(r["bcode"])),
                fresh_days=fresh_days,
                now=now,
                enable_soft_refresh=enable_soft_refresh,
                soft_min_days=soft_min_days,
            )
            due = kind in ("never", "stale", "soft")
            if kind == "never":
                counts["never"] += 1
            elif kind == "stale":
                counts["stale"] += 1
            elif kind == "soft":
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
    """Movers on this snap with no insight in Supabase or local SQLite."""
    remote = set(remote_generated_at(site))
    conn = connect_insights()
    try:
        rows = conn.execute(
            """
            SELECT q.bcode FROM insight_queue q
            LEFT JOIN product_insights pi ON pi.site=q.site AND pi.bcode=q.bcode
            WHERE q.site=? AND q.snap_id=? AND q.window=?
              AND pi.bcode IS NULL
            """,
            (site, snap_id, window),
        ).fetchall()
        return sum(1 for r in rows if str(r["bcode"]) not in remote)
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
        remote = remote_generated_at(site)
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
            kind = due_kind(
                local_generated_at=r["generated_at"],
                remote_generated_at=remote.get(str(r["bcode"])),
                fresh_days=fresh_days,
                now=now,
                enable_soft_refresh=enable_soft_refresh,
                soft_min_days=soft_min_days,
            )
            if kind == "never":
                tier = 0
            elif kind == "stale":
                tier = 1
            elif kind == "soft":
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
        from src.kcw.product_insight_snapshot import refresh_snapshot_incremental, run_snapshot

        # Prefer in-place incremental refresh when a snap already exists.
        try:
            existing = resolve_snap_id("latest")
            if snap_path(existing).is_file():
                print(
                    f"worker: incremental snapshot refresh site={site} snap={existing}",
                    flush=True,
                )
                out = refresh_snapshot_incremental(snap=existing, lookback_days=1)
                snap_id = Path(out).parent.name if out is not None else existing
                return snap_id, utc_now_iso()
        except FileNotFoundError:
            pass

        print(f"worker: full snapshot site={site} years={years}", flush=True)
        out = run_snapshot(site=site, years=years, snap_id=None)
        if isinstance(out, dict):
            snap_id = out.get("snap_id") or resolve_snap_id("latest")
        else:
            # run_snapshot returns Path to snapshot.sqlite
            snap_id = Path(out).parent.name if out is not None else resolve_snap_id("latest")
        return snap_id, utc_now_iso()

    return resolve_snap_id("latest"), last_snap_at or utc_now_iso()


def covered_bill_date(
    facts_as_of: str | None,
    generated_at: str | None,
    data_through: str | None = None,
) -> str | None:
    """Last bill date treated as already inside an insight (YYYY-MM-DD).

    `data_through` is the snap max bill stored at generation. Older rows only
    have `facts_as_of`. A morning snap stamp (before 12:00) closed on the
    previous calendar day — the 2026-10-10 01:18 refresh included bills
    through 2026-10-09.
    """
    if data_through and str(data_through)[:10]:
        return str(data_through)[:10]
    if not facts_as_of or not generated_at:
        return None
    try:
        facts_day = date.fromisoformat(str(facts_as_of)[:10])
        gen_day = date.fromisoformat(str(generated_at)[:10])
    except ValueError:
        return None
    stamp = str(facts_as_of)
    if len(stamp) > 10 and stamp[10] == "T":
        hour = int(stamp[11:13]) if len(stamp) >= 13 and stamp[11:13].isdigit() else 12
        if hour < 12:
            facts_day = facts_day - timedelta(days=1)
    return min(facts_day, gen_day).isoformat()


def delta_reason(
    *,
    generated_at: str | None,
    facts_as_of: str | None,
    data_through: str | None,
    max_bill: str | None,
) -> str:
    """never | moved | fresh. Supabase absence is never, even if a local row exists."""
    if not generated_at:
        return "never"
    bill = (max_bill or "")[:10]
    covered = covered_bill_date(facts_as_of, generated_at, data_through)
    if bill and covered and bill > covered:
        return "moved"
    return "fresh"


@contextmanager
def _delta_rebuild_lock() -> Iterator[bool]:
    """Exclusive lock so only one worker refreshes the snap and rebuilds the queue."""
    path = insights_data_dir() / "delta-rebuild.lock"
    handle = open(path, "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        yield False
        return
    try:
        yield True
    finally:
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def _snap_bill_index(snap_id: str) -> tuple[str, list[tuple[str, str, int]]]:
    """Snap facts_as_of plus (bcode, max bill date, line count) for every SI/PI product."""
    snap = _open_snap(snap_id)
    try:
        meta = _snap_meta(snap)
        facts_as_of = meta.get("facts_as_of") or utc_now_iso()
        rows = snap.execute(
            """
            SELECT bcode, MAX(max_date) AS max_date, SUM(n) AS n FROM (
              SELECT bcode, MAX(billdate) AS max_date, COUNT(*) AS n
              FROM sidet GROUP BY bcode
              UNION ALL
              SELECT bcode, MAX(billdate) AS max_date, COUNT(*) AS n
              FROM pidet GROUP BY bcode
            )
            GROUP BY bcode
            """
        ).fetchall()
    finally:
        snap.close()
    indexed: list[tuple[str, str, int]] = []
    for row in rows:
        bcode = str(row["bcode"] or "").strip()
        if not bcode:
            continue
        indexed.append((bcode, str(row["max_date"] or "")[:10], int(row["n"] or 0)))
    return facts_as_of, indexed


def _delta_open_counts(conn: sqlite3.Connection, *, site: str, snap_id: str) -> tuple[int, int]:
    row = conn.execute(
        """
        SELECT
          SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending,
          SUM(CASE WHEN status='running' THEN 1 ELSE 0 END) AS running
        FROM insight_queue
        WHERE site=? AND snap_id=? AND window=?
        """,
        (site, snap_id, DELTA_WINDOW),
    ).fetchone()
    return int(row["pending"] or 0), int(row["running"] or 0)


def sync_delta_queue(*, site: str, snap_id: str) -> dict[str, int]:
    """Replace the delta queue with never-analyzed, prior errors, and post-insight movers.

    Quiet products stay out. Call only when nothing on this window is running.
    """
    facts_as_of, bills = _snap_bill_index(snap_id)
    marks = remote_insight_marks(site)
    now_iso = utc_now_iso()
    due: list[tuple[str, int, int]] = []
    counts = {"snap_products": len(bills), "never": 0, "moved": 0, "fresh": 0, "pending": 0}
    for bcode, max_date, score in bills:
        mark = marks.get(bcode)
        if mark is None:
            reason = "never"
        else:
            reason = delta_reason(
                generated_at=mark.get("generated_at"),
                facts_as_of=mark.get("facts_as_of"),
                data_through=mark.get("data_through"),
                max_bill=max_date,
            )
        if reason == "fresh":
            counts["fresh"] += 1
            continue
        counts[reason] += 1
        tier = 0 if reason == "never" else 1
        due.append((bcode, tier, score))

    conn = connect_insights()
    try:
        init_insights_schema(conn)
        prior_errors = {
            str(row["bcode"])
            for row in conn.execute(
                """
                SELECT bcode FROM insight_queue
                WHERE site=? AND snap_id=? AND window=? AND status='error'
                """,
                (site, snap_id, BACKFILL_WINDOW),
            )
        }
        counts["retry_error"] = sum(1 for bcode, _tier, _score in due if bcode in prior_errors)
        conn.execute(
            """
            UPDATE insight_queue
            SET status='done', leased_at=NULL, updated_at=?
            WHERE site=? AND snap_id=? AND window=?
            """,
            (now_iso, site, snap_id, DELTA_WINDOW),
        )
        conn.executemany(
            """
            INSERT INTO insight_queue (
              site, bcode, snap_id, facts_as_of, window, movement_score,
              status, updated_at, leased_at, retry_count, last_error, due_tier
            ) VALUES (?,?,?,?,?,?, 'pending', ?, NULL, 0, NULL, ?)
            ON CONFLICT(site, bcode, snap_id, window) DO UPDATE SET
              facts_as_of=excluded.facts_as_of,
              movement_score=excluded.movement_score,
              due_tier=excluded.due_tier,
              status='pending',
              updated_at=excluded.updated_at,
              leased_at=NULL,
              retry_count=0,
              last_error=NULL
            """,
            [
                (site, bcode, snap_id, facts_as_of, DELTA_WINDOW, score, now_iso, tier)
                for bcode, tier, score in due
            ],
        )
        conn.commit()
        counts["pending"] = len(due)
        return counts
    finally:
        conn.close()


def claim_next_delta(*, site: str, snap_id: str) -> dict[str, Any] | None:
    """Claim the next delta row. None if the queue is empty. {_contended: True} on a lost race."""
    now_iso = utc_now_iso()
    conn = connect_insights()
    try:
        init_insights_schema(conn)
        for _ in range(5):
            row = conn.execute(
                """
                SELECT site, bcode, snap_id, facts_as_of, window, movement_score, due_tier
                FROM insight_queue
                WHERE site=? AND snap_id=? AND window=? AND status='pending'
                ORDER BY COALESCE(due_tier, 1), movement_score DESC, bcode
                LIMIT 1
                """,
                (site, snap_id, DELTA_WINDOW),
            ).fetchone()
            if row is None:
                return None
            cur = conn.execute(
                """
                UPDATE insight_queue
                SET status='running', leased_at=?, updated_at=?
                WHERE site=? AND bcode=? AND snap_id=? AND window=?
                  AND status='pending'
                """,
                (now_iso, now_iso, site, row["bcode"], snap_id, DELTA_WINDOW),
            )
            if cur.rowcount == 1:
                conn.commit()
                return dict(row)
            conn.rollback()
        return {"_contended": True}
    finally:
        conn.close()


def _refresh_age_hours(last_snap_at: str | None) -> float | None:
    prev = parse_iso_dt(last_snap_at)
    if prev is None:
        return None
    now = datetime.now().astimezone()
    if prev.tzinfo is None:
        prev = prev.replace(tzinfo=timezone.utc)
    return (now - prev.astimezone(now.tzinfo)).total_seconds() / 3600.0


def _run_delta_worker(
    *,
    site: str,
    idle_seconds: int,
    lease_minutes: int,
    max_retries: int,
    years: int,
    max_jobs: int | None,
    model: str | None,
    delta_every_hours: int,
    state: dict[str, Any],
) -> int:
    """Drain a movement queue, then refresh the snap and build the next shorter one."""
    del years  # full extracts stay on the age-mode path; delta refreshes in place
    idle_seconds = max(15, int(idle_seconds))
    every_hours = max(1, int(delta_every_hours))
    last_snap_at = state.get("last_snap_at")
    prompt = load_prompt()
    model_id: str | None = None

    def _model() -> str:
        nonlocal model_id
        while not model_id:
            try:
                model_id = resolve_model(model)
            except Exception as exc:
                print(
                    f"worker: model lookup failed: {exc}; retry in {idle_seconds}s",
                    flush=True,
                )
                time.sleep(idle_seconds)
        return model_id

    print(
        f"worker start site={site} mode=delta every={every_hours}h "
        f"model={model or 'auto'} prompt={prompt.get('version')} spark={SPARK_BASE}",
        flush=True,
    )
    jobs_done = 0

    while True:
        if max_jobs is not None and jobs_done >= max_jobs:
            print(f"worker: max_jobs={max_jobs} reached, exiting", flush=True)
            return 0

        try:
            snap_id = resolve_snap_id("latest")
        except FileNotFoundError:
            print("worker: no snapshot yet; taking a full snapshot", flush=True)
            try:
                from src.kcw.product_insight_snapshot import run_snapshot

                out = run_snapshot(site=site, years=5, snap_id=None)
                snap_id = (
                    out.get("snap_id")
                    if isinstance(out, dict)
                    else Path(out).parent.name
                )
            except Exception as exc:
                print(f"worker: full snapshot failed: {exc}", flush=True)
                time.sleep(idle_seconds)
                continue

        conn = connect_insights()
        try:
            init_insights_schema(conn)
            reclaimed = reclaim_stale_leases(
                site=site,
                snap_id=snap_id,
                window=DELTA_WINDOW,
                lease_minutes=lease_minutes,
            )
            pending, running = _delta_open_counts(conn, site=site, snap_id=snap_id)
            state = _load_worker_state(conn, site)
            last_snap_at = state.get("last_snap_at") or last_snap_at
        finally:
            conn.close()
        if reclaimed:
            print(f"worker: reclaimed {reclaimed} stale lease(s)", flush=True)
            continue

        if pending:
            _model()
            item = claim_next_delta(site=site, snap_id=snap_id)
            if item is None or item.get("_contended"):
                time.sleep(1)
                continue
            bcode = item["bcode"]
            raw_tier = item.get("due_tier")
            tier = 1 if raw_tier is None else int(raw_tier)
            print(
                f"worker: claim {bcode} tier={tier} score={item.get('movement_score')} "
                f"window={DELTA_WINDOW}",
                flush=True,
            )
            try:
                res = generate_one(
                    site=site,
                    bcode=bcode,
                    snap_id=snap_id,
                    window=DELTA_WINDOW,
                    facts_as_of=item.get("facts_as_of") or utc_now_iso(),
                    prompt=prompt,
                    model=_model(),
                )
                jobs_done += 1
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
                    window=DELTA_WINDOW,
                    error=err,
                    max_retries=max_retries,
                )
                time.sleep(min(30, idle_seconds))
            continue

        if running:
            print(
                f"worker: waiting on {running} running delta job(s); sleep {idle_seconds}s",
                flush=True,
            )
            time.sleep(idle_seconds)
            continue

        age_h = _refresh_age_hours(last_snap_at)
        if age_h is not None and age_h < every_hours:
            wait_h = every_hours - age_h
            print(
                f"worker: delta queue empty; next snap refresh in {wait_h:.1f}h "
                f"(every {every_hours}h)",
                flush=True,
            )
            time.sleep(idle_seconds)
            continue

        refresh_failed = False
        with _delta_rebuild_lock() as got:
            if not got:
                print("worker: delta refresh already running; waiting", flush=True)
                time.sleep(idle_seconds)
                continue
            conn = connect_insights()
            try:
                pending, running = _delta_open_counts(conn, site=site, snap_id=snap_id)
                state = _load_worker_state(conn, site)
            finally:
                conn.close()
            if pending or running:
                continue
            age_h = _refresh_age_hours(state.get("last_snap_at"))
            if age_h is not None and age_h < every_hours:
                continue
            refresh_failed = False
            try:
                from src.kcw.product_insight_snapshot import refresh_snapshot_incremental

                print(f"worker: incremental snapshot refresh site={site} snap={snap_id}", flush=True)
                refresh_snapshot_incremental(snap=snap_id, lookback_days=1)
                counts = sync_delta_queue(site=site, snap_id=snap_id)
            except Exception as exc:
                refresh_failed = True
                print(f"worker: delta refresh failed: {exc}", flush=True)
                traceback.print_exc()
            else:
                refreshed_at = utc_now_iso()
                conn = connect_insights()
                try:
                    _save_worker_state(
                        conn,
                        site=site,
                        mover_window=DELTA_WINDOW,
                        last_snap_id=snap_id,
                        last_snap_at=refreshed_at,
                    )
                finally:
                    conn.close()
                last_snap_at = refreshed_at
                print(
                    f"worker: delta sync snap={snap_id} "
                    f"products={counts['snap_products']} pending={counts['pending']} "
                    f"never={counts['never']} of_which_prior_error={counts['retry_error']} "
                    f"moved={counts['moved']} quiet={counts['fresh']}",
                    flush=True,
                )
        if refresh_failed:
            time.sleep(idle_seconds)

    return 0


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
    model: str | None = None,
    delta: bool = False,
    no_delta: bool = False,
    delta_every_hours: int = DEFAULT_DELTA_EVERY_HOURS,
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

    saved_window = str(state.get("mover_window") or "")
    if not no_delta and (delta or saved_window == DELTA_WINDOW):
        return _run_delta_worker(
            site=site,
            idle_seconds=idle_seconds,
            lease_minutes=lease_minutes,
            max_retries=max_retries,
            years=years,
            max_jobs=max_jobs,
            model=model,
            delta_every_hours=delta_every_hours,
            state=state,
        )

    if saved_window and saved_window != DELTA_WINDOW:
        # Resume prior age-mode window unless the caller forced delta above.
        mover_window = saved_window
    last_snap_at = state.get("last_snap_at")
    last_snap_id = state.get("last_snap_id")

    prompt = load_prompt()
    model_id = resolve_model(model)
    print(
        f"worker start site={site} window={mover_window} fresh_days={fresh_days} "
        f"auto_steady={auto_steady} soft={enable_soft_refresh} snap_every={snap_every_days}d "
        f"model={model_id} prompt={prompt.get('version')} spark={SPARK_BASE}",
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
                model=model_id,
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
