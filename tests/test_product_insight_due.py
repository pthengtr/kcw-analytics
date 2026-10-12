"""Due-status for the Supabase backfill (no PARTS9 / Spark)."""

from datetime import datetime, timezone

from src.kcw.product_insight_worker import due_kind


def test_old_local_row_is_stale_on_a_new_snap():
    kind = due_kind(
        local_generated_at="2026-09-15T01:46:20+07:00",
        remote_generated_at=None,
        fresh_days=14,
        now=datetime(2026, 10, 3, tzinfo=timezone.utc),
        enable_soft_refresh=False,
        soft_min_days=7,
    )
    assert kind == "stale"


def test_fresh_remote_row_wins_over_old_local():
    kind = due_kind(
        local_generated_at="2026-09-15T01:46:20+07:00",
        remote_generated_at="2026-10-02T12:00:00+07:00",
        fresh_days=14,
        now=datetime(2026, 10, 3, 0, 0, tzinfo=timezone.utc),
        enable_soft_refresh=False,
        soft_min_days=7,
    )
    assert kind == "fresh"


def test_missing_everywhere_is_never():
    kind = due_kind(
        local_generated_at=None,
        remote_generated_at=None,
        fresh_days=14,
        now=datetime(2026, 10, 3, tzinfo=timezone.utc),
        enable_soft_refresh=False,
        soft_min_days=7,
    )
    assert kind == "never"


def test_fresh_supabase_row_is_not_due():
    kind = due_kind(
        local_generated_at=None,
        remote_generated_at="2026-10-02T12:00:00+07:00",
        fresh_days=14,
        now=datetime(2026, 10, 3, 0, 0, tzinfo=timezone.utc),
        enable_soft_refresh=False,
        soft_min_days=7,
    )
    assert kind == "fresh"
