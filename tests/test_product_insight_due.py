"""Due-status for the Supabase backfill (no PARTS9 / Spark)."""

from datetime import datetime, timezone

from src.kcw.product_insight_worker import covered_bill_date, delta_reason, due_kind


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


def test_morning_snap_stamp_closes_on_the_previous_day():
    # 2026-10-10 01:18 refresh included bills through 2026-10-09.
    covered = covered_bill_date(
        "2026-10-10T01:18:49+07:00",
        "2026-10-16T13:00:00+07:00",
    )
    assert covered == "2026-10-09"


def test_same_day_afternoon_stamp_includes_that_day():
    covered = covered_bill_date(
        "2026-10-16T15:00:00+07:00",
        "2026-10-16T15:05:00+07:00",
    )
    assert covered == "2026-10-16"


def test_stored_data_through_wins_over_the_stamp():
    covered = covered_bill_date(
        "2026-10-10T01:18:49+07:00",
        "2026-10-16T13:00:00+07:00",
        data_through="2026-10-16",
    )
    assert covered == "2026-10-16"


def test_delta_reason_never_moved_and_quiet():
    assert (
        delta_reason(
            generated_at=None,
            facts_as_of=None,
            data_through=None,
            max_bill="2026-10-16",
        )
        == "never"
    )
    assert (
        delta_reason(
            generated_at="2026-10-16T13:00:00+07:00",
            facts_as_of="2026-10-10T01:18:49+07:00",
            data_through=None,
            max_bill="2026-10-12",
        )
        == "moved"
    )
    assert (
        delta_reason(
            generated_at="2026-10-16T15:05:00+07:00",
            facts_as_of="2026-10-16T15:00:00+07:00",
            data_through="2026-10-16",
            max_bill="2026-10-16",
        )
        == "fresh"
    )
    assert (
        delta_reason(
            generated_at="2026-10-16T15:05:00+07:00",
            facts_as_of="2026-10-16T15:00:00+07:00",
            data_through="2026-10-16",
            max_bill="2026-10-17",
        )
        == "moved"
    )


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
