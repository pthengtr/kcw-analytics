"""Due-status for the Supabase backfill (no PARTS9 / Spark)."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from src.kcw.product_insight_worker import (
    _delta_refresh_hold,
    covered_bill_date,
    delta_reason,
    delta_refresh_due,
    due_kind,
    next_delta_refresh_at,
)

BKK = ZoneInfo("Asia/Bangkok")


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


def test_daily_refresh_waits_until_19_bangkok():
    now = datetime(2026, 10, 21, 13, 0, tzinfo=BKK)
    last = "2026-10-21T03:30:20+07:00"
    assert delta_refresh_due(now, last) is False
    assert next_delta_refresh_at(now, last) == datetime(2026, 10, 21, 19, 0, tzinfo=BKK)
    msg = _delta_refresh_hold(last, every_hours=None, refresh_hour=19, now=now)
    assert msg == "worker: delta queue empty; next snap refresh at 2026-10-21 19:00 Asia/Bangkok"


def test_daily_refresh_runs_at_19_when_the_morning_snap_is_older():
    now = datetime(2026, 10, 21, 19, 5, tzinfo=BKK)
    assert delta_refresh_due(now, "2026-10-21T08:10:00+07:00") is True


def test_refresh_after_19_waits_until_the_next_evening():
    now = datetime(2026, 10, 21, 21, 0, tzinfo=BKK)
    last = "2026-10-21T19:10:00+07:00"
    assert delta_refresh_due(now, last) is False
    assert next_delta_refresh_at(now, last) == datetime(2026, 10, 22, 19, 0, tzinfo=BKK)


def test_missing_last_snap_is_due_immediately():
    now = datetime(2026, 10, 21, 8, 0, tzinfo=BKK)
    assert delta_refresh_due(now, None) is True


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
