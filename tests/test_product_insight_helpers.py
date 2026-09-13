"""Unit tests for product insight helpers (no PARTS9 / Spark required)."""

from src.kcw.product_insight_channels import billtype_std, channel_of
from src.kcw.product_insight_generate import parse_window
from datetime import date


def test_channel_online_and_transfer():
    assert channel_of("TAD6901-001", "1") == "online"
    assert channel_of("TFV6808-012", "1") == "transfer"
    assert channel_of("TF6808-001", "2") == "transfer"
    assert channel_of("8K69-001", "2") == "hq_store"
    assert channel_of("ANY", "0") == "excluded"


def test_billtype_std_prefixes():
    assert billtype_std("TAD1") == "TAD"
    assert billtype_std("CNTAD1") == "CNTAD"
    assert billtype_std("TFV1") == "TFV"


def test_parse_window():
    base = date(2026, 9, 13)
    assert parse_window("5y", as_of=base).isoformat() == "2021-09-14"
    assert parse_window("14d", as_of=base).isoformat() == "2026-08-30"
    assert parse_window("2w", as_of=base).isoformat() == "2026-08-30"
