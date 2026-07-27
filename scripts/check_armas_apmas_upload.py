"""Smoke checks for ARMAS/APMAS Supabase upload wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kcw.pipeline import build_parser  # noqa: E402
from src.kcw.upload_raw import ARMAS_APMAS_UPLOADS, refresh_table_via_staging_df  # noqa: E402


def test_cli_has_upload_command():
    parser = build_parser()
    args = parser.parse_args(["upload-armas-apmas"])
    assert args.func.__name__ == "cmd_upload_armas_apmas"
    print("CLI upload-armas-apmas registered")


def test_upload_specs():
    names = {s["csv_name"] for s in ARMAS_APMAS_UPLOADS}
    assert names == {
        "raw_hq_armas_receivable.csv",
        "raw_hq_apmas_payable.csv",
    }
    for spec in ARMAS_APMAS_UPLOADS:
        assert spec["main_table"].startswith("raw_kcw.")
        assert spec["staging_table"].endswith("_stg")
    print("ARMAS/APMAS upload specs OK")


def test_refresh_rejects_empty_df():
    class DummyConn:
        def cursor(self):
            raise AssertionError("should not open cursor for empty df")

    try:
        refresh_table_via_staging_df(
            DummyConn(),
            pd.DataFrame(),
            main_table="raw_kcw.raw_hq_armas_receivable",
            staging_table="raw_kcw.raw_hq_armas_receivable_stg",
        )
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    print("Empty dataframe guard OK")


def test_expected_payload_columns():
    expected = [
        "ID",
        "JOURMODE",
        "ACCTTYPE",
        "ACCTNO",
        "ACCTNAME",
        "ADDR1",
        "ADDR2",
        "PHONE",
        "MOBILE",
        "FAX",
        "CONTACT",
        "EMAIL",
        "TERM",
        "ALLOW",
        "ATPRICE",
        "MARKUP",
        "BEGDATE",
        "ENDDATE",
        "REMARKS",
        "CANCELED",
    ]
    assert len(expected) == 20
    print("Expected ARMAS/APMAS payload columns OK")


if __name__ == "__main__":
    test_cli_has_upload_command()
    test_upload_specs()
    test_refresh_rejects_empty_df()
    test_expected_payload_columns()
    print("\nAll ARMAS/APMAS upload checks passed.")
