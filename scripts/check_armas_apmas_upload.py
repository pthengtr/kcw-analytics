"""Smoke checks for daily raw Supabase upload wiring."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kcw.pipeline import build_parser  # noqa: E402
from src.kcw.upload_raw import (  # noqa: E402
    ARMAS_APMAS_UPLOADS,
    BRDET_BPDET_UPLOADS,
    DAILY_RAW_UPLOADS,
    ICLOW_UPLOADS,
    ICMAS_UPLOADS,
    PIMAS_PIDET_UPLOADS,
    POMAS_PODET_UPLOADS,
    PVMAS_UPLOADS,
    RVMAS_UPLOADS,
    SIMAS_SIDET_UPLOADS,
    refresh_table_via_staging_df,
)


def test_cli_has_upload_commands():
    parser = build_parser()
    args = parser.parse_args(["upload-armas-apmas"])
    assert args.func.__name__ == "cmd_upload_armas_apmas"
    args = parser.parse_args(["upload-daily-raw"])
    assert args.func.__name__ == "cmd_upload_daily_raw"
    args = parser.parse_args(["upload-pomas-podet", "--site", "hq"])
    assert args.func.__name__ == "cmd_upload_pomas_podet"
    args = parser.parse_args(["sync-pomas-podet", "--site", "syp"])
    assert args.func.__name__ == "cmd_sync_pomas_podet"
    args = parser.parse_args(["sync-iclow", "--site", "hq"])
    assert args.func.__name__ == "cmd_sync_iclow"
    args = parser.parse_args(["sync-iclow", "--site", "syp"])
    assert args.func.__name__ == "cmd_sync_iclow"
    args = parser.parse_args(["upload-iclow", "--site", "hq"])
    assert args.func.__name__ == "cmd_upload_iclow"
    args = parser.parse_args(["upload-iclow"])
    assert args.func.__name__ == "cmd_upload_iclow"
    assert args.site is None
    args = parser.parse_args(["sync-icmas", "--site", "hq"])
    assert args.func.__name__ == "cmd_sync_icmas"
    args = parser.parse_args(["sync-icmas", "--site", "syp"])
    assert args.func.__name__ == "cmd_sync_icmas"
    args = parser.parse_args(["upload-icmas", "--site", "hq"])
    assert args.func.__name__ == "cmd_upload_icmas"
    args = parser.parse_args(["upload-icmas"])
    assert args.func.__name__ == "cmd_upload_icmas"
    assert args.site is None
    args = parser.parse_args(["sync-po-related", "--site", "hq"])
    assert args.func.__name__ == "cmd_sync_po_related"
    assert args.site == "hq"
    args = parser.parse_args(["sync-po-related", "--site", "syp"])
    assert args.func.__name__ == "cmd_sync_po_related"
    assert args.site == "syp"
    args = parser.parse_args(["upload-po-related", "--site", "hq"])
    assert args.func.__name__ == "cmd_upload_po_related"
    args = parser.parse_args(["sync-simas-sidet"])
    assert args.func.__name__ == "cmd_sync_simas_sidet"
    args = parser.parse_args(["upload-simas-sidet"])
    assert args.func.__name__ == "cmd_upload_simas_sidet"
    args = parser.parse_args(["sync-brdet-bpdet"])
    assert args.func.__name__ == "cmd_sync_brdet_bpdet"
    args = parser.parse_args(["upload-brdet-bpdet"])
    assert args.func.__name__ == "cmd_upload_brdet_bpdet"
    args = parser.parse_args(["extract", "--site", "hq", "--tables", "POMAS,PODET"])
    assert args.tables == "POMAS,PODET"
    print("CLI upload/sync PO/ICLOW/ICMAS/po-related/brdet-bpdet commands registered")


def test_upload_specs():
    armas = {s["csv_name"] for s in ARMAS_APMAS_UPLOADS}
    assert armas == {
        "raw_hq_armas_receivable.csv",
        "raw_hq_apmas_payable.csv",
    }
    po = {s["csv_name"] for s in POMAS_PODET_UPLOADS}
    assert po == {
        "raw_hq_pomas_purchase_orders.csv",
        "raw_hq_podet_purchase_order_lines.csv",
        "raw_syp_pomas_purchase_orders.csv",
        "raw_syp_podet_purchase_order_lines.csv",
    }
    pi = {s["csv_name"] for s in PIMAS_PIDET_UPLOADS}
    assert pi == {
        "raw_hq_pimas_purchase_bills.csv",
        "raw_hq_pidet_purchase_lines.csv",
    }
    icmas = {s["csv_name"] for s in ICMAS_UPLOADS}
    assert icmas == {
        "raw_hq_icmas_products.csv",
        "raw_syp_icmas_products.csv",
    }
    rvmas = {s["csv_name"] for s in RVMAS_UPLOADS}
    assert rvmas == {"raw_hq_rvmas_notes_vouchers.csv"}
    pvmas = {s["csv_name"] for s in PVMAS_UPLOADS}
    assert pvmas == {"raw_hq_pvmas_notes_vouchers.csv"}
    cheque = {s["csv_name"] for s in BRDET_BPDET_UPLOADS}
    assert cheque == {
        "raw_hq_brdet_cheques_received.csv",
        "raw_hq_bpdet_cheques_paid.csv",
    }
    iclow = {s["csv_name"] for s in ICLOW_UPLOADS}
    assert iclow == {
        "raw_hq_iclow_stock_orders.csv",
        "raw_syp_iclow_stock_orders.csv",
    }
    si = {s["csv_name"] for s in SIMAS_SIDET_UPLOADS}
    assert si == {
        "raw_hq_sidet_sales_lines.csv",
        "raw_hq_simas_sales_bills.csv",
    }
    for spec in SIMAS_SIDET_UPLOADS:
        assert spec["months"] == 6
        assert spec["date_col"] == "BILLDATE"
    assert len(DAILY_RAW_UPLOADS) == (
        len(ARMAS_APMAS_UPLOADS)
        + len(POMAS_PODET_UPLOADS)
        + len(PIMAS_PIDET_UPLOADS)
        + len(ICMAS_UPLOADS)
        + len(RVMAS_UPLOADS)
        + len(PVMAS_UPLOADS)
        + len(BRDET_BPDET_UPLOADS)
        + len(ICLOW_UPLOADS)
        + len(SIMAS_SIDET_UPLOADS)
    )
    for spec in DAILY_RAW_UPLOADS:
        assert spec["main_table"].startswith("raw_kcw.")
        assert spec["staging_table"].endswith("_stg")
    print("Daily raw upload specs OK")


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


def test_filter_last_months_from_latest():
    from src.kcw.supabase_utils import filter_last_months_from_latest

    df = pd.DataFrame(
        {
            "BILLDATE": ["2026-01-01", "2026-06-15", "2025-12-01"],
            "ID": ["1", "2", "3"],
        }
    )
    out = filter_last_months_from_latest(df, "BILLDATE", months=6)
    assert set(out["ID"]) == {"1", "2"}
    print("6-month date filter OK")


def test_extract_includes_pomas_podet():
    from src.kcw.extract_parts9 import (
        CHEQUE_TABLES,
        ICLOW_TABLES,
        ICMAS_TABLES,
        PO_RELATED_TABLES,
        PO_TABLES,
        SI_TABLES,
        SYP_MINIMAL,
        TABLE_SPECS,
    )

    assert "POMAS" in TABLE_SPECS and "PODET" in TABLE_SPECS
    assert TABLE_SPECS["POMAS"]["date_col"] == "DOCDATE"
    assert TABLE_SPECS["PODET"]["date_col"] == "DOCDATE"
    assert "POMAS" in SYP_MINIMAL and "PODET" in SYP_MINIMAL
    assert "PIMAS" not in SYP_MINIMAL and "PIDET" not in SYP_MINIMAL
    assert PO_TABLES == ("PODET", "POMAS")
    assert "ICLOW" in TABLE_SPECS
    assert TABLE_SPECS["ICLOW"]["years"] is None
    assert TABLE_SPECS["ICLOW"]["suffix"] == "iclow_stock_orders"
    assert ICLOW_TABLES == ("ICLOW",)
    assert "ICLOW" in SYP_MINIMAL
    assert "ICMAS" in TABLE_SPECS
    assert TABLE_SPECS["ICMAS"]["years"] is None
    assert TABLE_SPECS["ICMAS"]["suffix"] == "icmas_products"
    assert ICMAS_TABLES == ("ICMAS",)
    assert "ICMAS" in SYP_MINIMAL
    assert PO_RELATED_TABLES == ("PODET", "POMAS", "ICLOW")
    assert SI_TABLES == ("SIDET", "SIMAS")
    assert "ICMAS" not in PO_RELATED_TABLES
    assert "BRDET" in TABLE_SPECS and "BPDET" in TABLE_SPECS
    assert TABLE_SPECS["BRDET"]["suffix"] == "brdet_cheques_received"
    assert TABLE_SPECS["BPDET"]["suffix"] == "bpdet_cheques_paid"
    assert CHEQUE_TABLES == ("BRDET", "BPDET")
    assert "BRDET" not in SYP_MINIMAL and "BPDET" not in SYP_MINIMAL
    print("Extract TABLE_SPECS / SYP_MINIMAL / PO_TABLES / ICLOW / ICMAS / PO_RELATED / SI / CHEQUE OK")


def test_supabase_db_url_rejects_https_api_url(monkeypatch_env=None):
    import os

    from src.kcw.tar import supabase_db_url

    old = {
        k: os.environ.get(k)
        for k in (
            "SUPABASE_DB_URL",
            "DB_PASSWORD",
            "SUPABASE_DB_PASSWORD",
            "SUPABASE_DB_HOST",
            "SUPABASE_DB_USER",
            "SUPABASE_DB_PORT",
            "SUPABASE_DB_NAME",
        )
    }
    try:
        os.environ["SUPABASE_DB_URL"] = "https://example.supabase.co"
        os.environ["DB_PASSWORD"] = "secret-pass"
        os.environ["SUPABASE_DB_HOST"] = "aws-0-ap-southeast-1.pooler.supabase.com"
        os.environ["SUPABASE_DB_USER"] = "postgres.jdzitzsucntqbjvwiwxm"
        os.environ["SUPABASE_DB_PORT"] = "5432"
        os.environ["SUPABASE_DB_NAME"] = "postgres"
        url = supabase_db_url()
        assert url.startswith("postgresql://"), url
        assert "secret-pass" in url
        assert "https://" not in url
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print("HTTPS API URL fallback OK")


if __name__ == "__main__":
    test_cli_has_upload_commands()
    test_upload_specs()
    test_refresh_rejects_empty_df()
    test_filter_last_months_from_latest()
    test_extract_includes_pomas_podet()
    test_supabase_db_url_rejects_https_api_url()
    print("\nAll daily-raw upload checks passed.")
