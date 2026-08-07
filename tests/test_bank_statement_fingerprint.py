"""Unit tests for auto_v2 bank statement transaction fingerprints.

Mirrors kcw-v2 fingerprint.test.ts (incl. 3557 overlapping-import regression).
"""

from __future__ import annotations

import hashlib
from datetime import date
from decimal import Decimal

from src.kcw.bank_statement import (
    compute_transaction_fingerprint,
    extract_stable_transaction_detail,
)


def _fp(**kwargs) -> str:
    return compute_transaction_fingerprint(**kwargs)


def test_description_drift_same_fingerprint():
    """Same real txn with different display รายการ must share fingerprint."""
    detail = "จาก KTB X8740 MISS NARUMON WITHA++"
    base = dict(
        account_no="064-8-92039-3",
        txn_date=date(2026, 5, 1),
        amount=Decimal("33718.50"),
        direction="in",
        bank_reference=None,
        balance_after=Decimal("139636.74"),
    )
    fp_time = _fp(
        **base,
        raw_json={"รายการ": "09:12:00", "รายละเอียด": detail},
        description="09:12:00",
    )
    fp_label = _fp(
        **base,
        raw_json={"รายการ": "รับโอนเงิน", "รายละเอียด": detail},
        description="รับโอนเงิน",
    )
    assert fp_time == fp_label
    assert len(fp_time) == 64


def test_display_description_not_used_even_when_passed():
    """description kwarg must not change identity when stable detail exists."""
    detail = "โอนไป SCB X7654 บริษัท  คูโบต้า ก.++"
    raw = {"รายละเอียด": detail, "รายการ": "โอนเงิน"}
    a = _fp(
        account_no="141-1-72355-7",
        txn_date=date(2026, 4, 1),
        amount=3866,
        direction="out",
        bank_reference=None,
        balance_after=Decimal("130186.73"),
        raw_json=raw,
        description="โอนเงิน",
    )
    b = _fp(
        account_no="141-1-72355-7",
        txn_date=date(2026, 4, 1),
        amount=3866,
        direction="out",
        bank_reference=None,
        balance_after=Decimal("130186.73"),
        raw_json=raw,
        description="08:31:00",
    )
    assert a == b


def test_no_fallback_to_display_description_when_detail_missing():
    """Missing stable keys must NOT hash รายการ/time (auto_v1 bug class)."""
    fp_a = _fp(
        account_no="141-1-72355-7",
        txn_date=date(2026, 4, 1),
        amount=3866,
        direction="out",
        bank_reference=None,
        balance_after=Decimal("130186.73"),
        raw_json={"รายการ": "โอนเงิน"},
        description="โอนเงิน",
    )
    fp_b = _fp(
        account_no="141-1-72355-7",
        txn_date=date(2026, 4, 1),
        amount=3866,
        direction="out",
        bank_reference=None,
        balance_after=Decimal("130186.73"),
        raw_json={"รายการ": "08:31:00"},
        description="08:31:00",
    )
    # Both have empty stable detail → same fingerprint (dedupe still works).
    assert fp_a == fp_b
    assert extract_stable_transaction_detail({"รายการ": "โอนเงิน"}) == ""


def test_legitimate_same_day_same_amount_different_balance():
    """balance_after disambiguates two real transfers on the same day."""
    common = dict(
        account_no="064-8-92039-3",
        txn_date=date(2026, 5, 25),
        amount=1000,
        direction="in",
        bank_reference=None,
    )
    first = _fp(
        **common,
        balance_after=100500,
        raw_json={"รายละเอียด": "จาก X3557 บจก. เกียรติชัยอะไ++"},
    )
    second = _fp(
        **common,
        balance_after=101500,
        raw_json={"รายละเอียด": "จาก KTB X8740 MISS NARUMON WITHA++"},
    )
    assert first != second


def test_overlapping_3557_april_exports_share_fingerprint():
    """04_3557.xlsx vs 3557 ด.4.xlsx style: same detail+balance, different รายการ."""
    detail = "โอนไป SCB X7654 บริษัท  คูโบต้า ก.++"
    from_april = _fp(
        account_no="141-1-72355-7",
        txn_date=date(2026, 4, 1),
        amount=3866,
        direction="out",
        bank_reference=None,
        balance_after=Decimal("130186.73"),
        raw_json={"รายการ": "โอนเงิน", "รายละเอียด": detail},
        description="โอนเงิน",
    )
    from_month4 = _fp(
        account_no="141-1-72355-7",
        txn_date=date(2026, 4, 1),
        amount=3866,
        direction="out",
        bank_reference=None,
        balance_after=Decimal("130186.73"),
        raw_json={"รายการ": "08:31:00", "รายละเอียด": detail},
        description="08:31:00",
    )
    assert from_april == from_month4

    # auto_v1 would have hashed description and missed the duplicate.
    v1_a = hashlib.sha256(
        "141-1-72355-7|2026-04-01|3866.00|OUT|โอนเงิน||130186.73".encode()
    ).hexdigest()
    v1_b = hashlib.sha256(
        "141-1-72355-7|2026-04-01|3866.00|OUT|08:31:00||130186.73".encode()
    ).hexdigest()
    assert v1_a != v1_b


def test_stable_detail_keys_include_narration_and_particular():
    assert extract_stable_transaction_detail({"NARRATION": "N1"}) == "N1"
    assert extract_stable_transaction_detail({"PARTICULAR": "P1"}) == "P1"
    assert extract_stable_transaction_detail({"DESCRIPTION": "D1"}) == "D1"
    # Prefer รายละเอียด over DESCRIPTION when both present.
    assert (
        extract_stable_transaction_detail(
            {"DESCRIPTION": "EN", "รายละเอียด": "TH"}
        )
        == "TH"
    )


def test_transaction_detail_kwarg_overrides_raw():
    fp = _fp(
        account_no="141-1-72355-7",
        txn_date=date(2026, 4, 1),
        amount=1,
        direction="in",
        bank_reference=None,
        balance_after=10,
        raw_json={"รายละเอียด": "from-raw"},
        transaction_detail="from-kwarg",
    )
    fp2 = _fp(
        account_no="141-1-72355-7",
        txn_date=date(2026, 4, 1),
        amount=1,
        direction="in",
        bank_reference=None,
        balance_after=10,
        transaction_detail="from-kwarg",
    )
    assert fp == fp2
