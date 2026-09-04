"""Unit tests for HQ→SYP ICMAS master sync (no live DB)."""

from __future__ import annotations

from src.kcw.sync_icmas_master import (
    BRANCH_DENYLIST,
    MASTER_ALLOWLIST,
    build_sync_plan,
    master_columns_present,
    normalize_bcode,
    normalize_value,
    values_equal,
)


def test_normalize_bcode_strips():
    assert normalize_bcode(" 01010044 ") == "01010044"
    assert normalize_bcode(None) == ""
    assert normalize_bcode(float("nan")) == ""


def test_values_equal_string_and_numeric():
    assert values_equal("  abc ", "abc")
    assert values_equal("", None)
    assert values_equal(1, 1.0)
    assert values_equal(25.0, 25.0000001)
    assert not values_equal("a", "b")
    assert not values_equal(10, 11)


def test_normalize_value_float_zero():
    assert normalize_value(0.0) == 0.0
    assert normalize_value(1e-15) == 0.0


def test_master_and_branch_disjoint():
    overlap = MASTER_ALLOWLIST & BRANCH_DENYLIST
    assert not overlap
    assert "QTYOH2" in BRANCH_DENYLIST
    assert "LOCATION1" in BRANCH_DENYLIST
    assert "PRICE1" in MASTER_ALLOWLIST
    assert "DESCR" in MASTER_ALLOWLIST
    assert "QTYMIN" in BRANCH_DENYLIST


def test_master_columns_present_filters():
    cols = ["BCODE", "DESCR", "PRICE1", "QTYOH2", "LOCATION1", "NOPE"]
    assert master_columns_present(cols) == ["DESCR", "PRICE1"]


def test_build_sync_plan_added_updated_syp_only():
    master_cols = ["DESCR", "PRICE1"]
    hq = {
        "A": {"BCODE": "A", "DESCR": "new", "PRICE1": 10.0, "QTYOH2": 99},
        "B": {"BCODE": "B", "DESCR": "hq", "PRICE1": 20.0, "QTYOH2": 5},
        "C": {"BCODE": "C", "DESCR": "same", "PRICE1": 1.0, "QTYOH2": 1},
    }
    syp = {
        "B": {"BCODE": "B", "DESCR": "old", "PRICE1": 20.0, "QTYOH2": 50},
        "C": {"BCODE": "C", "DESCR": "same", "PRICE1": 1.0, "QTYOH2": 9},
        "D": {"BCODE": "D", "DESCR": "syp-only", "PRICE1": 3.0, "QTYOH2": 7},
    }
    plan = build_sync_plan(hq, syp, master_cols=master_cols)

    assert plan.added == ["A"]
    assert plan.insert_payloads["A"]["DESCR"] == "new"
    assert plan.insert_payloads["A"]["QTYOH2"] == 0.0  # not HQ stock
    assert plan.insert_payloads["A"]["LOCATION1"] is None

    assert plan.updated == ["B"]
    assert plan.update_payloads["B"] == {"DESCR": "hq"}
    assert "QTYOH2" not in plan.update_payloads["B"]

    assert plan.unchanged == ["C"]
    assert plan.syp_only == ["D"]

    # Column change record
    assert len(plan.column_changes) == 1
    assert plan.column_changes[0].bcode == "B"
    assert plan.column_changes[0].column == "DESCR"
    assert plan.column_changes[0].old_syp == "old"
    assert plan.column_changes[0].new_hq == "hq"


def test_sql_bind_value_nan_nat():
    from src.kcw.sync_icmas_master import sql_bind_value
    import math
    import pandas as pd

    assert sql_bind_value(float("nan")) is None
    assert sql_bind_value(math.nan) is None
    assert sql_bind_value(pd.NaT) is None
    assert sql_bind_value(pd.NA) is None
    assert sql_bind_value(25.5) == 25.5
    assert sql_bind_value("x") == "x"

    hq = {str(i): {"BCODE": str(i), "DESCR": "x", "PRICE1": 1} for i in range(5)}
    syp = {"99": {"BCODE": "99", "DESCR": "only", "PRICE1": 1}}
    plan = build_sync_plan(hq, syp, master_cols=["DESCR", "PRICE1"], limit=2)
    assert len(plan.added) + len(plan.updated) + len(plan.unchanged) == 2
    assert plan.syp_only == ["99"]
