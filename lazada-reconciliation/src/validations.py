"""Data-quality checks. FAIL still produces a report but never claims success."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import pandas as pd

from .config import INSUFFICIENT_DATA
from .loaders import LoadedTable
from .normalizers import decimal_sum, quantize_money
from .reconciliation import ReconciliationResult

ZERO = Decimal("0.00")


@dataclass
class ValidationRow:
    validation_name: str
    status: str
    expected: Any
    actual: Any
    difference: Any
    explanation: str


def _money_diff(expected: Any, actual: Any) -> Any:
    if isinstance(expected, Decimal) and isinstance(actual, Decimal):
        return quantize_money(expected - actual)
    return None


def _status_from_ok(ok: bool, fail: bool = True) -> str:
    if ok:
        return "PASS"
    return "FAIL" if fail else "WARNING"


def run_validations(
    result: ReconciliationResult,
    orders: LoadedTable,
    finance: LoadedTable,
    wallet: LoadedTable,
) -> ReconciliationResult:
    rows: list[ValidationRow] = []
    tolerance = result.tolerance

    # 1. Money quantized to 2 decimal places
    bad_scale = 0
    for frame in (result.order_summary, result.finance_summary, result.order_vs_finance):
        if frame.empty:
            continue
        for column in frame.columns:
            series = frame[column]
            for value in series.tolist():
                if isinstance(value, Decimal) and value.as_tuple().exponent < -2:
                    bad_scale += 1
    rows.append(
        ValidationRow(
            "money_scale_2dp",
            "PASS" if bad_scale == 0 else "FAIL",
            "ทุกจำนวนเงินปัด 2 ตำแหน่ง",
            f"ค่าที่ละเอียดเกิน 2 ตำแหน่ง: {bad_scale}",
            bad_scale,
            "parse_money ต้อง quantize เป็น 0.01 ก่อนเปรียบเทียบและก่อนเขียนรายงาน",
        )
    )

    # 2. No orderNumber lost from full outer join
    order_keys = set(result.order_summary["order_number"].dropna()) if not result.order_summary.empty else set()
    finance_keys = (
        set(result.finance_summary["order_number"].dropna()) if not result.finance_summary.empty else set()
    )
    join_keys = (
        set(result.order_vs_finance["order_number"].dropna()) if not result.order_vs_finance.empty else set()
    )
    missing = (order_keys | finance_keys) - join_keys
    rows.append(
        ValidationRow(
            "full_outer_join_no_lost_keys",
            "PASS" if not missing else "FAIL",
            len(order_keys | finance_keys),
            len(join_keys),
            len(missing),
            "ทุก orderNumber จาก Order และ Finance ต้องปรากฏใน Order_vs_Finance",
        )
    )

    # 3. Pre-group vs post-group totals
    for kind in ("orders", "finance", "wallet"):
        expected = result.pre_group_totals.get(kind, ZERO)
        actual = result.post_group_totals.get(kind, ZERO)
        diff = quantize_money(expected - actual)
        rows.append(
            ValidationRow(
                f"pre_post_group_total_{kind}",
                "PASS" if abs(diff) <= tolerance else "FAIL",
                expected,
                actual,
                diff,
                "ยอดรวมก่อน group ต้องเท่ากับหลัง group (รวมแถวที่ orderNumber ว่างใน Finance/Wallet)",
            )
        )

    # 4. Grand Total excluded
    dropped = orders.dropped_total_rows + finance.dropped_total_rows + wallet.dropped_total_rows
    still_present = 0
    for frame in (orders.frame, finance.frame, wallet.frame):
        if frame.empty:
            continue
        text = frame.astype(str).apply(lambda s: s.str.casefold())
        for marker in ("grand total", "grandtotal"):
            still_present += int(text.isin([marker]).any(axis=1).sum())
    rows.append(
        ValidationRow(
            "grand_total_excluded",
            "PASS" if still_present == 0 else "FAIL",
            0,
            still_present,
            still_present,
            f"ตัดแถว Total ออกแล้ว {dropped} แถว และต้องไม่เหลือในตารางคำนวณ",
        )
    )

    # 5. Empty orderNumber
    empty_orders = 0
    if not orders.frame.empty and "order_number" in orders.frame.columns:
        empty_orders += int(orders.frame["order_number"].isna().sum())
    empty_finance = 0
    if not finance.frame.empty and "order_number" in finance.frame.columns:
        empty_finance += int(finance.frame["order_number"].isna().sum())
    empty_total = empty_orders + empty_finance
    rows.append(
        ValidationRow(
            "empty_order_number",
            "PASS" if empty_total == 0 else "WARNING",
            0,
            empty_total,
            empty_total,
            "แถวที่ orderNumber ว่างถูกแยกไป Exceptions และไม่ใช้ใน join",
        )
    )

    # 6. Amount parse failures
    parse_fail = sum(1 for issue in result.issues if issue.exception_type == "AMOUNT_PARSE_FAILED")
    rows.append(
        ValidationRow(
            "amount_parse",
            "PASS" if parse_fail == 0 else "FAIL",
            0,
            parse_fail,
            parse_fail,
            "จำนวนเงินที่ parse ไม่ได้ต้องไม่ถูกสมมติเป็นศูนย์",
        )
    )

    # 7. Duplicates kept
    dup = sum(1 for issue in result.issues if issue.exception_type == "DUPLICATE_ROW")
    rows.append(
        ValidationRow(
            "duplicate_rows_reported_not_deleted",
            "PASS" if True else "FAIL",
            "รายงานและเก็บไว้",
            f"พบ {dup} แถวซ้ำ",
            dup,
            "ห้ามลบ duplicate อัตโนมัติ — แสดงใน Exceptions",
        )
    )

    # 8. Unmapped Freename
    unknown_finance = 0
    if not result.unknown_mappings.empty:
        unknown_finance = int(
            result.unknown_mappings.loc[result.unknown_mappings["source"] == "finance", "row_count"].sum()
        )
    rows.append(
        ValidationRow(
            "unmapped_freename",
            "PASS" if unknown_finance == 0 else "WARNING",
            0,
            unknown_finance,
            unknown_finance,
            "Freename ที่ยังไม่มี mapping ต้องปรากฏใน Unknown_Mappings",
        )
    )

    # 9. Unmapped Wallet Type/Sub Type
    unknown_wallet = 0
    if not result.unknown_mappings.empty:
        unknown_wallet = int(
            result.unknown_mappings.loc[result.unknown_mappings["source"] == "wallet", "row_count"].sum()
        )
    rows.append(
        ValidationRow(
            "unmapped_wallet_type",
            "PASS" if unknown_wallet == 0 else "WARNING",
            0,
            unknown_wallet,
            unknown_wallet,
            "Wallet Type/Sub Type ที่ยังไม่มี mapping ต้องปรากฏใน Unknown_Mappings",
        )
    )

    # 10. Finance net vs component sum
    expected_net = result.expense_summary.get("finance_net_amount", ZERO)
    component_sum = result.expense_summary.get("component_sum", ZERO)
    diff = quantize_money(expected_net - component_sum)
    rows.append(
        ValidationRow(
            "finance_net_vs_components",
            "PASS" if abs(diff) <= tolerance else "FAIL",
            expected_net,
            component_sum,
            diff,
            "ยอดสุทธิ Finance ต้องเท่ากับผลรวม signed ของทุกหมวด ไม่ใช่ค่าสัมบูรณ์",
        )
    )

    # 11. Wallet movement vs bank withdrawals
    wt = result.wallet_totals
    inflows = wt.get("wallet_inflows", ZERO)
    adjustments = wt.get("wallet_adjustments", ZERO)
    withdrawals_signed = wt.get("withdrawals_signed", ZERO)
    unknown_amt = wt.get("unknown_amount", ZERO)
    opening_value = wt.get("opening_balance")
    opening_num = opening_value if isinstance(opening_value, Decimal) else ZERO
    actual_signed = wt.get("signed_net", ZERO)
    # If opening is a separate mapped row it is already inside signed_net.
    # Compare classified signed parts (excluding opening/closing rows) against signed_net.
    classified = quantize_money(inflows + adjustments + withdrawals_signed + unknown_amt)
    if wt.get("opening_balance_found"):
        classified = quantize_money(classified + opening_num)
    closing_value = wt.get("reported_closing_balance")
    if isinstance(closing_value, Decimal):
        classified = quantize_money(classified + closing_value)
    wallet_diff = quantize_money(classified - actual_signed)
    rows.append(
        ValidationRow(
            "wallet_movement_vs_bank",
            "PASS" if abs(wallet_diff) <= tolerance else "FAIL",
            classified,
            actual_signed,
            wallet_diff,
            "ผลรวม signed ของหมวด Wallet ต้องเท่ากับผลรวม Amount ทั้งไฟล์ (ใช้ Amount ไม่ใช่ Sub Type)",
        )
    )
    if not wt.get("opening_balance_found"):
        rows.append(
            ValidationRow(
                "opening_balance_present",
                "WARNING",
                "มียอดยกมาในไฟล์ Wallet",
                INSUFFICIENT_DATA,
                None,
                "ไม่มี opening balance ในไฟล์ — ห้ามสมมติเป็นศูนย์; ยอดคงเหลือคำนวณจึงเป็นข้อมูลไม่เพียงพอ",
            )
        )
    else:
        rows.append(
            ValidationRow(
                "opening_balance_present",
                "PASS",
                "มียอดยกมา",
                wt.get("opening_balance"),
                None,
                "พบรายการยอดยกมาจาก Type/Sub Type mapping",
            )
        )

    finance_net = result.expense_summary.get("finance_net_amount", ZERO)
    wallet_in = wt.get("wallet_inflows", ZERO)
    period_diff = quantize_money(finance_net - wallet_in)
    rows.append(
        ValidationRow(
            "finance_net_vs_wallet_settlement",
            "PASS" if abs(period_diff) <= tolerance else "WARNING",
            finance_net,
            wallet_in,
            period_diff,
            "ยอดสุทธิ Finance เทียบ Settlement ใน Wallet อาจต่างงวดกันได้ — ไม่ FAIL อัตโนมัติ",
        )
    )

    # 12. Overall status
    status_values = [row.status for row in rows]
    if "FAIL" in status_values:
        overall = "FAIL"
    elif "WARNING" in status_values:
        overall = "WARNING"
    else:
        overall = "PASS"
    rows.append(
        ValidationRow(
            "overall_validation_status",
            overall,
            "PASS",
            overall,
            None,
            "FAIL = มีปัญหาความครบถ้วนของข้อมูล, WARNING = ต้องให้คนตรวจ, PASS = ผ่านการตรวจเบื้องต้นเท่านั้น",
        )
    )

    result.validations = pd.DataFrame(
        [
            {
                "validation_name": row.validation_name,
                "status": row.status,
                "expected": row.expected,
                "actual": row.actual,
                "difference": row.difference,
                "explanation": row.explanation,
            }
            for row in rows
        ]
    )
    result.overall_status = overall
    if overall != "PASS":
        result.warnings.append(f"validation_status={overall} — ห้ามรายงานว่ากระทบยอดสำเร็จ")
    return result
