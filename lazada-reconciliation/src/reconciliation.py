"""Order / Finance / Wallet reconciliation (full outer join, Decimal math)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import pandas as pd

from .config import (
    FEE_CATEGORIES,
    FREENAME_CONTAINS,
    FREENAME_EXACT,
    INSUFFICIENT_DATA,
    POSSIBLE_REASONS,
    WALLET_EXACT,
    WALLET_TYPE_CONTAINS,
)
from .loaders import LoadIssue, LoadedTable
from .normalizers import as_money, decimal_sum, quantize_money, status_group

logger = logging.getLogger("lazada_reconciliation")

ZERO = Decimal("0.00")


@dataclass
class ReconciliationResult:
    month: str
    generated_at: str
    tolerance: Decimal
    finance_profile: str | None
    source_files: dict[str, list[str]]
    order_summary: pd.DataFrame
    finance_summary: pd.DataFrame
    finance_details: pd.DataFrame
    order_vs_finance: pd.DataFrame
    wallet_details: pd.DataFrame
    wallet_totals: dict[str, Any]
    expense_summary: dict[str, Any]
    exceptions: pd.DataFrame
    unknown_mappings: pd.DataFrame
    validations: pd.DataFrame
    overall_status: str
    warnings: list[str]
    methodology_notes: list[str]
    match_counts: dict[str, int]
    pre_group_totals: dict[str, Decimal]
    post_group_totals: dict[str, Decimal]
    issues: list[LoadIssue] = field(default_factory=list)


def _norm_key(value: Any) -> str:
    return str(value or "").strip().casefold()


def map_freename(freename: str | None) -> tuple[str, bool]:
    """Return (category, known). known=False means caller should list in Unknown_Mappings."""
    if freename is None or str(freename).strip() == "":
        return "unknown", False
    key = _norm_key(freename)
    if key in FREENAME_EXACT:
        return FREENAME_EXACT[key], True
    for needle, category in FREENAME_CONTAINS:
        if needle in key:
            return category, True
    return "unknown", False


def map_wallet(wallet_type: str | None, sub_type: str | None) -> tuple[str, bool]:
    type_key = _norm_key(wallet_type)
    sub_key = _norm_key(sub_type)
    exact = WALLET_EXACT.get((type_key, sub_key))
    if exact:
        return exact, True
    combined = f"{type_key} {sub_key}".strip()
    for needle, category in WALLET_TYPE_CONTAINS:
        if needle in combined:
            return category, True
    return "unknown", False


def classify_finance_rows(frame: pd.DataFrame, finance_profile: str | None) -> pd.DataFrame:
    details = frame.copy()
    if details.empty:
        details["mapped_category"] = pd.Series(dtype=object)
        details["mapping_known"] = pd.Series(dtype=bool)
        details["mapping_key"] = pd.Series(dtype=object)
        details["finance_profile"] = pd.Series(dtype=object)
        return details

    categories: list[str] = []
    known_flags: list[bool] = []
    keys: list[str] = []
    profile = finance_profile or "finance_overview"

    for _, row in details.iterrows():
        freename = row["freename"] if "freename" in details.columns else None
        if freename is not None and not (isinstance(freename, float) and pd.isna(freename)):
            text = str(freename).strip()
        else:
            text = ""
        amount = row.get("signed_amount")
        order_number = row.get("order_number")
        has_order = isinstance(order_number, str) and order_number != ""

        if text:
            category, known = map_freename(text)
            keys.append(text)
            categories.append(category)
            known_flags.append(known)
            continue

        # Income Order Overview has no Freename. Classify from sign, never hide.
        if profile == "finance_transaction":
            categories.append("unknown")
            known_flags.append(False)
            keys.append("(blank Freename)")
            continue

        if not has_order:
            categories.append("unknown")
            known_flags.append(False)
            keys.append("(blank orderNumber, blank Freename)")
            continue
        if amount is None:
            categories.append("unknown")
            known_flags.append(False)
            keys.append("(blank amount, Income Order Overview)")
            continue
        if amount < ZERO:
            categories.append("refund")
            known_flags.append(True)
            keys.append("[overview:negative_with_order]")
        else:
            categories.append("order_income")
            known_flags.append(True)
            keys.append("[overview:positive_net]")

    details["mapped_category"] = categories
    details["mapping_known"] = known_flags
    details["mapping_key"] = keys
    details["finance_profile"] = profile
    if "freename" not in details.columns:
        details["freename"] = details["mapping_key"]
    else:
        details["freename"] = details["freename"].where(
            details["freename"].notna() & (details["freename"].astype(str).str.strip() != ""),
            details["mapping_key"],
        )
    return details


def classify_wallet_rows(frame: pd.DataFrame) -> pd.DataFrame:
    details = frame.copy()
    if details.empty:
        details["mapped_category"] = pd.Series(dtype=object)
        details["mapping_known"] = pd.Series(dtype=bool)
        details["is_auto_withdrawal"] = pd.Series(dtype=bool)
        details["is_bank_transfer"] = pd.Series(dtype=bool)
        return details

    categories: list[str] = []
    known_flags: list[bool] = []
    auto_flags: list[bool] = []
    bank_flags: list[bool] = []
    for _, row in details.iterrows():
        wallet_type = row.get("wallet_type")
        sub_type = row.get("wallet_sub_type")
        category, known = map_wallet(wallet_type, sub_type)
        categories.append(category)
        known_flags.append(known)
        is_auto = _norm_key(sub_type) == "auto withdrawal"
        auto_flags.append(is_auto)
        bank_flags.append(category == "bank_withdrawal" or is_auto)
    details["mapped_category"] = categories
    details["mapping_known"] = known_flags
    details["is_auto_withdrawal"] = auto_flags
    details["is_bank_transfer"] = bank_flags
    return details


def summarize_orders(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "order_number",
                "gross_order_amount",
                "order_status",
                "all_statuses",
                "mixed_status",
                "status_group",
                "source_line_count",
                "first_source_row",
                "last_source_row",
                "source_file",
            ]
        )
    usable = frame[frame["order_number"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for order_number, group in usable.groupby("order_number", dropna=False, sort=False):
        statuses = [s for s in group["order_status"].tolist() if s]
        unique_statuses = list(dict.fromkeys(statuses))
        mixed = len(unique_statuses) > 1
        source_files = list(dict.fromkeys(group["source_file"].astype(str).tolist()))
        rows.append(
            {
                "order_number": order_number,
                "gross_order_amount": decimal_sum(group["signed_amount"]),
                "order_status": unique_statuses[0] if len(unique_statuses) == 1 else "MIXED",
                "all_statuses": " | ".join(unique_statuses),
                "mixed_status": mixed,
                "status_group": (
                    "mixed" if mixed else status_group(unique_statuses[0] if unique_statuses else None)
                ),
                "source_line_count": int(len(group)),
                "first_source_row": int(group["source_row"].min()),
                "last_source_row": int(group["source_row"].max()),
                "source_file": "; ".join(source_files),
            }
        )
    return pd.DataFrame(rows)


def summarize_finance(details: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "order_number",
        "finance_income",
        "shipping_fee",
        "service_fee",
        "other_fee",
        "refund",
        "adjustment",
        "unknown_amount",
        "finance_net_amount",
        "source_line_count",
        "first_source_row",
        "last_source_row",
        "source_file",
        "has_unknown_mapping",
    ]
    if details.empty:
        return pd.DataFrame(columns=columns)

    usable = details[details["order_number"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for order_number, group in usable.groupby("order_number", dropna=False, sort=False):
        bucket = {name: ZERO for name in FEE_CATEGORIES}

        def _sum_cat(name: str) -> Decimal:
            return decimal_sum(group.loc[group["mapped_category"] == name, "signed_amount"])

        for name in FEE_CATEGORIES:
            bucket[name] = _sum_cat(name)
        source_files = list(dict.fromkeys(group["source_file"].astype(str).tolist()))
        rows.append(
            {
                "order_number": order_number,
                "finance_income": bucket["order_income"],
                "shipping_fee": bucket["shipping_fee"],
                "service_fee": bucket["service_fee"],
                "other_fee": bucket["other_fee"],
                "refund": bucket["refund"],
                "adjustment": bucket["adjustment"],
                "unknown_amount": bucket["unknown"],
                "finance_net_amount": decimal_sum(group["signed_amount"]),
                "source_line_count": int(len(group)),
                "first_source_row": int(group["source_row"].min()),
                "last_source_row": int(group["source_row"].max()),
                "source_file": "; ".join(source_files),
                "has_unknown_mapping": bool((~group["mapping_known"]).any()),
            }
        )
    return pd.DataFrame(rows)


def _reason_for(status: str) -> str:
    return POSSIBLE_REASONS.get(status, POSSIBLE_REASONS["UNKNOWN"])


def match_orders_to_finance(
    order_summary: pd.DataFrame,
    finance_summary: pd.DataFrame,
    tolerance: Decimal,
    finance_profile: str | None,
) -> pd.DataFrame:
    left = order_summary.copy()
    right = finance_summary.copy()
    if "order_number" not in left.columns:
        left["order_number"] = pd.Series(dtype=object)
    if "order_number" not in right.columns:
        right["order_number"] = pd.Series(dtype=object)
    merged = left.merge(right, on="order_number", how="outer", suffixes=("_order", "_finance"), indicator=True)
    rows: list[dict[str, Any]] = []
    is_overview = finance_profile == "finance_overview"
    for _, row in merged.iterrows():
        in_orders = row["_merge"] in {"both", "left_only"}
        in_finance = row["_merge"] in {"both", "right_only"}
        gross = as_money(row.get("gross_order_amount")) if in_orders else None
        income = as_money(row.get("finance_income")) if in_finance else None
        net = as_money(row.get("finance_net_amount")) if in_finance else None
        difference = None
        if gross is not None and income is not None:
            difference = quantize_money(gross - income)
        elif gross is not None:
            difference = gross
        elif income is not None:
            difference = quantize_money(-income)

        status_group_value = row.get("status_group") if in_orders else None
        cancelled = status_group_value in {"cancelled", "refund"}
        match_status = "UNKNOWN"
        if in_orders and in_finance:
            if difference is not None and abs(difference) <= tolerance:
                match_status = "MATCHED"
            elif cancelled:
                match_status = "CANCELLED_OR_REFUNDED"
            elif is_overview:
                match_status = "NET_SETTLED"
            else:
                match_status = "AMOUNT_MISMATCH"
        elif in_orders and not in_finance:
            match_status = "CANCELLED_OR_REFUNDED" if cancelled else "ORDER_NOT_RELEASED"
        elif in_finance and not in_orders:
            match_status = "FINANCE_FROM_OTHER_PERIOD"

        implied_gap = difference
        rows.append(
            {
                "order_number": row.get("order_number"),
                "gross_order_amount": gross,
                "finance_income": income,
                "finance_net_amount": net,
                "order_to_finance_difference": difference,
                "implied_fee_or_net_gap": implied_gap,
                "shipping_fee": row.get("shipping_fee") if in_finance else None,
                "service_fee": row.get("service_fee") if in_finance else None,
                "other_fee": row.get("other_fee") if in_finance else None,
                "refund": row.get("refund") if in_finance else None,
                "adjustment": row.get("adjustment") if in_finance else None,
                "unknown_amount": row.get("unknown_amount") if in_finance else None,
                "order_status": row.get("order_status") if in_orders else None,
                "all_statuses": row.get("all_statuses") if in_orders else None,
                "mixed_status": bool(row.get("mixed_status")) if in_orders else False,
                "status_group": status_group_value,
                "match_status": match_status,
                "possible_reason": _reason_for(match_status),
                "in_orders": in_orders,
                "in_finance": in_finance,
                "order_source_file": row.get("source_file_order") if in_orders else None,
                "order_source_row_first": row.get("first_source_row_order") if in_orders else None,
                "order_source_row_last": row.get("last_source_row_order") if in_orders else None,
                "finance_source_file": row.get("source_file_finance") if in_finance else None,
                "finance_source_row_first": row.get("first_source_row_finance") if in_finance else None,
                "finance_source_row_last": row.get("last_source_row_finance") if in_finance else None,
                "order_line_count": row.get("source_line_count_order") if in_orders else 0,
                "finance_line_count": row.get("source_line_count_finance") if in_finance else 0,
            }
        )
    return pd.DataFrame(rows)


def expense_totals(details: pd.DataFrame) -> dict[str, Any]:
    totals = {name: ZERO for name in FEE_CATEGORIES}
    if not details.empty:
        for name in FEE_CATEGORIES:
            totals[name] = decimal_sum(details.loc[details["mapped_category"] == name, "signed_amount"])
    net = decimal_sum(details["signed_amount"]) if not details.empty else ZERO
    component_sum = quantize_money(sum((totals[name] for name in FEE_CATEGORIES), ZERO))
    return {
        "order_income": totals["order_income"],
        "shipping_fee": totals["shipping_fee"],
        "service_fee": totals["service_fee"],
        "other_fee": totals["other_fee"],
        "refund": totals["refund"],
        "adjustment": totals["adjustment"],
        "unknown_amount": totals["unknown"],
        "finance_net_amount": net,
        "component_sum": component_sum,
        "order_income_abs": abs(totals["order_income"]),
        "shipping_fee_abs": abs(totals["shipping_fee"]),
        "service_fee_abs": abs(totals["service_fee"]),
        "other_fee_abs": abs(totals["other_fee"]),
        "refund_abs": abs(totals["refund"]),
        "adjustment_abs": abs(totals["adjustment"]),
        "unknown_abs": abs(totals["unknown"]),
    }


def wallet_totals(details: pd.DataFrame) -> dict[str, Any]:
    empty = {
        "opening_balance": None,
        "opening_balance_found": False,
        "reported_closing_balance": None,
        "reported_closing_found": False,
        "wallet_inflows": ZERO,
        "wallet_adjustments": ZERO,
        "withdrawals_signed": ZERO,
        "withdrawals_as_positive": ZERO,
        "auto_withdrawal_signed": ZERO,
        "auto_withdrawal_positive": ZERO,
        "bank_withdrawals": ZERO,
        "unknown_amount": ZERO,
        "signed_net": ZERO,
        "movement_net": ZERO,
        "calculated_closing_balance": INSUFFICIENT_DATA,
        "wallet_difference": INSUFFICIENT_DATA,
        "formula": (
            "รายการถอนแสดงเป็นค่าบวกเพื่ออ่านง่ายเท่านั้น "
            "calculated_closing_balance = opening_balance + movement_net "
            "โดย movement_net = sum(signed_amount) ของแถวที่ไม่ใช่ opening/closing "
            "ห้ามใช้ opening + signed_net เพราะ signed_net นับยอดยกมาอยู่แล้ว"
        ),
    }
    if details.empty:
        return empty

    def _sum(mask: pd.Series) -> Decimal:
        return decimal_sum(details.loc[mask, "signed_amount"])

    opening_mask = details["mapped_category"] == "opening_balance"
    closing_mask = details["mapped_category"] == "closing_balance"
    inflow_mask = details["mapped_category"] == "settlement_inflow"
    adj_mask = details["mapped_category"] == "adjustment"
    wd_mask = details["mapped_category"] == "bank_withdrawal"
    auto_mask = details["is_auto_withdrawal"] if "is_auto_withdrawal" in details.columns else wd_mask
    unknown_mask = details["mapped_category"] == "unknown"

    opening = _sum(opening_mask) if opening_mask.any() else None
    reported_closing = _sum(closing_mask) if closing_mask.any() else None
    inflows = _sum(inflow_mask)
    adjustments = _sum(adj_mask)
    withdrawals_signed = _sum(wd_mask)
    auto_signed = _sum(auto_mask)
    unknown_amount = _sum(unknown_mask)
    signed_net = decimal_sum(details["signed_amount"])
    movement_mask = ~opening_mask & ~closing_mask
    movement_net = _sum(movement_mask) if movement_mask.any() else ZERO
    withdrawals_positive = quantize_money(-withdrawals_signed) if withdrawals_signed < ZERO else withdrawals_signed
    auto_positive = quantize_money(-auto_signed) if auto_signed < ZERO else auto_signed

    calculated: Any = INSUFFICIENT_DATA
    difference: Any = INSUFFICIENT_DATA
    if opening is not None:
        # Opening is a signed row in the file. Do not add it twice via signed_net.
        # Presentation may show withdrawals as positive; calculation uses signed movement.
        calculated = quantize_money(opening + movement_net)
        if reported_closing is not None:
            difference = quantize_money(calculated - reported_closing)

    return {
        "opening_balance": opening if opening is not None else INSUFFICIENT_DATA,
        "opening_balance_found": opening is not None,
        "reported_closing_balance": reported_closing if reported_closing is not None else INSUFFICIENT_DATA,
        "reported_closing_found": reported_closing is not None,
        "wallet_inflows": inflows,
        "wallet_adjustments": adjustments,
        "withdrawals_signed": withdrawals_signed,
        "withdrawals_as_positive": withdrawals_positive,
        "auto_withdrawal_signed": auto_signed,
        "auto_withdrawal_positive": auto_positive,
        "bank_withdrawals": withdrawals_positive,
        "unknown_amount": unknown_amount,
        "signed_net": signed_net,
        "movement_net": movement_net,
        "calculated_closing_balance": calculated,
        "wallet_difference": difference,
        "formula": empty["formula"],
    }


def unknown_mapping_table(finance_details: pd.DataFrame, wallet_details: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not finance_details.empty:
        unknown = finance_details.loc[~finance_details["mapping_known"]]
        if not unknown.empty:
            grouped = unknown.groupby("mapping_key", dropna=False)
            for key, group in grouped:
                rows.append(
                    {
                        "source": "finance",
                        "raw_value": key,
                        "field": "Freename / ชื่อรายการธุรกรรม",
                        "row_count": int(len(group)),
                        "signed_amount_sum": decimal_sum(group["signed_amount"]),
                        "suggested_mapping": (
                            "เพิ่มใน FREENAME_EXACT ของ src/config.py "
                            "เป็น order_income / shipping_fee / service_fee / other_fee / refund / adjustment"
                        ),
                    }
                )
    if not wallet_details.empty:
        unknown = wallet_details.loc[~wallet_details["mapping_known"]]
        if not unknown.empty:
            grouped = unknown.groupby(["wallet_type", "wallet_sub_type"], dropna=False)
            for (wtype, subtype), group in grouped:
                rows.append(
                    {
                        "source": "wallet",
                        "raw_value": f"Type={wtype} | Sub Type={subtype}",
                        "field": "Type / Sub Type",
                        "row_count": int(len(group)),
                        "signed_amount_sum": decimal_sum(group["signed_amount"]),
                        "suggested_mapping": (
                            "เพิ่มใน WALLET_EXACT ของ src/config.py "
                            "เป็น settlement_inflow / bank_withdrawal / opening_balance / closing_balance / adjustment"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _exception_rows_from_matches(matches: pd.DataFrame) -> list[LoadIssue]:
    issues: list[LoadIssue] = []
    if matches.empty:
        return issues
    for _, row in matches.iterrows():
        status = row.get("match_status")
        if status in {"MATCHED"}:
            continue
        severity = "INFO"
        if status in {"AMOUNT_MISMATCH"}:
            severity = "WARNING"
        elif status in {"ORDER_NOT_RELEASED", "FINANCE_FROM_OTHER_PERIOD", "NET_SETTLED"}:
            severity = "WARNING"
        elif status in {"CANCELLED_OR_REFUNDED"}:
            severity = "INFO"
        elif status == "UNKNOWN":
            severity = "WARNING"
        issues.append(
            LoadIssue(
                exception_type=str(status),
                severity=severity,
                description=f"match_status={status}",
                source_file=str(row.get("order_source_file") or row.get("finance_source_file") or ""),
                source_row=(
                    int(row["order_source_row_first"])
                    if pd.notna(row.get("order_source_row_first"))
                    else (
                        int(row["finance_source_row_first"])
                        if pd.notna(row.get("finance_source_row_first"))
                        else None
                    )
                ),
                order_number=row.get("order_number"),
                expected_amount=row.get("gross_order_amount"),
                actual_amount=row.get("finance_income"),
                possible_reason=row.get("possible_reason"),
            )
        )
    return issues


def issues_to_frame(issues: list[LoadIssue]) -> pd.DataFrame:
    if not issues:
        return pd.DataFrame(
            columns=[
                "exception_type",
                "severity",
                "orderNumber",
                "description",
                "expected_amount",
                "actual_amount",
                "difference",
                "possible_reason",
                "source_file",
                "source_row",
            ]
        )
    rows = []
    for issue in issues:
        expected = issue.expected_amount
        actual = issue.actual_amount
        difference = None
        if isinstance(expected, Decimal) and isinstance(actual, Decimal):
            difference = quantize_money(expected - actual)
        rows.append(
            {
                "exception_type": issue.exception_type,
                "severity": issue.severity,
                "orderNumber": issue.order_number,
                "description": issue.description,
                "expected_amount": expected,
                "actual_amount": actual,
                "difference": difference,
                "possible_reason": issue.possible_reason,
                "source_file": issue.source_file,
                "source_row": issue.source_row,
            }
        )
    return pd.DataFrame(rows)


def run_reconciliation(
    orders: LoadedTable,
    finance: LoadedTable,
    wallet: LoadedTable,
    *,
    month: str,
    generated_at: str,
    tolerance: Decimal,
) -> ReconciliationResult:
    finance_profile = finance.finance_profile
    finance_details = classify_finance_rows(finance.frame, finance_profile)
    wallet_details = classify_wallet_rows(wallet.frame)
    order_summary = summarize_orders(orders.frame)
    finance_summary = summarize_finance(finance_details)
    matches = match_orders_to_finance(order_summary, finance_summary, tolerance, finance_profile)
    expenses = expense_totals(finance_details)
    wallet_summary = wallet_totals(wallet_details)
    unknowns = unknown_mapping_table(finance_details, wallet_details)

    issues = list(orders.issues) + list(finance.issues) + list(wallet.issues)
    issues.extend(_exception_rows_from_matches(matches))
    warnings = list(orders.warnings) + list(finance.warnings) + list(wallet.warnings)

    match_counts = (
        matches["match_status"].value_counts().to_dict() if not matches.empty else {}
    )
    match_counts = {str(k): int(v) for k, v in match_counts.items()}

    post_group_orders = decimal_sum(order_summary["gross_order_amount"]) if not order_summary.empty else ZERO
    orders_without_number = ZERO
    if not orders.frame.empty and "order_number" in orders.frame.columns:
        orders_without_number = decimal_sum(
            orders.frame.loc[orders.frame["order_number"].isna(), "signed_amount"]
        )
    post_group_finance = (
        decimal_sum(finance_summary["finance_net_amount"]) if not finance_summary.empty else ZERO
    )
    finance_no_order = ZERO
    if not finance_details.empty:
        finance_no_order = decimal_sum(
            finance_details.loc[finance_details["order_number"].isna(), "signed_amount"]
        )
    post_group_wallet = decimal_sum(wallet_details["signed_amount"]) if not wallet_details.empty else ZERO

    methodology = [
        "order_to_finance_difference = gross_order_amount - finance_income",
        "gross_order_amount = sum(paidPrice) ต่อ orderNumber จากไฟล์คำสั่งซื้อ",
        "finance_net_amount = sum(signed_amount) ของทุกรายการ Finance ของออเดอร์นั้น ไม่ใช่ผลรวมค่าสัมบูรณ์",
        "เมื่อไฟล์ Finance เป็น Income Order Overview ยอดเป็นยอดสุทธิ จึงใช้ match_status=NET_SETTLED หากผลต่างเกิน tolerance",
        "Wallet: ใช้คอลัมน์ Amount เท่านั้น ไม่ใช้ Sub Type เป็นจำนวนเงิน",
        "รายการถอน Auto Withdrawal ใช้ signed amount ตามไฟล์ ไม่พลิกเครื่องหมายก่อนคำนวณ",
        "calculated_closing_balance = opening_balance + movement_net (ไม่นับ opening/closing ซ้ำใน movement)",
        f"tolerance = {tolerance} บาท",
        "โปรแกรมนี้ช่วยกระทบยอด ไม่ใช่คำรับรองทางบัญชี",
    ]

    result = ReconciliationResult(
        month=month,
        generated_at=generated_at,
        tolerance=tolerance,
        finance_profile=finance_profile,
        source_files={
            "orders": orders.source_files,
            "finance": finance.source_files,
            "wallet": wallet.source_files,
        },
        order_summary=order_summary,
        finance_summary=finance_summary,
        finance_details=finance_details,
        order_vs_finance=matches,
        wallet_details=wallet_details,
        wallet_totals=wallet_summary,
        expense_summary=expenses,
        exceptions=issues_to_frame(issues),
        unknown_mappings=unknowns,
        validations=pd.DataFrame(),
        overall_status="WARNING",
        warnings=warnings,
        methodology_notes=methodology,
        match_counts=match_counts,
        pre_group_totals={
            "orders": orders.pre_group_amount_sum or ZERO,
            "finance": finance.pre_group_amount_sum or ZERO,
            "wallet": wallet.pre_group_amount_sum or ZERO,
        },
        post_group_totals={
            "orders": quantize_money(post_group_orders + orders_without_number),
            "finance": quantize_money(post_group_finance + finance_no_order),
            "wallet": post_group_wallet,
        },
        issues=issues,
    )
    logger.info(
        "reconciled orders=%s finance_orders=%s matches=%s net_status_counts=%s",
        len(order_summary),
        len(finance_summary),
        len(matches),
        match_counts,
    )
    return result
