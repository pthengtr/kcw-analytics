"""Load Lazada Excel files without modifying the originals."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import FINANCE_KEEP, ORDERS_KEEP, PII_HEADER_HINTS, WALLET_KEEP
from .file_detector import DetectedFile, FileDetectionError, detect_excel_file, list_excel_files, normalize_header
from .normalizers import (
    MoneyParseError,
    decimal_sum,
    is_total_row,
    normalize_order_number,
    parse_money,
)

logger = logging.getLogger("lazada_reconciliation")


@dataclass
class LoadIssue:
    exception_type: str
    severity: str
    description: str
    source_file: str
    source_row: int | None = None
    order_number: str | None = None
    expected_amount: Any = None
    actual_amount: Any = None
    possible_reason: str | None = None


@dataclass
class LoadedTable:
    kind: str
    finance_profile: str | None
    frame: pd.DataFrame
    source_files: list[str]
    detected: list[DetectedFile]
    issues: list[LoadIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pre_group_amount_sum: Any = None
    row_count_before_total_filter: int = 0
    dropped_total_rows: int = 0


def _engine_for(path: Path) -> str:
    return "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"


def _is_pii_header(header: str) -> bool:
    key = normalize_header(header).replace(" ", "")
    return any(hint in key for hint in PII_HEADER_HINTS)


def _drop_empty_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = frame.shape[1]
    keep: list[Any] = []
    for column in frame.columns:
        label = str(column).strip()
        if label == "" or label.lower() in {"nan", "none", "unnamed"}:
            continue
        if label.lower().startswith("unnamed:"):
            series = frame[column]
            if series.isna().all() or (series.astype(str).str.strip() == "").all():
                continue
        keep.append(column)
    frame = frame.loc[:, keep]
    frame = frame.dropna(axis=1, how="all")
    return frame, before - frame.shape[1]


def _read_detected(detected: DetectedFile) -> pd.DataFrame:
    frame = pd.read_excel(
        detected.path,
        sheet_name=detected.sheet_name,
        header=detected.header_row_index,
        dtype=object,
        engine=_engine_for(detected.path),
    )
    frame, dropped = _drop_empty_columns(frame)
    detected.empty_columns_dropped = dropped
    # Excel row number of first data row = header_row_index + 2 (1-based).
    frame = frame.copy()
    frame["source_file"] = detected.path.name
    frame["source_row"] = frame.index.astype(int) + detected.header_row_index + 2
    return frame


def _rename_canonical(frame: pd.DataFrame, detected: DetectedFile) -> pd.DataFrame:
    rename: dict[str, str] = {}
    used: set[str] = set()
    for original in frame.columns:
        canonical = detected.canonical_map.get(str(original))
        if not canonical:
            # Try matching by normalized header in case pandas altered names.
            from .file_detector import map_headers

            mapped = map_headers([str(original)])
            canonical = mapped.get(str(original))
        if canonical and canonical not in used and canonical not in frame.columns:
            rename[original] = canonical
            used.add(canonical)
    return frame.rename(columns=rename)


def _strip_pii(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in frame.columns if not _is_pii_header(str(c))]
    dropped = [c for c in frame.columns if c not in keep]
    if dropped:
        logger.info("dropped_pii_columns count=%s", len(dropped))
    return frame.loc[:, keep]


def _apply_keep_list(frame: pd.DataFrame, keep: tuple[str, ...]) -> pd.DataFrame:
    columns = [c for c in keep if c in frame.columns]
    extra = [c for c in ("source_file", "source_row") if c in frame.columns and c not in columns]
    return frame.loc[:, columns + extra].copy()


def load_single_excel(path: Path, expected_kinds: set[str] | None = None) -> tuple[DetectedFile, pd.DataFrame]:
    detected = detect_excel_file(path)
    if expected_kinds and detected.file_type not in expected_kinds:
        raise FileDetectionError(
            (
                f"ไฟล์ {path.name} ถูกตรวจว่าเป็น {detected.file_type} "
                f"ซึ่งไม่ตรงกับโฟลเดอร์ที่คาดไว้ ({sorted(expected_kinds)})"
            ),
            {"file": path.name, "detected": detected.file_type, "expected": sorted(expected_kinds or [])},
        )
    frame = _read_detected(detected)
    frame = _rename_canonical(frame, detected)
    frame = _strip_pii(frame)
    return detected, frame


def _normalize_shared(frame: pd.DataFrame, issues: list[LoadIssue], kind: str) -> pd.DataFrame:
    working = frame.copy()
    if "order_number" in working.columns:
        working["order_number"] = working["order_number"].map(normalize_order_number)

    amount_col = "paid_price" if kind == "orders" else "amount"
    signed: list[Any] = []
    absolute: list[Any] = []
    for idx, row in working.iterrows():
        raw = row[amount_col] if amount_col in working.columns else None
        source_file = str(row.get("source_file") or "")
        source_row = int(row["source_row"]) if pd.notna(row.get("source_row")) else None
        order_number = row.get("order_number")
        try:
            parsed = parse_money(raw) if amount_col in working.columns else None
        except MoneyParseError:
            issues.append(
                LoadIssue(
                    exception_type="AMOUNT_PARSE_FAILED",
                    severity="FAIL",
                    description="แปลงจำนวนเงินไม่สำเร็จ",
                    source_file=source_file,
                    source_row=source_row,
                    order_number=order_number if isinstance(order_number, str) else None,
                    actual_amount=raw,
                    possible_reason="รูปแบบจำนวนเงินมีอักขระที่ไม่รองรับ",
                )
            )
            signed.append(None)
            absolute.append(None)
            continue
        signed.append(parsed)
        absolute.append(abs(parsed) if parsed is not None else None)
    if amount_col in working.columns:
        working["signed_amount"] = signed
        working["absolute_amount"] = absolute
        if kind == "orders":
            working["paid_price"] = signed
        else:
            working["amount"] = signed
    if "order_status" in working.columns:
        working["order_status"] = working["order_status"].map(
            lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).strip()
        )
    return working


def _filter_total_rows(frame: pd.DataFrame, issues: list[LoadIssue]) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame, 0
    mask = frame.apply(is_total_row, axis=1)
    dropped = int(mask.sum())
    if dropped:
        for _, row in frame.loc[mask].iterrows():
            issues.append(
                LoadIssue(
                    exception_type="GRAND_TOTAL_EXCLUDED",
                    severity="WARNING",
                    description="ตัดแถว Grand Total / Total ออก ไม่นับเป็นรายการ",
                    source_file=str(row.get("source_file") or ""),
                    source_row=int(row["source_row"]) if pd.notna(row.get("source_row")) else None,
                    possible_reason="แถวสรุปจากไฟล์ต้นทางต้องไม่ถูกนำมาคำนวณ",
                )
            )
        logger.info("excluded_total_rows count=%s", dropped)
    return frame.loc[~mask].copy(), dropped


def _flag_duplicates(frame: pd.DataFrame, issues: list[LoadIssue]) -> None:
    if frame.empty:
        return
    compare_cols = [c for c in frame.columns if c not in {"source_file", "source_row"}]
    duplicated = frame.duplicated(subset=compare_cols, keep=False)
    if not duplicated.any():
        return
    for _, row in frame.loc[duplicated].iterrows():
        issues.append(
            LoadIssue(
                exception_type="DUPLICATE_ROW",
                severity="WARNING",
                description="พบแถวที่เหมือนกันทุกคอลัมน์ (รวมข้ามไฟล์) — ไม่ได้ลบอัตโนมัติ",
                source_file=str(row.get("source_file") or ""),
                source_row=int(row["source_row"]) if pd.notna(row.get("source_row")) else None,
                order_number=row.get("order_number") if isinstance(row.get("order_number"), str) else None,
                actual_amount=row.get("signed_amount"),
                possible_reason="อาจส่งออกไฟล์ซ้ำหรือมีรายการซ้ำใน Seller Center",
            )
        )


def _flag_empty_order_numbers(frame: pd.DataFrame, issues: list[LoadIssue], kind: str) -> None:
    if "order_number" not in frame.columns:
        return
    empty = frame["order_number"].isna() | (frame["order_number"].astype(str).str.strip() == "")
    for _, row in frame.loc[empty].iterrows():
        issues.append(
            LoadIssue(
                exception_type="EMPTY_ORDER_NUMBER",
                severity="WARNING",
                description=f"orderNumber ว่างในไฟล์ {kind}",
                source_file=str(row.get("source_file") or ""),
                source_row=int(row["source_row"]) if pd.notna(row.get("source_row")) else None,
                actual_amount=row.get("signed_amount"),
                possible_reason="ข้อมูล orderNumber ว่างหรือรูปแบบไม่ตรงกัน",
            )
        )


def load_folder(folder: Path, expected_kinds: set[str], kind_label: str) -> LoadedTable:
    files = list_excel_files(folder)
    if not files:
        raise FileDetectionError(
            f"ไม่พบไฟล์ในโฟลเดอร์ {kind_label}: {folder}",
            {"folder": str(folder)},
        )
    if len(files) > 1:
        logger.info("multiple_files folder=%s count=%s names=%s", folder, len(files), [p.name for p in files])

    frames: list[pd.DataFrame] = []
    detected_list: list[DetectedFile] = []
    issues: list[LoadIssue] = []
    warnings: list[str] = []
    finance_profiles: set[str] = set()

    for path in files:
        detected, frame = load_single_excel(path, expected_kinds=expected_kinds)
        detected_list.append(detected)
        if detected.file_type.startswith("finance"):
            finance_profiles.add(detected.file_type)
        logger.info("loaded file=%s type=%s rows=%s cols=%s", path.name, detected.file_type, len(frame), frame.shape[1])
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    kind = "orders"
    if all(d.file_type.startswith("finance") for d in detected_list):
        kind = "finance"
    elif all(d.file_type == "wallet" for d in detected_list):
        kind = "wallet"

    combined = _normalize_shared(combined, issues, kind)
    before_total = len(combined)
    combined, dropped_totals = _filter_total_rows(combined, issues)
    _flag_duplicates(combined, issues)
    if kind != "wallet":
        _flag_empty_order_numbers(combined, issues, kind)

    if kind == "orders":
        combined = _apply_keep_list(combined, ORDERS_KEEP + ("signed_amount", "absolute_amount"))
        pre_group = decimal_sum(combined["signed_amount"])
    elif kind == "finance":
        combined = _apply_keep_list(combined, FINANCE_KEEP)
        pre_group = decimal_sum(combined["signed_amount"])
    else:
        combined = _apply_keep_list(combined, WALLET_KEEP)
        pre_group = decimal_sum(combined["signed_amount"])

    if len(files) > 1:
        warnings.append(
            f"พบหลายไฟล์ในโฟลเดอร์ {kind_label} จึงรวม {len(files)} ไฟล์: "
            + ", ".join(p.name for p in files)
        )

    finance_profile = None
    if kind == "finance":
        if len(finance_profiles) == 1:
            finance_profile = next(iter(finance_profiles))
        elif finance_profiles:
            finance_profile = "mixed"
            warnings.append(
                "พบไฟล์ Finance หลายโปรไฟล์ในโฟลเดอร์เดียวกัน: " + ", ".join(sorted(finance_profiles))
            )

    return LoadedTable(
        kind=kind,
        finance_profile=finance_profile,
        frame=combined,
        source_files=[p.name for p in files],
        detected=detected_list,
        issues=issues,
        warnings=warnings,
        pre_group_amount_sum=pre_group,
        row_count_before_total_filter=before_total,
        dropped_total_rows=dropped_totals,
    )


def load_orders(folder: Path) -> LoadedTable:
    return load_folder(Path(folder), {"orders"}, "คำสั่งซื้อ")


def load_finance(folder: Path) -> LoadedTable:
    return load_folder(Path(folder), {"finance_transaction", "finance_overview"}, "รายการทางบัญชี")


def load_wallet(folder: Path) -> LoadedTable:
    return load_folder(Path(folder), {"wallet"}, "ยอดของฉัน")
