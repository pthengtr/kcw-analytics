"""Detect Lazada export type from headers, not from file names."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    FILE_TYPE_LABELS,
    FILE_TYPE_REQUIRED,
    HEADER_ALIASES,
    HEADER_SCAN_ROWS,
)

logger = logging.getLogger("lazada_reconciliation")

EXCEL_SUFFIXES = {".xlsx", ".xls"}


class FileDetectionError(Exception):
    """Raised when a file cannot be classified from its headers."""

    def __init__(self, message_th: str, details: dict[str, Any] | None = None):
        super().__init__(message_th)
        self.message_th = message_th
        self.details = details or {}


@dataclass
class DetectedFile:
    path: Path
    file_type: str
    sheet_name: str
    header_row_index: int
    headers: list[str]
    canonical_map: dict[str, str]
    empty_columns_dropped: int = 0
    warnings: list[str] = field(default_factory=list)


def normalize_header(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).replace("\ufeff", "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip().strip(":").strip()
    return text.casefold()


def _alias_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for canonical, aliases in HEADER_ALIASES.items():
        lookup[canonical.casefold()] = canonical
        for alias in aliases:
            lookup[normalize_header(alias)] = canonical
    return lookup


_ALIAS_LOOKUP = _alias_lookup()


def map_headers(headers: list[str]) -> dict[str, str]:
    """Return {original_header: canonical_field} for recognized columns."""
    mapped: dict[str, str] = {}
    used_canonical: set[str] = set()
    for header in headers:
        key = normalize_header(header)
        if not key:
            continue
        canonical = _ALIAS_LOOKUP.get(key)
        if canonical and canonical not in used_canonical:
            mapped[header] = canonical
            used_canonical.add(canonical)
    return mapped


def matched_canonicals(headers: list[str]) -> set[str]:
    return set(map_headers(headers).values())


def classify_headers(headers: list[str]) -> str | None:
    fields = matched_canonicals(headers)
    # More specific types first.
    if all(req in fields for req in FILE_TYPE_REQUIRED["wallet"]):
        return "wallet"
    if all(req in fields for req in FILE_TYPE_REQUIRED["finance_transaction"]):
        return "finance_transaction"
    if all(req in fields for req in FILE_TYPE_REQUIRED["finance_overview"]):
        return "finance_overview"
    if all(req in fields for req in FILE_TYPE_REQUIRED["orders"]):
        return "orders"
    return None


def _list_sheets(path: Path) -> list[str]:
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    xl = pd.ExcelFile(path, engine=engine)
    try:
        return list(xl.sheet_names)
    finally:
        xl.close()


def _read_preview(path: Path, sheet_name: str) -> pd.DataFrame:
    engine = "xlrd" if path.suffix.lower() == ".xls" else "openpyxl"
    return pd.read_excel(
        path,
        sheet_name=sheet_name,
        header=None,
        nrows=HEADER_SCAN_ROWS,
        dtype=object,
        engine=engine,
    )


def _row_headers(row: pd.Series) -> list[str]:
    values: list[str] = []
    for value in row.tolist():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        text = str(value).replace("\ufeff", "").strip()
        if text and text.lower() not in {"nan", "none"}:
            values.append(text)
    return values


def scan_sheet_for_header(path: Path, sheet_name: str) -> DetectedFile | None:
    preview = _read_preview(path, sheet_name)
    if preview.empty:
        return None
    best: tuple[int, list[str], str] | None = None
    for idx, row in preview.iterrows():
        headers = _row_headers(row)
        file_type = classify_headers(headers)
        if file_type:
            best = (int(idx), headers, file_type)
            break
    if best is None:
        return None
    header_row_index, headers, file_type = best
    canonical_map = map_headers(headers)
    return DetectedFile(
        path=path,
        file_type=file_type,
        sheet_name=sheet_name,
        header_row_index=header_row_index,
        headers=headers,
        canonical_map=canonical_map,
    )


def detect_excel_file(path: Path) -> DetectedFile:
    path = Path(path)
    if path.suffix.lower() not in EXCEL_SUFFIXES:
        raise FileDetectionError(
            f"ไฟล์ {path.name} ไม่ใช่ Excel (.xlsx/.xls)",
            {"file": path.name, "suffix": path.suffix},
        )
    try:
        sheets = _list_sheets(path)
    except Exception as exc:  # noqa: BLE001 — surface as Thai user error
        raise FileDetectionError(
            f"อ่านไฟล์ไม่ได้: {path.name} ({exc.__class__.__name__})",
            {"file": path.name, "error": str(exc)},
        ) from exc

    last_headers: list[str] = []
    for sheet_name in sheets:
        try:
            detected = scan_sheet_for_header(path, sheet_name)
        except Exception as exc:  # noqa: BLE001
            raise FileDetectionError(
                f"อ่านไฟล์ไม่ได้: {path.name} ชีต {sheet_name}",
                {"file": path.name, "sheet": sheet_name, "error": str(exc)},
            ) from exc
        if detected is not None:
            logger.info(
                "detected file=%s type=%s sheet=%s header_row=%s",
                path.name,
                detected.file_type,
                detected.sheet_name,
                detected.header_row_index + 1,
            )
            return detected
        preview = _read_preview(path, sheet_name)
        if not preview.empty:
            last_headers = _row_headers(preview.iloc[0])

    wanted = []
    for file_type, required in FILE_TYPE_REQUIRED.items():
        aliases = []
        for field in required:
            aliases.append(f"{field}={list(HEADER_ALIASES[field][:4])}")
        wanted.append(f"{FILE_TYPE_LABELS[file_type]} ต้องการ {', '.join(aliases)}")
    raise FileDetectionError(
        (
            f"ไม่สามารถระบุประเภทไฟล์ได้: {path.name}\n"
            f"header ที่ตรวจพบ: {last_headers or '(ว่าง)'}\n"
            f"header ที่ต้องการอย่างน้อยหนึ่งชุด:\n- "
            + "\n- ".join(wanted)
            + "\nวิธีแก้ไข: ดาวน์โหลดไฟล์จาก Lazada Seller Center โดยไม่ลบแถวหัวตาราง "
            "และไม่แปลงเป็น CSV ก่อนนำเข้า"
        ),
        {
            "file": path.name,
            "found_headers": last_headers,
            "wanted": wanted,
        },
    )


def list_excel_files(folder: Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileDetectionError(
            f"ไม่พบโฟลเดอร์: {folder}",
            {"folder": str(folder)},
        )
    files = sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXCEL_SUFFIXES and not p.name.startswith("~$")
    )
    return files
