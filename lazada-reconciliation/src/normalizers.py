"""Parse order numbers and money amounts without using float comparison."""

from __future__ import annotations

import logging
import math
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

import pandas as pd

from .config import (
    MONEY_QUANT,
    STATUS_GROUPS,
    STRONG_TOTAL_ROW_MARKERS,
    WEAK_TOTAL_ROW_MARKERS,
)

logger = logging.getLogger("lazada_reconciliation")


class MoneyParseError(ValueError):
    """Amount text could not be parsed."""


def quantize_money(value: Decimal) -> Decimal:
    return value.quantize(MONEY_QUANT, rounding=ROUND_HALF_UP)


def normalize_order_number(value: Any) -> str | None:
    """Keep Lazada order numbers as strings; strip Excel float artifacts only."""
    if value is None:
        return None
    if isinstance(value, float) and (pd.isna(value) or math.isnan(value)):
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        text = format(value, "f").rstrip("0").rstrip(".")
        return text or None
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return str(int(value))
        return str(value)

    text = str(value).replace("\ufeff", "").replace("\u00a0", " ").strip()
    if text == "" or text.casefold() in {"nan", "none", "null", "<na>"}:
        return None
    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def parse_money(value: Any) -> Decimal | None:
    """Parse a money value. Empty -> None (not zero). Raises MoneyParseError."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise MoneyParseError(repr(value))
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, Decimal):
        return quantize_money(value)
    if isinstance(value, int):
        return quantize_money(Decimal(value))
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        # Excel numeric cells arrive as float; quantize via from_float so we
        # never add binary float values together.
        return quantize_money(Decimal.from_float(value))

    text = str(value).replace("\ufeff", "").replace("\u00a0", " ").strip()
    if text == "" or text.casefold() in {"nan", "none", "null", "-", "–", "—"}:
        return None

    for token in ("฿", "thb", "THB", "usd", "USD", "บาท", "บ."):
        text = text.replace(token, "")
    text = text.replace(",", "").replace(" ", "")

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]
    if text.startswith("+"):
        text = text[1:]
    if text == "":
        return None
    try:
        amount = Decimal(text)
    except InvalidOperation as exc:
        raise MoneyParseError(str(value)) from exc
    if negative:
        amount = -amount
    return quantize_money(amount)


def as_money(value: Any) -> Decimal | None:
    """Coerce a cell to Decimal money. Empty/NaN -> None. Never returns float."""
    if value is None:
        return None
    try:
        if isinstance(value, float) and pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    if isinstance(value, Decimal):
        return quantize_money(value)
    try:
        return parse_money(value)
    except MoneyParseError:
        return None


def decimal_sum(values: Any) -> Decimal:
    total = Decimal("0.00")
    for value in values:
        parsed = value if isinstance(value, Decimal) else as_money(value)
        if parsed is None:
            continue
        total += parsed
    return quantize_money(total)


def is_float_money(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    if isinstance(value, float):
        return not pd.isna(value)
    return False


def status_group(status: str | None) -> str:
    if not status:
        return "other"
    return STATUS_GROUPS.get(str(status).strip().casefold(), "other")


def _marker_kind(value: Any) -> str | None:
    text = normalize_order_number(value)
    if not text:
        return None
    compact = re.sub(r"\s+", " ", text).strip().casefold()
    compact_nospace = compact.replace(" ", "")
    for marker in STRONG_TOTAL_ROW_MARKERS:
        if compact == marker or compact_nospace == marker.replace(" ", ""):
            return "strong"
    for marker in WEAK_TOTAL_ROW_MARKERS:
        if compact == marker or compact_nospace == marker.replace(" ", ""):
            return "weak"
    return None


def is_total_marker(value: Any) -> bool:
    return _marker_kind(value) is not None


def is_total_row(row: pd.Series) -> bool:
    """Drop Grand Total rows without treating a product named Total as a total."""
    kinds: list[str] = []
    identifier_weak = False
    order_number = None
    if "order_number" in row.index:
        order_number = normalize_order_number(row.get("order_number"))
        if _marker_kind(order_number) == "weak":
            identifier_weak = True
    for column in ("wallet_type", "wallet_sub_type", "transaction_number", "freename"):
        if column in row.index and _marker_kind(row.get(column)) == "weak":
            identifier_weak = True
    for value in row.tolist():
        kind = _marker_kind(value)
        if kind:
            kinds.append(kind)
    if "strong" in kinds:
        return True
    if identifier_weak:
        return True
    if not order_number and "weak" in kinds:
        return True
    return False
