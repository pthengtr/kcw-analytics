"""Bank statement Excel parsing helpers (account metadata + transaction fingerprints)."""

from __future__ import annotations

import hashlib
import os
import re
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd

ACCOUNT_METADATA_LABELS = frozenset(
    {
        "ACCOUNT NO.",
        "ACCOUNT NO",
        "ACCOUNT NUMBER",
        "เลขที่บัญชี",
        "เลขที่บัญชีเงินฝาก",
    }
)

ACCOUNT_METADATA_LABEL_PREFIXES = (
    "ACCOUNT NO",
    "ACCOUNT NUMBER",
    "เลขที่บัญชี",
)

STATEMENT_EXTENSIONS = {".xlsx", ".xls", ".xlsm"}


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def norm_text(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x)
    s = s.replace("\u00A0", " ")
    s = s.strip().upper()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_date(d) -> str:
    if d is None or pd.isna(d):
        return ""
    ts = pd.to_datetime(d, errors="coerce", dayfirst=True)
    if pd.isna(ts):
        return ""
    return ts.date().isoformat()


def norm_money(x) -> str:
    if x is None or pd.isna(x):
        return ""
    try:
        d = Decimal(str(x).replace(",", "").strip())
    except Exception:
        d = Decimal(str(pd.to_numeric(x, errors="coerce")))
    if d.is_nan():
        return ""
    d = d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return f"{d:.2f}"


def detect_bank_from_path(path: str) -> str | None:
    up = path.upper()
    if "\\KBANK\\" in up or up.endswith("KBANK"):
        return "KBANK"
    if "\\KTB\\" in up or up.endswith("KTB"):
        return "KTB"
    folder = os.path.basename(os.path.dirname(path)).upper()
    if folder in {"KBANK", "KTB"}:
        return folder
    return None


def infer_account_from_filename(path: str, bank_name: str | None) -> str:
    base = os.path.splitext(os.path.basename(path))[0].upper()
    if bank_name and base.startswith(bank_name.upper()):
        rest = base[len(bank_name) :]
        m = re.match(r"(\d+)", rest)
        if m:
            return m.group(1)
    m = re.search(r"(\d{3,})", base)
    return m.group(1) if m else ""


def open_excel_file(path: str) -> pd.ExcelFile:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xlsm", ".xls"}:
        try:
            return pd.ExcelFile(path, engine="openpyxl")
        except Exception:
            if ext == ".xls":
                return pd.ExcelFile(path, engine="xlrd")
    return pd.ExcelFile(path)


def _is_account_metadata_label(label: str) -> bool:
    if not label:
        return False
    if label in ACCOUNT_METADATA_LABELS:
        return True
    return any(label.startswith(prefix) for prefix in ACCOUNT_METADATA_LABEL_PREFIXES)


def _split_label_value_cell(cell) -> tuple[str, str]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return "", ""
    s = str(cell).strip()
    for sep in (":", "："):
        if sep in s:
            left, right = s.split(sep, 1)
            return norm_text(left), right.strip()
    return norm_text(s), ""


def _extract_account_from_metadata(df0: pd.DataFrame) -> str:
    for i in range(min(len(df0), 20)):
        row = df0.iloc[i].tolist()
        for j, cell in enumerate(row):
            sub_label, sub_val = _split_label_value_cell(cell)
            if sub_val and _is_account_metadata_label(sub_label):
                return sub_val

            label = norm_text(cell)
            if _is_account_metadata_label(label):
                for k in range(j + 1, len(row)):
                    val = row[k]
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        continue
                    s = str(val).strip()
                    if s:
                        return s
    return ""


def _find_header_row(df0: pd.DataFrame) -> int | None:
    for i in range(min(len(df0), 60)):
        row = df0.iloc[i].astype("string").fillna("")
        joined = "|".join([norm_text(x) for x in row.tolist()])
        hits = 0
        if "DATE" in joined:
            hits += 1
        if "DESCRIPTION" in joined or "DETAIL" in joined or "PARTICULAR" in joined:
            hits += 1
        if "DEBIT" in joined or "WITHDRAW" in joined:
            hits += 1
        if "CREDIT" in joined or "DEPOSIT" in joined:
            hits += 1
        if re.search(r"\bAMOUNT\b", joined):
            hits += 1
        if "BAL" in joined or "BALANCE" in joined:
            hits += 1
        if "วันที่" in joined:
            hits += 1
        if "รายการ" in joined or "รายละเอียด" in joined:
            hits += 1
        if "เดบิต" in joined or "ถอน" in joined:
            hits += 1
        if "เครดิต" in joined or "ฝาก" in joined:
            hits += 1
        if "คงเหลือ" in joined or "ยอดคงเหลือ" in joined:
            hits += 1
        if hits >= 3:
            return i
    return None


def extract_account_from_file(path: str) -> str:
    """Read full account number from statement Excel metadata (all sheets)."""
    if not os.path.isfile(path):
        return ""

    bank_name = detect_bank_from_path(path)
    fallback = infer_account_from_filename(path, bank_name)
    resolved = fallback or ""

    try:
        xls = open_excel_file(path)
    except Exception:
        return resolved

    for sheet in xls.sheet_names:
        try:
            df0 = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=object)
        except Exception:
            continue
        if df0.empty:
            continue
        if _find_header_row(df0) is None:
            continue
        meta_account = _extract_account_from_metadata(df0)
        if meta_account:
            resolved = meta_account

    return resolved


# Match kcw-v2 / bank.fp_transaction_detail — NOT display รายการ / time labels.
STABLE_DETAIL_KEYS = (
    "รายละเอียด",  # KBANK Thai export
    "DESCRIPTION",  # KTB / English exports
    "PARTICULAR",
    "NARRATION",
)


def extract_stable_transaction_detail(raw: dict) -> str:
    """Stable bank detail from raw_json keys (canonical identity component).

    Looks up รายละเอียด / DESCRIPTION / PARTICULAR / NARRATION only.
    Does not use display columns such as รายการ or TIME — those drift between
    overlapping monthly vs cumulative KBANK exports and caused auto_v1 duplicates.
    """
    if not raw:
        return ""
    # Prefer exact key match first (preserves Thai keys that norm_text would alter).
    for key in STABLE_DETAIL_KEYS:
        val = raw.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        s = str(val).strip()
        if s:
            return s
    # Case-/space-insensitive fallback for odd header casing from Excel.
    by_norm = {norm_text(k): v for k, v in raw.items()}
    for key in STABLE_DETAIL_KEYS:
        val = by_norm.get(norm_text(key))
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        s = str(val).strip()
        if s:
            return s
    return ""


def compute_transaction_fingerprint(
    *,
    account_no: str,
    txn_date: date | datetime,
    amount: Decimal | float,
    direction: str,
    bank_reference: str | None,
    balance_after: Decimal | float | None,
    raw_json: dict | None = None,
    description: str | None = None,
    transaction_detail: str | None = None,
) -> str:
    """Canonical transaction identity (parser_version auto_v2).

    Identity: account | date | amount | direction | stable_detail | bank_reference | balance.
    Must match production ``bank.fp_build_hash`` / kcw-v2 ``buildTransactionFingerprint``.

    Display ``description`` is intentionally unused. Overlapping KBANK exports may
    show the same transfer as ``โอนเงิน`` in one file and ``08:31:00`` in another;
    hashing that field recreated duplicate ``statement_lines`` (0393 / 3557 / …).

    Fallback policy: if neither ``transaction_detail`` nor a stable raw_json key is
    present, use empty string — do **not** fall back to display description. Empty
    detail is still unique enough when amount + balance_after differ; falling back
    to รายการ/time is what broke dedupe.
    """
    # `description` is intentionally unused (call-site compat only).
    _ = description
    if transaction_detail is not None and str(transaction_detail).strip():
        stable_detail = str(transaction_detail).strip()
    else:
        stable_detail = extract_stable_transaction_detail(raw_json or {})
    fp_input = "|".join(
        [
            norm_text(account_no),
            norm_date(txn_date),
            norm_money(amount),
            norm_text(direction),
            norm_text(stable_detail),
            norm_text(bank_reference),
            norm_money(balance_after) if balance_after is not None else "",
        ]
    )
    return sha256_text(fp_input)


def is_short_account_no(account_no: str | None) -> bool:
    if not account_no:
        return True
    s = str(account_no).strip()
    if not s:
        return True
    if "-" in s or len(s) > 6:
        return False
    return s.isdigit() and len(s) <= 5
