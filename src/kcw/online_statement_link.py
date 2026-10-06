"""Link marketplace settlement payouts to PARTS9 TAD bills.

Reads KCW-Data/kcw_analytics/01_raw/statement/online/{Lazada,Shopee,Tiktok}.
Each payout is money that actually left the marketplace wallet for the bank.
Orders and fee lines inside that payout are matched to raw HQ SIMAS by the PO
column: exact order id, or the order id with a short head/tail trimmed
(Lazada drops the first 2 digits, TikTok the first 3, Shopee is exact).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from zoneinfo import ZoneInfo

BKK = ZoneInfo("Asia/Bangkok")
Q = Decimal("0.01")
PLATFORM_DIRS = {
    "Lazada": "lazada",
    "Shopee": "shopee",
    "Tiktok": "tiktok",
}
MATCH_TRIMS = (1, 2, 3)
MIN_PO_LEN = 10


def _q(value: Decimal) -> Decimal:
    return value.quantize(Q, rounding=ROUND_HALF_UP)


def as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value).strip()
    return str(value).strip()


def as_decimal(value: object) -> Decimal:
    if value is None or value == "":
        return Decimal("0")
    if isinstance(value, Decimal):
        return _q(value)
    text = as_text(value).replace(",", "")
    if text in {"", "-"}:
        return Decimal("0")
    return _q(Decimal(text))


def as_aware(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=BKK)
    return value.astimezone(BKK)


def _parse_date(text: str, formats: tuple[str, ...]) -> datetime | None:
    raw = (text or "").strip()
    if not raw:
        return None
    for fmt in formats:
        try:
            return as_aware(datetime.strptime(raw, fmt))
        except ValueError:
            continue
    return None


@dataclass
class FeeLine:
    name: str
    amount: Decimal


@dataclass
class StatementLine:
    kind: str  # order | adjustment
    order_id: str
    fee_name: str
    detail: str
    gross: Decimal
    expense: Decimal
    net: Decimal
    fees: list[FeeLine] = field(default_factory=list)
    txn_at: datetime | None = None


@dataclass
class Payout:
    platform: str
    shop: str
    payout_at: datetime | None
    amount: Decimal
    reference: str
    status: str  # transferred | pending
    source_file: str
    note: str
    lines: list[StatementLine] = field(default_factory=list)

    @property
    def payout_key(self) -> str:
        when = self.payout_at.isoformat() if self.payout_at else ""
        # Wallet remainder is one row per shop. A dated statement stays unique.
        if self.status == "pending" and self.reference == "ยังไม่ถอนเข้าบัญชี":
            return f"pending|{self.platform}|{self.shop}"
        return f"{self.platform}|{self.shop}|{when}|{self.reference}|{self.amount}"


@dataclass
class PeakReceipt:
    platform: str
    order_id: str
    receipt_no: str | None
    receipt_date: str | None
    receipt_status: str
    receipt_amount: Decimal | None
    order_status: str
    order_amount: Decimal | None
    shop_name: str
    source_file: str


@dataclass
class BillLink:
    platform: str
    order_id: str
    payout_key: str
    billno: str
    po: str
    match_method: str
    bill_date: str
    aftertax: Decimal | None
    acctname: str
    canceled: bool


def statement_root() -> Path:
    from src.kcw.paths import raw_dir

    return raw_dir() / "statement" / "online"


def simas_csv_path() -> Path:
    from src.kcw.paths import raw_dir

    return raw_dir() / "raw_hq_simas_sales_bills.csv"


def _iter_workbooks(root: Path):
    from openpyxl import load_workbook

    for folder_name, platform in PLATFORM_DIRS.items():
        platform_dir = root / folder_name
        if not platform_dir.is_dir():
            continue
        for shop_dir in sorted(p for p in platform_dir.iterdir() if p.is_dir()):
            for path in sorted(shop_dir.glob("*.xlsx")):
                if path.name.startswith("~$"):
                    continue
                wb = load_workbook(path, read_only=True, data_only=True)
                try:
                    yield platform, shop_dir.name, path, wb
                finally:
                    wb.close()


def _sheet_rows(wb, preferred: tuple[str, ...]) -> list[tuple]:
    name = next((n for n in preferred if n in wb.sheetnames), None)
    if name is None:
        for sheet_name in wb.sheetnames:
            if any(token in sheet_name for token in preferred):
                name = sheet_name
                break
    if name is None:
        return []
    return [tuple(row) for row in wb[name].iter_rows(values_only=True)]


def parse_lazada_rows(
    rows: list[tuple],
    *,
    shop: str,
    source_file: str,
) -> list[Payout]:
    if not rows:
        return []
    header = [as_text(c) for c in rows[0]]
    idx = {name: i for i, name in enumerate(header)}
    needed = ("Amount", "Statement", "Paid Status", "Order No.", "Fee Name", "Transaction Type")
    if any(name not in idx for name in needed):
        return []

    grouped: dict[str, list[tuple]] = {}
    for row in rows[1:]:
        if not row or all(cell is None or as_text(cell) == "" for cell in row):
            continue
        statement = as_text(row[idx["Statement"]]) or "(no statement)"
        grouped.setdefault(statement, []).append(row)

    payouts: list[Payout] = []
    for statement, group in grouped.items():
        by_order: dict[str, list[tuple]] = {}
        for row in group:
            by_order.setdefault(as_text(row[idx["Order No."]]), []).append(row)
        lines: list[StatementLine] = []
        paid_flags = set()
        for order_id, order_rows in by_order.items():
            fees: list[FeeLine] = []
            gross = Decimal("0")
            expense = Decimal("0")
            detail = ""
            for row in order_rows:
                amount = as_decimal(row[idx["Amount"]])
                fee_name = as_text(row[idx["Fee Name"]]) or as_text(row[idx["Transaction Type"]])
                if not detail:
                    detail = as_text(row[idx["Transaction Type"]])
                paid_flags.add(as_text(row[idx["Paid Status"]]).lower())
                if amount >= 0:
                    gross += amount
                else:
                    expense += amount
                fees.append(FeeLine(name=fee_name, amount=amount))
            kind = "order" if order_id else "adjustment"
            lines.append(
                StatementLine(
                    kind=kind,
                    order_id=order_id,
                    fee_name="",
                    detail=detail,
                    gross=_q(gross),
                    expense=_q(expense),
                    net=_q(gross + expense),
                    fees=fees,
                )
            )
        net = _q(sum((line.net for line in lines), Decimal("0")))
        start = statement.split("-")[0].strip()
        payout_at = _parse_date(start, ("%d %b %Y", "%d %B %Y"))
        transferred = paid_flags <= {"paid", ""}
        payouts.append(
            Payout(
                platform="lazada",
                shop=shop,
                payout_at=payout_at,
                amount=net,
                reference=statement,
                status="transferred" if transferred else "pending",
                source_file=source_file,
                note="ยอดสุทธิของ Statement ที่ Lazada จ่ายแล้ว",
                lines=lines,
            )
        )
    return payouts


def parse_shopee_rows(
    rows: list[tuple],
    *,
    shop: str,
    source_file: str,
) -> list[Payout]:
    header_at = None
    for i, row in enumerate(rows):
        if row and as_text(row[0]) == "วันที่" and len(row) > 3 and as_text(row[1]) == "ประเภทการทำธุรกรรม":
            header_at = i
            break
    if header_at is None:
        return []

    parsed: list[tuple[datetime, str, str, str, Decimal, str]] = []
    for row in rows[header_at + 1 :]:
        if not row or as_text(row[0]) == "":
            continue
        txn_at = _parse_date(as_text(row[0]), ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"))
        if txn_at is None:
            continue
        txn_type = as_text(row[1])
        detail = as_text(row[2]) if len(row) > 2 else ""
        order_id = as_text(row[3]) if len(row) > 3 else ""
        if order_id == "-":
            order_id = ""
        amount = as_decimal(row[5]) if len(row) > 5 else Decimal("0")
        status = as_text(row[6]) if len(row) > 6 else ""
        parsed.append((txn_at, txn_type, detail, order_id, amount, status))
    parsed.sort(key=lambda item: item[0])

    payouts: list[Payout] = []
    bucket: list[tuple] = []

    def close_pending(items: list[tuple]) -> None:
        if not items:
            return
        lines = [_shopee_line(*item) for item in items]
        net = _q(sum((line.net for line in lines), Decimal("0")))
        payouts.append(
            Payout(
                platform="shopee",
                shop=shop,
                payout_at=items[-1][0],
                amount=net,
                reference="ยังไม่ถอนเข้าบัญชี",
                status="pending",
                source_file=source_file,
                note="รายการในวอลเล็ตหลังการถอนครั้งล่าสุด ยังไม่เข้าบัญชี",
                lines=lines,
            )
        )

    for item in parsed:
        txn_at, txn_type, detail, _order_id, amount, status = item
        if txn_type == "การถอนเงิน" and status == "ทำรายการสำเร็จ":
            lines = [_shopee_line(*b) for b in bucket]
            payouts.append(
                Payout(
                    platform="shopee",
                    shop=shop,
                    payout_at=txn_at,
                    amount=_q(abs(amount)),
                    reference=detail or "การถอนเงิน",
                    status="transferred",
                    source_file=source_file,
                    note="ผลรวมรายรับและค่าปรับในวอลเล็ตก่อนการถอนเข้าบัญชี",
                    lines=lines,
                )
            )
            bucket = []
            continue
        if txn_type == "การถอนเงิน":
            continue
        bucket.append(item)
    close_pending(bucket)
    return payouts


def _shopee_line(
    txn_at: datetime,
    txn_type: str,
    detail: str,
    order_id: str,
    amount: Decimal,
    _status: str,
) -> StatementLine:
    if order_id and txn_type == "รายรับจากคำสั่งซื้อ":
        return StatementLine(
            kind="order",
            order_id=order_id,
            fee_name=txn_type,
            detail=detail,
            gross=amount,
            expense=Decimal("0"),
            net=amount,
            txn_at=txn_at,
        )
    return StatementLine(
        kind="adjustment",
        order_id=order_id,
        fee_name=txn_type or detail or "รายการปรับปรุง",
        detail=detail,
        gross=amount if amount > 0 else Decimal("0"),
        expense=amount if amount < 0 else Decimal("0"),
        net=amount,
        fees=[FeeLine(name=txn_type or detail or "รายการปรับปรุง", amount=amount)],
        txn_at=txn_at,
    )


def parse_tiktok_rows(
    order_rows: list[tuple],
    withdraw_rows: list[tuple],
    *,
    shop: str,
    source_file: str,
) -> tuple[list[StatementLine], list[Payout]]:
    orders = _tiktok_orders(order_rows)
    withdrawals = _tiktok_withdrawals(withdraw_rows, shop=shop, source_file=source_file)
    return orders, withdrawals


def _tiktok_orders(rows: list[tuple]) -> list[StatementLine]:
    if not rows:
        return []
    header = [as_text(c) for c in rows[0]]
    idx = {name: i for i, name in enumerate(header) if name}
    id_col = idx.get("หมายเลขคำสั่งซื้อ/การปรับ")
    if id_col is None:
        return []
    type_col = idx.get("ประเภทธุรกรรม")
    when_col = idx.get("เวลาที่ชำระคำสั่งซื้อ")
    net_col = idx.get("ยอดการชำระเงินทั้งหมด")
    gross_col = idx.get("รายได้ทั้งหมด")
    fee_total_col = idx.get("ค่าธรรมเนียมทั้งหมด")
    fee_start = fee_total_col + 1 if fee_total_col is not None else None
    fee_end = idx.get("จำนวนการปรับยอด", len(header))

    lines: list[StatementLine] = []
    for row in rows[1:]:
        if not row or id_col >= len(row):
            continue
        order_id = as_text(row[id_col])
        if not order_id:
            continue
        net = as_decimal(row[net_col]) if net_col is not None and net_col < len(row) else Decimal("0")
        gross = as_decimal(row[gross_col]) if gross_col is not None and gross_col < len(row) else net
        expense = (
            as_decimal(row[fee_total_col])
            if fee_total_col is not None and fee_total_col < len(row)
            else _q(net - gross)
        )
        fees: list[FeeLine] = []
        if fee_start is not None:
            for col in range(fee_start, min(fee_end, len(header), len(row))):
                amount = as_decimal(row[col])
                if amount == 0:
                    continue
                fees.append(FeeLine(name=header[col], amount=amount))
        txn_type = as_text(row[type_col]) if type_col is not None and type_col < len(row) else ""
        txn_at = None
        if when_col is not None and when_col < len(row):
            txn_at = _parse_date(as_text(row[when_col]), ("%Y/%m/%d", "%Y-%m-%d"))
        lines.append(
            StatementLine(
                kind="order" if txn_type in {"", "คำสั่งซื้อ"} else "adjustment",
                order_id=order_id,
                fee_name=txn_type,
                detail=txn_type,
                gross=gross,
                expense=expense,
                net=net,
                fees=fees,
                txn_at=txn_at,
            )
        )
    return lines


def _tiktok_withdrawals(
    rows: list[tuple],
    *,
    shop: str,
    source_file: str,
) -> list[Payout]:
    if not rows:
        return []
    header = [as_text(c) for c in rows[0]]
    idx = {name: i for i, name in enumerate(header) if name}
    type_col = idx.get("ประเภทธุรกรรม", 0)
    ref_col = idx.get("ID อ้างอิง", 1)
    amount_col = idx.get("จำนวน", 3)
    status_col = idx.get("สถานะ", 4)
    when_col = idx.get("เวลาที่สำเร็จ", 5)
    bank_col = idx.get("บัญชีธนาคาร", 6)
    payouts: list[Payout] = []
    for row in rows[1:]:
        if not row or type_col >= len(row):
            continue
        if as_text(row[type_col]).lower() != "withdrawal":
            continue
        status = as_text(row[status_col]) if status_col < len(row) else ""
        if status.lower() != "transferred":
            continue
        when_text = as_text(row[when_col]) if when_col < len(row) else ""
        payout_at = _parse_date(when_text, ("%Y/%m/%d", "%Y-%m-%d"))
        amount = as_decimal(row[amount_col]) if amount_col < len(row) else Decimal("0")
        reference = as_text(row[ref_col]) if ref_col < len(row) else ""
        bank = as_text(row[bank_col]) if bank_col < len(row) else ""
        payouts.append(
            Payout(
                platform="tiktok",
                shop=shop,
                payout_at=payout_at,
                amount=_q(abs(amount)),
                reference=reference or "Withdrawal",
                status="transferred",
                source_file=source_file,
                note=(
                    f"ถอนเข้าบัญชี {bank}".strip()
                    + " — ออเดอร์ที่วันเงินเข้าไม่หลังวันถอน"
                ),
                lines=[],
            )
        )
    return payouts


def assign_tiktok(
    orders: list[StatementLine],
    withdrawals: list[Payout],
    *,
    shop: str,
    source_file: str,
) -> list[Payout]:
    withdrawals = sorted(withdrawals, key=lambda p: p.payout_at or datetime.min.replace(tzinfo=BKK))
    pending: list[StatementLine] = []
    for order in orders:
        placed = False
        prev: datetime | None = None
        for payout in withdrawals:
            if payout.payout_at is None or order.txn_at is None:
                continue
            if order.txn_at.date() <= payout.payout_at.date() and (
                prev is None or order.txn_at.date() > prev.date()
            ):
                payout.lines.append(order)
                placed = True
                break
            prev = payout.payout_at
        if not placed:
            pending.append(order)
    if pending:
        last = max((line.txn_at for line in pending if line.txn_at), default=None)
        net = _q(sum((line.net for line in pending), Decimal("0")))
        withdrawals.append(
            Payout(
                platform="tiktok",
                shop=shop,
                payout_at=last,
                amount=net,
                reference="ยังไม่ถอนเข้าบัญชี",
                status="pending",
                source_file=source_file,
                note="เงินเข้าวอลเล็ตหลังการถอนครั้งล่าสุด ยังไม่เข้าบัญชี",
                lines=pending,
            )
        )
    return withdrawals


def collect_payouts(root: Path | None = None) -> list[Payout]:
    root = root or statement_root()
    lazada_shopee: list[Payout] = []
    tiktok_orders: dict[str, list[StatementLine]] = {}
    tiktok_withdrawals: dict[str, list[Payout]] = {}
    tiktok_files: dict[str, str] = {}

    for platform, shop, path, wb in _iter_workbooks(root):
        rel = str(path.relative_to(root))
        if platform == "lazada":
            lazada_shopee.extend(parse_lazada_rows(_sheet_rows(wb, ("Transaction Overview",)), shop=shop, source_file=rel))
        elif platform == "shopee":
            lazada_shopee.extend(parse_shopee_rows(_sheet_rows(wb, ("Transaction Report",)), shop=shop, source_file=rel))
        elif platform == "tiktok":
            orders, withdrawals = parse_tiktok_rows(
                _sheet_rows(wb, ("รายละเอียดคำสั่งซื้อ", "Order details")),
                _sheet_rows(wb, ("บันทึกการถอน", "Withdrawals")),
                shop=shop,
                source_file=rel,
            )
            tiktok_orders.setdefault(shop, []).extend(orders)
            tiktok_withdrawals.setdefault(shop, []).extend(withdrawals)
            tiktok_files[shop] = rel

    payouts = _dedupe_payouts(lazada_shopee)
    for shop, orders in tiktok_orders.items():
        unique_orders = _dedupe_lines(orders)
        assigned = assign_tiktok(
            unique_orders,
            _dedupe_payouts(tiktok_withdrawals.get(shop, [])),
            shop=shop,
            source_file=tiktok_files.get(shop, ""),
        )
        payouts.extend(assigned)
    return payouts


def _dedupe_lines(lines: list[StatementLine]) -> list[StatementLine]:
    seen: set[tuple] = set()
    out: list[StatementLine] = []
    for line in lines:
        key = (line.kind, line.order_id, line.txn_at.isoformat() if line.txn_at else "", str(line.net))
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def _dedupe_payouts(payouts: list[Payout]) -> list[Payout]:
    """Overlapping weekly files repeat the same withdrawal. Keep the richer copy.

    The not-yet-withdrawn wallet row keeps the latest snapshot for that shop.
    """
    chosen: dict[str, Payout] = {}
    for payout in payouts:
        key = payout.payout_key
        current = chosen.get(key)
        if current is None:
            chosen[key] = payout
            continue
        if key.startswith("pending|"):
            current_at = current.payout_at or datetime.min.replace(tzinfo=BKK)
            incoming_at = payout.payout_at or datetime.min.replace(tzinfo=BKK)
            if incoming_at >= current_at:
                chosen[key] = payout
            continue
        if len(payout.lines) > len(current.lines):
            chosen[key] = payout
    return list(chosen.values())


def match_order(order_id: str, po_index: dict[str, list[dict]]) -> tuple[str, str, list[dict]] | None:
    oid = order_id.strip()
    if len(oid) < MIN_PO_LEN:
        return None
    strategies = [("exact", oid)]
    for n in MATCH_TRIMS:
        if len(oid) - n < MIN_PO_LEN:
            continue
        strategies.append((f"drop_front_{n}", oid[n:]))
        strategies.append((f"drop_end_{n}", oid[:-n]))
    for method, key in strategies:
        bills = po_index.get(key)
        if bills:
            return method, key, bills
    return None


def _is_tad(billno: str) -> bool:
    upper = billno.upper()
    return upper.startswith("TAD") or upper.startswith("CNTAD")


def _canceled(value: str) -> bool:
    return value.strip().upper() in {"1", "Y", "YES", "TRUE", "T"}


def load_tad_po_index(path: Path) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        for row in csv.DictReader(handle):
            billno = (row.get("BILLNO") or "").strip()
            if not _is_tad(billno):
                continue
            po = (row.get("PO") or "").strip()
            if len(po) < MIN_PO_LEN:
                continue
            index.setdefault(po, []).append(
                {
                    "billno": billno,
                    "po": po,
                    "bill_date": (row.get("BILLDATE") or "").strip()[:10],
                    "aftertax": (row.get("AFTERTAX") or "").strip(),
                    "acctname": (row.get("ACCTNAME") or "").strip(),
                    "canceled": _canceled(row.get("CANCELED") or ""),
                }
            )
    return index


def link_bills(payouts: list[Payout], po_index: dict[str, list[dict]]) -> list[BillLink]:
    links: list[BillLink] = []
    seen: set[tuple] = set()
    for payout in payouts:
        for line in payout.lines:
            if not line.order_id:
                continue
            matched = match_order(line.order_id, po_index)
            if not matched:
                continue
            method, po, bills = matched
            for bill in bills:
                key = (payout.payout_key, line.order_id, bill["billno"])
                if key in seen:
                    continue
                seen.add(key)
                after = bill["aftertax"]
                links.append(
                    BillLink(
                        platform=payout.platform,
                        order_id=line.order_id,
                        payout_key=payout.payout_key,
                        billno=bill["billno"],
                        po=po,
                        match_method=method,
                        bill_date=bill["bill_date"],
                        aftertax=as_decimal(after) if after else None,
                        acctname=bill["acctname"],
                        canceled=bool(bill["canceled"]),
                    )
                )
    return links


def payout_totals(payout: Payout, links: list[BillLink]) -> dict:
    order_ids = {line.order_id for line in payout.lines if line.order_id}
    expense = _q(sum((line.expense for line in payout.lines), Decimal("0")))
    component_net = _q(sum((line.net for line in payout.lines), Decimal("0")))
    matched_ids = {link.order_id for link in links if link.payout_key == payout.payout_key}
    return {
        "order_count": len(order_ids),
        "expense_amount": expense,
        "component_net": component_net,
        "matched_order_count": len(matched_ids),
    }


def _fees_json(fees: list[FeeLine]) -> list[dict]:
    return [{"name": fee.name, "amount": str(fee.amount)} for fee in fees]


def _peak_platform(label: str, filename: str) -> str:
    text = f"{label} {filename}".lower()
    if "lazada" in text:
        return "lazada"
    if "shopee" in text:
        return "shopee"
    if "tiktok" in text or "tik tok" in text:
        return "tiktok"
    return ""


def _peak_date(text: str) -> str | None:
    parsed = _parse_date(text, ("%d/%m/%Y", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S"))
    if parsed is None:
        return None
    return parsed.date().isoformat()


def _optional_amount(value: object) -> Decimal | None:
    text = as_text(value)
    if text in {"", "-"}:
        return None
    return as_decimal(value)


def parse_peak_rows(
    rows: list[tuple],
    *,
    source_file: str,
) -> list[PeakReceipt]:
    platform = ""
    shop = ""
    for row in rows[:40]:
        if not row or len(row) < 12:
            continue
        label = as_text(row[10])
        value = as_text(row[11])
        if "แพลตฟอร์ม" in label and value:
            platform = _peak_platform(value, source_file)
        if "ชื่อร้าน" in label and value:
            shop = value
    if not platform:
        platform = _peak_platform("", source_file)
    if not platform:
        return []

    found: list[PeakReceipt] = []
    for row in rows[1:]:
        if not row or len(row) < 7:
            continue
        order_id = as_text(row[2])
        if len(order_id) < 8 or not order_id.replace("-", "").isalnum():
            continue
        document = as_text(row[6]) if len(row) > 6 else ""
        receipt_no = document if document.upper().startswith("RT-") else None
        status = as_text(row[7]) if len(row) > 7 and receipt_no else document
        found.append(
            PeakReceipt(
                platform=platform,
                order_id=order_id,
                receipt_no=receipt_no,
                receipt_date=_peak_date(as_text(row[5])) if len(row) > 5 else None,
                receipt_status=status,
                receipt_amount=_optional_amount(row[8]) if len(row) > 8 else None,
                order_status=as_text(row[4]) if len(row) > 4 else "",
                order_amount=_optional_amount(row[3]) if len(row) > 3 else None,
                shop_name=shop,
                source_file=source_file,
            )
        )
    return found


def collect_peak_receipts(root: Path | None = None) -> list[PeakReceipt]:
    from openpyxl import load_workbook

    root = root or statement_root()
    folder = root / "peak"
    if not folder.is_dir():
        return []
    chosen: dict[tuple[str, str], PeakReceipt] = {}
    for path in sorted(folder.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            rows = [tuple(row) for row in wb.active.iter_rows(values_only=True)]
        finally:
            wb.close()
        for receipt in parse_peak_rows(rows, source_file=f"peak/{path.name}"):
            key = (receipt.platform, receipt.order_id)
            current = chosen.get(key)
            if current is None or (receipt.receipt_no and not current.receipt_no):
                chosen[key] = receipt
    return list(chosen.values())


def replace_links(
    payouts: list[Payout],
    links: list[BillLink],
    receipts: list[PeakReceipt] | None = None,
) -> dict:
    import psycopg2
    from psycopg2.extras import Json, execute_values

    from src.kcw.tar import supabase_db_url

    payout_rows = []
    line_rows = []
    for payout in payouts:
        totals = payout_totals(payout, links)
        payout_rows.append(
            (
                payout.payout_key,
                payout.platform,
                payout.shop,
                payout.payout_at,
                payout.amount,
                "THB",
                payout.reference,
                payout.status,
                payout.source_file,
                payout.note,
                totals["order_count"],
                totals["expense_amount"],
                totals["component_net"],
                totals["matched_order_count"],
            )
        )
        for i, line in enumerate(payout.lines, start=1):
            line_rows.append(
                (
                    payout.payout_key,
                    i,
                    line.kind,
                    line.order_id or None,
                    line.fee_name or None,
                    line.detail or None,
                    line.gross,
                    line.expense,
                    line.net,
                    Json(_fees_json(line.fees)),
                    line.txn_at,
                )
            )
    bill_rows = [
        (
            link.payout_key,
            link.platform,
            link.order_id,
            link.billno,
            link.po,
            link.match_method,
            link.bill_date or None,
            link.aftertax,
            link.acctname,
            link.canceled,
        )
        for link in links
    ]
    receipt_rows = [
        (
            receipt.platform,
            receipt.order_id,
            receipt.receipt_no,
            receipt.receipt_date,
            receipt.receipt_status or None,
            receipt.receipt_amount,
            receipt.order_status or None,
            receipt.order_amount,
            receipt.shop_name or None,
            receipt.source_file,
        )
        for receipt in (receipts or [])
    ]

    conn = psycopg2.connect(supabase_db_url(), sslmode="require")
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    truncate table
                      curated_kcw.online_payout_lines,
                      curated_kcw.online_order_bills,
                      curated_kcw.online_peak_receipts,
                      curated_kcw.online_payouts
                    """
                )
                execute_values(
                    cur,
                    """
                    insert into curated_kcw.online_payouts (
                      payout_key, platform, shop, payout_at, amount, currency,
                      reference, status, source_file, note,
                      order_count, expense_amount, component_net, matched_order_count
                    ) values %s
                    """,
                    payout_rows,
                    page_size=500,
                )
                if line_rows:
                    execute_values(
                        cur,
                        """
                        insert into curated_kcw.online_payout_lines (
                          payout_key, line_no, line_kind, order_id, fee_name, detail,
                          gross_amount, expense_amount, net_amount, fees, txn_at
                        ) values %s
                        """,
                        line_rows,
                        page_size=1000,
                    )
                if bill_rows:
                    execute_values(
                        cur,
                        """
                        insert into curated_kcw.online_order_bills (
                          payout_key, platform, order_id, billno, po, match_method,
                          bill_date, aftertax, acctname, canceled
                        ) values %s
                        """,
                        bill_rows,
                        page_size=1000,
                    )
                if receipt_rows:
                    execute_values(
                        cur,
                        """
                        insert into curated_kcw.online_peak_receipts (
                          platform, order_id, receipt_no, receipt_date, receipt_status,
                          receipt_amount, order_status, order_amount, shop_name, source_file
                        ) values %s
                        """,
                        receipt_rows,
                        page_size=1000,
                    )
    finally:
        conn.close()
    return {
        "payouts": len(payout_rows),
        "lines": len(line_rows),
        "bills": len(bill_rows),
        "receipts": len(receipt_rows),
    }


def run_link(*, upload: bool = True, root: Path | None = None) -> dict:
    payouts = collect_payouts(root)
    simas = simas_csv_path()
    po_index = load_tad_po_index(simas) if simas.is_file() else {}
    links = link_bills(payouts, po_index) if po_index else []
    receipts = collect_peak_receipts(root)
    summary = {
        "payouts": len(payouts),
        "transferred": sum(1 for p in payouts if p.status == "transferred"),
        "pending": sum(1 for p in payouts if p.status == "pending"),
        "orders": sum(1 for p in payouts for line in p.lines if line.order_id),
        "bills": len(links),
        "receipts": len(receipts),
        "receipts_issued": sum(1 for receipt in receipts if receipt.receipt_no),
        "simas_po_keys": len(po_index),
        "uploaded": False,
    }
    if upload:
        summary.update(replace_links(payouts, links, receipts))
        summary["uploaded"] = True
    return summary
