"""CLI for Lazada Seller Center reconciliation. Offline only."""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .config import DEFAULT_TOLERANCE, PROGRAM_NAME, PROGRAM_VERSION
from .file_detector import FileDetectionError
from .loaders import load_finance, load_orders, load_wallet
from .reconciliation import run_reconciliation
from .report_writer import write_report
from .validations import run_validations

logger = logging.getLogger("lazada_reconciliation")

MONTH_TOKENS = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
    "ม.ค.": "01",
    "ก.พ.": "02",
    "มี.ค.": "03",
    "เม.ย.": "04",
    "พ.ค.": "05",
    "มิ.ย.": "06",
    "ก.ค.": "07",
    "ส.ค.": "08",
    "ก.ย.": "09",
    "ต.ค.": "10",
    "พ.ย.": "11",
    "ธ.ค.": "12",
    "มกราคม": "01",
    "กุมภาพันธ์": "02",
    "มีนาคม": "03",
    "เมษายน": "04",
    "พฤษภาคม": "05",
    "มิถุนายน": "06",
    "กรกฎาคม": "07",
    "สิงหาคม": "08",
    "กันยายน": "09",
    "ตุลาคม": "10",
    "พฤศจิกายน": "11",
    "ธันวาคม": "12",
}


class UserMessage(Exception):
    def __init__(self, message_th: str):
        super().__init__(message_th)
        self.message_th = message_th


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="กระทบยอดรายได้และเงินรับจาก Lazada Seller Center (ออฟไลน์)",
    )
    parser.add_argument("--orders", default="input/orders", help="โฟลเดอร์ไฟล์คำสั่งซื้อ")
    parser.add_argument("--finance", default="input/finance", help="โฟลเดอร์ไฟล์รายการทางบัญชี")
    parser.add_argument("--wallet", default="input/wallet", help="โฟลเดอร์ไฟล์ยอดของฉัน")
    parser.add_argument("--output", default="output", help="โฟลเดอร์ผลลัพธ์")
    parser.add_argument("--month", default=None, help="เดือนที่ประมวลผล เช่น 2026-08")
    parser.add_argument("--tolerance", default=str(DEFAULT_TOLERANCE), help="tolerance บาท เช่น 0.01")
    return parser.parse_args(argv)


def setup_logging(output_dir: Path, month_label: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"lazada_reconciliation_{month_label}.log"
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    stream_handler.setLevel(logging.WARNING)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info("start program=%s version=%s", PROGRAM_NAME, PROGRAM_VERSION)
    return log_path


def _parse_tolerance(text: str) -> Decimal:
    try:
        return Decimal(str(text).strip())
    except (InvalidOperation, ValueError) as exc:
        raise UserMessage(f"ค่า tolerance ไม่ถูกต้อง: {text}") from exc


def infer_month(explicit: str | None, filenames: list[str]) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if explicit:
        value = explicit.strip()
        if len(value) == 7 and value[4] == "-":
            return value, warnings
        warnings.append(f"รูปแบบ --month={explicit} ไม่ใช่ YYYY-MM จะพยายามอ่านจากชื่อไฟล์")
    joined = " ".join(filenames).casefold()
    import re

    match = re.search(r"(20\d{2})[-_/ ](0[1-9]|1[0-2])", joined)
    if match:
        return f"{match.group(1)}-{match.group(2)}", warnings
    match = re.search(
        r"(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december|ส\.ค\.|สิงหาคม)[_\s-]*(20\d{2})",
        joined,
    )
    if match:
        month = MONTH_TOKENS.get(match.group(1), None)
        if month:
            return f"{match.group(2)}-{month}", warnings
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    warnings.append(
        "ไม่ทราบเดือนจากข้อมูลหรือชื่อไฟล์ จึงใช้ timestamp ในชื่อไฟล์ผลลัพธ์ และควรระบุ --month เช่น 2026-08"
    )
    return timestamp, warnings


def _thai_status_message(overall: str) -> str:
    if overall == "FAIL":
        return (
            "ประมวลผลเสร็จแต่รายงานมีสถานะ FAIL — สร้างไฟล์แล้ว "
            "ห้ามถือว่ากระทบยอดสำเร็จ ต้องเปิดชีต Validations และ Exceptions ก่อนลงบัญชี"
        )
    if overall == "WARNING":
        return (
            "ประมวลผลเสร็จแต่รายงานมี WARNING — สร้างไฟล์แล้ว "
            "นักบัญชีต้องตรวจรายการผิดปกติก่อนลงบัญชี"
        )
    return (
        "ประมวลผลเสร็จ และผ่านการตรวจสอบเบื้องต้น "
        "นักบัญชียังต้องตรวจผลอีกครั้ง โปรแกรมนี้ไม่ใช่คำรับรองทางบัญชี"
    )


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output)
    month_guess, month_warnings = infer_month(args.month, [])
    log_path = setup_logging(output_dir, month_guess)
    try:
        tolerance = _parse_tolerance(args.tolerance)
        orders_dir = Path(args.orders)
        finance_dir = Path(args.finance)
        wallet_dir = Path(args.wallet)

        print("กำลังอ่านไฟล์ต้นทาง (ไม่แก้ไขไฟล์ต้นฉบับ)...")
        orders = load_orders(orders_dir)
        finance = load_finance(finance_dir)
        wallet = load_wallet(wallet_dir)

        if len(orders.source_files) > 1:
            print(f"พบหลายไฟล์ในโฟลเดอร์คำสั่งซื้อ จึงรวมไฟล์: {', '.join(orders.source_files)}")
        if len(finance.source_files) > 1:
            print(f"พบหลายไฟล์ในโฟลเดอร์รายการทางบัญชี จึงรวมไฟล์: {', '.join(finance.source_files)}")
        if len(wallet.source_files) > 1:
            print(f"พบหลายไฟล์ในโฟลเดอร์ยอดของฉัน จึงรวมไฟล์: {', '.join(wallet.source_files)}")

        filenames = orders.source_files + finance.source_files + wallet.source_files
        month, inferred_warnings = infer_month(args.month, filenames)
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.info(
            "loaded orders_files=%s orders_rows=%s finance_files=%s finance_rows=%s wallet_files=%s wallet_rows=%s",
            orders.source_files,
            len(orders.frame),
            finance.source_files,
            len(finance.frame),
            wallet.source_files,
            len(wallet.frame),
        )

        result = run_reconciliation(
            orders,
            finance,
            wallet,
            month=month,
            generated_at=generated_at,
            tolerance=tolerance,
        )
        result.warnings.extend(month_warnings)
        result.warnings.extend(inferred_warnings)
        result = run_validations(result, orders, finance, wallet)

        output_name = f"lazada_reconciliation_{month}.xlsx"
        output_path = output_dir / output_name
        write_report(result, output_path)
        logger.info(
            "wrote report=%s overall=%s match_counts=%s exceptions=%s",
            output_path.name,
            result.overall_status,
            result.match_counts,
            0 if result.exceptions.empty else len(result.exceptions),
        )
        print(f"สร้างรายงานแล้ว: {output_path}")
        print(f"ไฟล์บันทึกขั้นตอน: {log_path}")
        print(_thai_status_message(result.overall_status))
        return 0 if result.overall_status != "FAIL" else 2
    except FileDetectionError as exc:
        logger.exception("detection_or_load_failed")
        print(exc.message_th)
        return 1
    except UserMessage as exc:
        logger.exception("user_error")
        print(exc.message_th)
        return 1
    except FileNotFoundError as exc:
        logger.exception("missing_path")
        print(f"ไม่พบไฟล์หรือโฟลเดอร์: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001
        logger.error("unhandled_error\n%s", traceback.format_exc())
        print(f"ประมวลผลไม่สำเร็จ: {exc.__class__.__name__} — ดูรายละเอียดในไฟล์ log")
        return 1


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
