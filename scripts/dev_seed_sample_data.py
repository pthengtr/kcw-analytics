"""Generate small synthetic PARTS9 raw CSVs for local development / testing.

The real `extract` step reads Windows SQL Server (PARTS9) and cannot run on
Linux, so this script fabricates the handful of raw CSVs that the downstream
TAR / gap-check pipeline consumes. It lets a developer exercise
`python -m src.kcw.pipeline tar` and `gap-check` end-to-end against a local
Postgres without any Windows/Drive dependency.

Usage:
    python scripts/dev_seed_sample_data.py [--data-root DIR] [--date YYYY-MM-DD]

Writes CSVs into <data-root>/01_raw/. Defaults to KCW_ANALYTICS_DATA_ROOT or
./dev_data/kcw_analytics.
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import date
from pathlib import Path

SIDET_HEADER = [
    "BILLNO", "BILLDATE", "BCODE", "DETAIL", "QTY", "MTP", "UI",
    "PRICE", "AMOUNT", "ISVAT", "CANCELED",
]
PIDET_HEADER = [
    "BILLNO", "BILLDATE", "BCODE", "DETAIL", "QTY", "MTP", "UI",
    "PRICE", "AMOUNT", "ISVAT", "TAXIC",
]
SIMAS_HEADER = ["BILLNO", "BILLDATE", "PO", "PAID", "VOUCDATE2"]


def _write(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--date", default="2026-07-20", help="Sales billdate (YYYY-MM-DD)")
    args = parser.parse_args()

    root = args.data_root or os.getenv("KCW_ANALYTICS_DATA_ROOT") or "dev_data/kcw_analytics"
    raw = Path(root).expanduser() / "01_raw"
    raw.mkdir(parents=True, exist_ok=True)

    d = args.date
    py = date.fromisoformat(d)
    purchase_date = py.replace(month=1, day=15).isoformat()  # earlier VAT purchases

    # Purchases (HQ pidet is used as the cost/VAT history for BOTH sites).
    _write(raw / "raw_hq_pidet_purchase_lines.csv", PIDET_HEADER, [
        ["P100", purchase_date, "12345678", "BRAKE PAD", 10, 1, "PCS", 900, 900, "Y", "Y"],
        ["P101", purchase_date, "23456789", "OIL FILTER", 20, 1, "PCS", 2000, 2000, "Y", "Y"],
        ["P102", purchase_date, "34567890", "WIPER", 5, 1, "PCS", 250, 250, "N", "N"],
    ])

    # HQ sales lines (non-VAT sales whose last purchase was VAT => TAR eligible).
    _write(raw / "raw_hq_sidet_sales_lines.csv", SIDET_HEADER, [
        ["C001", d, "12345678", "BRAKE PAD", 1, 1, "PCS", 100, 100, "N", "N"],
        ["C002", d, "23456789", "OIL FILTER", 2, 1, "PCS", 125, 250, "N", "N"],
        ["C003", d, "34567890", "WIPER", 1, 1, "PCS", 50, 50, "N", "N"],   # last purchase non-VAT -> excluded
        ["C004", d, "12345678", "BRAKE PAD RETURN", -1, 1, "PCS", 30, -30, "N", "N"],  # negative -> CNTAR
    ])

    # SYP sales lines (=> 3TAR eligible).
    _write(raw / "raw_syp_sidet_sales_lines.csv", SIDET_HEADER, [
        ["S001", d, "12345678", "BRAKE PAD", 1, 1, "PCS", 80, 80, "N", "N"],
        ["S002", d, "23456789", "OIL FILTER", 1, 1, "PCS", 120, 120, "N", "N"],
    ])

    # Sales bill headers (provide PO for negative/CNTAR join).
    _write(raw / "raw_hq_simas_sales_bills.csv", SIMAS_HEADER, [
        ["C001", d, "PO-C001", "N", ""],
        ["C004", d, "PO-C004", "N", ""],
    ])
    _write(raw / "raw_syp_simas_sales_bills.csv", SIMAS_HEADER, [
        ["S001", d, "PO-S001", "N", ""],
        ["S002", d, "PO-S002", "N", ""],
    ])

    print(f"\nSample raw CSVs ready in {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
