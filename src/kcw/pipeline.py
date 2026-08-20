"""CLI entrypoints for BAT / Claude Cowork: extract, tar catch-up, gap-check."""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path


def _setup_path() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def cmd_gap_check(_args: argparse.Namespace) -> int:
    from src.kcw import paths
    from src.kcw.tar import eligible_max_billdate, load_raw_csvs, max_fin_billdate, prepare_eligible_frames

    raw = paths.raw_dir()
    print(f"raw_dir={raw}")
    data = load_raw_csvs(raw)
    hq, syp = prepare_eligible_frames(data)
    eligible_end = eligible_max_billdate(hq, syp)
    max_fin = max_fin_billdate()
    today = date.today()
    print(f"today={today}")
    print(f"eligible_max_raw_billdate={eligible_end}")
    print(f"max_fin_billdate={max_fin}")
    if max_fin and eligible_end and max_fin < eligible_end:
        lag = (eligible_end - max_fin).days
        print(f"GAP: billgen lags raw by {lag} day(s) — run: python -m src.kcw.pipeline tar --catch-up")
        return 2
    if max_fin and eligible_end and max_fin == eligible_end:
        print("OK: billgen caught up to latest eligible raw billdate")
        return 0
    print("OK: no lag detected (or empty fin / raw)")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    from src.kcw.extract_parts9 import extract_tables

    tables = None
    if getattr(args, "tables", None):
        tables = tuple(t.strip().upper() for t in args.tables.split(",") if t.strip())
    extract_tables(args.site, tables=tables)
    return 0


def cmd_upload_armas_apmas(_args: argparse.Namespace) -> int:
    """HQ A follow-up: copy Drive ARMAS/APMAS raw CSVs into raw_kcw.*."""
    from src.kcw.upload_raw import upload_armas_apmas

    upload_armas_apmas()
    return 0


def cmd_upload_daily_raw(_args: argparse.Namespace) -> int:
    """After SYP+HQ extracts: masters + POs + invoices + ICMAS + RVMAS + PVMAS + BRDET/BPDET."""
    from src.kcw.upload_raw import upload_daily_raw

    upload_daily_raw()
    return 0


def cmd_upload_pomas_podet(args: argparse.Namespace) -> int:
    """Drive raw_{site}_pomas/podet CSVs -> raw_kcw (one site or both)."""
    from src.kcw.upload_raw import upload_pomas_podet

    upload_pomas_podet(args.site)
    return 0


def cmd_sync_pomas_podet(args: argparse.Namespace) -> int:
    """Extract POMAS/PODET for a site, then upload that site to Supabase."""
    from src.kcw.extract_parts9 import PO_TABLES, extract_tables
    from src.kcw.upload_raw import upload_pomas_podet

    extract_tables(args.site, tables=PO_TABLES)
    upload_pomas_podet(args.site)
    return 0


def cmd_upload_iclow(args: argparse.Namespace) -> int:
    """Drive raw_{site}_iclow_stock_orders.csv -> raw_kcw (one site or both)."""
    from src.kcw.upload_raw import upload_iclow

    upload_iclow(args.site)
    return 0


def cmd_sync_iclow(args: argparse.Namespace) -> int:
    """Extract ICLOW for a site, then upload that site to Supabase."""
    from src.kcw.extract_parts9 import ICLOW_TABLES, extract_tables
    from src.kcw.upload_raw import upload_iclow

    extract_tables(args.site, tables=ICLOW_TABLES)
    upload_iclow(args.site)
    return 0


def cmd_upload_icmas(args: argparse.Namespace) -> int:
    """Drive raw_{site}_icmas_products.csv -> raw_kcw (one site or both)."""
    from src.kcw.upload_raw import upload_icmas

    upload_icmas(args.site)
    return 0


def cmd_sync_icmas(args: argparse.Namespace) -> int:
    """Extract ICMAS for a site, then upload that site to Supabase."""
    from src.kcw.extract_parts9 import ICMAS_TABLES, extract_tables
    from src.kcw.upload_raw import upload_icmas

    extract_tables(args.site, tables=ICMAS_TABLES)
    upload_icmas(args.site)
    return 0


def cmd_upload_po_related(args: argparse.Namespace) -> int:
    """Drive PO/ICLOW CSVs -> raw_kcw (one site or both)."""
    from src.kcw.upload_raw import upload_po_related

    upload_po_related(args.site)
    return 0


def cmd_sync_po_related(args: argparse.Namespace) -> int:
    """
    Extract POMAS/PODET + ICLOW for one site, then upload to Supabase.

    HQ also extracts SIDET/SIMAS and uploads latest 6 months to raw_kcw (HQ-only).
    SYP runs separately (different PARTS9 servers / machines).
    Inventory on-hand qty is separate (run_inventory_sync.bat / notebook 50).
    """
    from src.kcw.extract_parts9 import PO_RELATED_TABLES, SI_TABLES, extract_tables
    from src.kcw.upload_raw import upload_po_related, upload_simas_sidet

    site = args.site
    print(f"[sync-po-related] site={site} tables={','.join(PO_RELATED_TABLES)}")
    extract_tables(site, tables=PO_RELATED_TABLES)
    upload_po_related(site)

    if site == "hq":
        print(f"[sync-po-related] site=hq sales tables={','.join(SI_TABLES)} (6 months)")
        extract_tables("hq", tables=SI_TABLES)
        upload_simas_sidet()

    print(f"[sync-po-related] DONE site={site}")
    return 0


def cmd_upload_simas_sidet(_args: argparse.Namespace) -> int:
    """Drive raw_hq_simas/sidet CSVs -> raw_kcw (HQ only, latest 6 months)."""
    from src.kcw.upload_raw import upload_simas_sidet

    upload_simas_sidet()
    return 0


def cmd_sync_simas_sidet(_args: argparse.Namespace) -> int:
    """Extract HQ SIDET/SIMAS, then upload latest 6 months to raw_kcw."""
    from src.kcw.extract_parts9 import SI_TABLES, extract_tables
    from src.kcw.upload_raw import upload_simas_sidet

    print(f"[sync-simas-sidet] site=hq tables={','.join(SI_TABLES)}")
    extract_tables("hq", tables=SI_TABLES)
    upload_simas_sidet()
    print("[sync-simas-sidet] DONE site=hq")
    return 0


def cmd_upload_brdet_bpdet(_args: argparse.Namespace) -> int:
    """Drive raw_hq_brdet/bpdet CSVs -> raw_kcw (HQ only)."""
    from src.kcw.upload_raw import upload_brdet_bpdet

    upload_brdet_bpdet()
    return 0


def cmd_sync_brdet_bpdet(_args: argparse.Namespace) -> int:
    """Extract HQ BRDET/BPDET (cheque/transfer registers), then upload to raw_kcw."""
    from src.kcw.extract_parts9 import CHEQUE_TABLES, extract_tables
    from src.kcw.upload_raw import upload_brdet_bpdet

    print(f"[sync-brdet-bpdet] site=hq tables={','.join(CHEQUE_TABLES)}")
    extract_tables("hq", tables=CHEQUE_TABLES)
    upload_brdet_bpdet()
    print("[sync-brdet-bpdet] DONE site=hq")
    return 0


def cmd_tar(args: argparse.Namespace) -> int:
    from src.kcw.tar import delete_fin_for_day, run_bill_generation_for_day, run_catchup
    from src.kcw.tar import load_raw_csvs, prepare_eligible_frames, supabase_db_url

    if args.reprocess:
        delete_fin_for_day(args.reprocess)
        data = load_raw_csvs()
        hq, syp = prepare_eligible_frames(data)
        run_bill_generation_for_day(
            args.reprocess,
            data=data,
            hq_eligible=hq,
            syp_eligible=syp,
            skip_if_done=False,
        )
        return 0

    if args.catch_up or args.date is None:
        run_catchup(
            start_date=args.start,
            end_date=args.end or args.date,
            skip_if_done=not args.force,
        )
        return 0

    data = load_raw_csvs()
    hq, syp = prepare_eligible_frames(data)
    run_bill_generation_for_day(
        args.date,
        data=data,
        hq_eligible=hq,
        syp_eligible=syp,
        db_url=supabase_db_url(),
        skip_if_done=not args.force,
    )
    return 0


def cmd_tar_report(args: argparse.Namespace) -> int:
    from src.kcw.tar_report import run_tar_reports

    start = args.start or args.date
    end = args.end or args.date
    if not start:
        start = date.today().isoformat()
        end = start
    summary = run_tar_reports(
        start,
        end,
        prune_stale=not args.keep_stale,
    )
    print(f"[tar-report] done days={len(summary['days'])} pruned={len(summary['pruned'])}")
    for row in summary["days"]:
        print(
            f"  {row['date']} hq_tar={row['hq_tar']} hq_cntar={row['hq_cntar']} "
            f"syp_tar={row['syp_tar']} syp_cntar={row['syp_cntar']}"
        )
    for path in summary["pruned"]:
        print(f"  pruned {path}")
    return 0


def cmd_bank_statement_report(args: argparse.Namespace) -> int:
    """Monthly multi-account bank statement Excel (VAT-style layout)."""
    from src.kcw.bank_statement_report import (
        run_bank_statement_report,
        write_fixture_sample,
    )

    if getattr(args, "fixture_sample", False):
        out = write_fixture_sample(
            Path(args.output) if getattr(args, "output", None) else None
        )
        print(f"fixture_sample={out}")
        return 0

    out = run_bank_statement_report(
        year=args.year,
        month=args.month,
        out_path=Path(args.output) if getattr(args, "output", None) else None,
    )
    print(f"output={out}")
    return 0


def cmd_backfill_statement_accounts(args: argparse.Namespace) -> int:
    from src.kcw.backfill_statement_accounts import backfill

    return backfill(apply=bool(args.apply))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.kcw.pipeline",
        description="KCW analytics pipeline steps (Windows Task Scheduler / Claude Cowork)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("gap-check", help="Compare raw eligible max date vs billgen max fin date")
    g.set_defaults(func=cmd_gap_check)

    e = sub.add_parser("extract", help="PARTS9 -> Drive 01_raw")
    e.add_argument("--site", choices=("hq", "syp"), required=True)
    e.add_argument(
        "--tables",
        help="Optional comma-separated PARTS9 tables (default: site full/minimal set)",
    )
    e.set_defaults(func=cmd_extract)

    u = sub.add_parser(
        "upload-armas-apmas",
        help="Drive raw_hq_armas/apmas CSVs -> raw_kcw (staging replace)",
    )
    u.set_defaults(func=cmd_upload_armas_apmas)

    ud = sub.add_parser(
        "upload-daily-raw",
        help=(
            "Drive daily raw CSVs -> raw_kcw: armas/apmas, "
            "pomas/podet (hq+syp), pimas/pidet (hq), icmas (hq+syp), "
            "rvmas/pvmas (hq), brdet/bpdet (hq), iclow (hq+syp), "
            "simas/sidet (hq, 6 months)"
        ),
    )
    ud.set_defaults(func=cmd_upload_daily_raw)

    upo = sub.add_parser(
        "upload-pomas-podet",
        help="Drive raw_{site}_pomas/podet CSVs -> raw_kcw (staging replace)",
    )
    upo.add_argument(
        "--site",
        choices=("hq", "syp"),
        default=None,
        help="Upload one site only (default: both)",
    )
    upo.set_defaults(func=cmd_upload_pomas_podet)

    spo = sub.add_parser(
        "sync-pomas-podet",
        help="Extract POMAS/PODET for a site then upload that site to raw_kcw",
    )
    spo.add_argument("--site", choices=("hq", "syp"), required=True)
    spo.set_defaults(func=cmd_sync_pomas_podet)

    ui = sub.add_parser(
        "upload-iclow",
        help="Drive raw_{site}_iclow_stock_orders.csv -> raw_kcw (staging replace)",
    )
    ui.add_argument(
        "--site",
        choices=("hq", "syp"),
        default=None,
        help="Upload one site only (default: both)",
    )
    ui.set_defaults(func=cmd_upload_iclow)

    si = sub.add_parser(
        "sync-iclow",
        help="Extract ICLOW for a site then upload that site to raw_kcw (pending-receive tracker)",
    )
    si.add_argument("--site", choices=("hq", "syp"), required=True)
    si.set_defaults(func=cmd_sync_iclow)

    uic = sub.add_parser(
        "upload-icmas",
        help="Drive raw_{site}_icmas_products.csv -> raw_kcw (staging replace)",
    )
    uic.add_argument(
        "--site",
        choices=("hq", "syp"),
        default=None,
        help="Upload one site only (default: both)",
    )
    uic.set_defaults(func=cmd_upload_icmas)

    sic = sub.add_parser(
        "sync-icmas",
        help="Extract ICMAS for a site then upload that site to raw_kcw (product masters)",
    )
    sic.add_argument("--site", choices=("hq", "syp"), required=True)
    sic.set_defaults(func=cmd_sync_icmas)

    upr = sub.add_parser(
        "upload-po-related",
        help="Drive POMAS/PODET + ICLOW CSVs -> raw_kcw (staging replace)",
    )
    upr.add_argument(
        "--site",
        choices=("hq", "syp"),
        default=None,
        help="Upload one site only (default: both)",
    )
    upr.set_defaults(func=cmd_upload_po_related)

    spr = sub.add_parser(
        "sync-po-related",
        help=(
            "Extract POMAS/PODET + ICLOW for one site then upload to raw_kcw. "
            "HQ also syncs SIDET/SIMAS (latest 6 months). "
            "HQ and SYP must run separately. "
            "Does not update inventory_qty_latest — use run_inventory_sync.bat for that."
        ),
    )
    spr.add_argument("--site", choices=("hq", "syp"), required=True)
    spr.set_defaults(func=cmd_sync_po_related)

    usi = sub.add_parser(
        "upload-simas-sidet",
        help="Drive raw_hq_simas/sidet CSVs -> raw_kcw (HQ only, latest 6 months)",
    )
    usi.set_defaults(func=cmd_upload_simas_sidet)

    ssi = sub.add_parser(
        "sync-simas-sidet",
        help="Extract HQ SIDET/SIMAS then upload latest 6 months to raw_kcw",
    )
    ssi.set_defaults(func=cmd_sync_simas_sidet)

    ubc = sub.add_parser(
        "upload-brdet-bpdet",
        help="Drive raw_hq_brdet/bpdet CSVs -> raw_kcw (HQ cheque/transfer registers)",
    )
    ubc.set_defaults(func=cmd_upload_brdet_bpdet)

    sbc = sub.add_parser(
        "sync-brdet-bpdet",
        help=(
            "Extract HQ BRDET/BPDET (ทะเบียนเช็ครับ/จ่าย) then upload to raw_kcw. "
            "CHKNO is either a cheque number or a method label (โอน, KSHOP, …)."
        ),
    )
    sbc.set_defaults(func=cmd_sync_brdet_bpdet)

    t = sub.add_parser("tar", help="TAR/3TAR/CNTAR catch-up or single day")
    t.add_argument("--catch-up", action="store_true", help="Process all missing days (default if no --date)")
    t.add_argument("--date", help="Single day YYYY-MM-DD")
    t.add_argument("--start", help="Catch-up start override YYYY-MM-DD")
    t.add_argument("--end", help="Catch-up end override YYYY-MM-DD")
    t.add_argument("--force", action="store_true", help="Do not skip-if-done")
    t.add_argument(
        "--reprocess",
        metavar="YYYY-MM-DD",
        help="Delete fin_* for day then regenerate (explicit; does not rewind seq)",
    )
    t.set_defaults(func=cmd_tar)

    tr = sub.add_parser(
        "tar-report",
        help="Regenerate TAR/3TAR/CNTAR PDFs+CSVs from billgen.fin_* (does not re-number bills)",
    )
    tr.add_argument("--date", help="Single day YYYY-MM-DD (default: start/end or today)")
    tr.add_argument("--start", help="Range start YYYY-MM-DD")
    tr.add_argument("--end", help="Range end YYYY-MM-DD")
    tr.add_argument(
        "--prune-stale",
        action="store_true",
        default=True,
        help="Delete month-folder PDFs whose billno is no longer in fin_* (default)",
    )
    tr.add_argument(
        "--keep-stale",
        action="store_true",
        help="Leave leftover PDFs in the month folder",
    )
    tr.set_defaults(func=cmd_tar_report)

    bsr = sub.add_parser(
        "bank-statement-report",
        help=(
            "Monthly bank statement Excel (one sheet per account) "
            "under 04_outputs/04_Bank_Statement_Report"
        ),
    )
    bsr.add_argument("--year", type=int, help="Override reporting year (default: Bangkok today-10d)")
    bsr.add_argument("--month", type=int, help="Override reporting month 1-12")
    bsr.add_argument(
        "--output",
        help="Optional output .xlsx path (default: Drive 04_outputs/04_Bank_Statement_Report/...)",
    )
    bsr.add_argument(
        "--fixture-sample",
        action="store_true",
        help="Write a synthetic layout sample (no DB/Drive); default under logs/",
    )
    bsr.set_defaults(func=cmd_bank_statement_report)

    bsa = sub.add_parser(
        "backfill-statement-accounts",
        help=(
            "Re-read statement Excel on Drive; update account_no + fingerprints "
            "(HQ PC with Drive mounted; dry-run by default)"
        ),
    )
    bsa.add_argument(
        "--apply",
        action="store_true",
        help="Write updates to Supabase (default: dry-run only)",
    )
    bsa.set_defaults(func=cmd_backfill_statement_accounts)

    return p


def main(argv: list[str] | None = None) -> int:
    _setup_path()
    try:
        from dotenv import load_dotenv

        # Load repo .env even if cwd differs.
        from pathlib import Path as _Path

        load_dotenv(_Path(__file__).resolve().parents[2] / ".env")
        load_dotenv()  # also cwd .env if present
    except ImportError:
        print(
            "WARN: python-dotenv not installed; relying on process env only. "
            "pip install python-dotenv"
        )

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
