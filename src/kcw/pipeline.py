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

    extract_tables(args.site)
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
    e.set_defaults(func=cmd_extract)

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
