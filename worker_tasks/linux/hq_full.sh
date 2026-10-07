#!/usr/bin/env bash
# Linux stand-in for run_hq_parts9_full_pipeline.bat (HQ B).
source "$(dirname "$0")/common.sh"

echo "=========================================="
echo "HQ full pipeline (raw + notebooks)"
echo "Python: $PY"
echo "Repo: $REPO"
echo "Logs: $LOGDIR"
echo "=========================================="

"$LINUX_DIR/hq_raw.sh"

run_nb "00_archive_output.ipynb" fail
run_nb "51_parts9_to_drive.ipynb" fail
echo
echo "Running: RV catch-up (CLI)"
RV_LOG="$LOGDIR/rv_catchup.log"
if "$PY" -m src.kcw.pipeline rv --catch-up > "$RV_LOG" 2>&1; then
  echo "DONE: RV catch-up CLI"
else
  echo "CLI RV catch-up failed"
  tail -n 40 "$RV_LOG" || true
  exit 1
fi

echo "Running: RV report (CLI)"
RV_REPORT_LOG="$LOGDIR/rv_report.log"
if "$PY" -m src.kcw.pipeline rv-report > "$RV_REPORT_LOG" 2>&1; then
  echo "DONE: RV report CLI"
else
  echo "CLI RV report failed"
  tail -n 40 "$RV_REPORT_LOG" || true
  exit 1
fi

echo
echo "Running: TAR catch-up (CLI)"
TAR_LOG="$LOGDIR/tar_catchup.log"
if "$PY" -m src.kcw.pipeline tar --catch-up > "$TAR_LOG" 2>&1; then
  echo "DONE: TAR catch-up CLI"
else
  echo "CLI TAR catch-up failed - falling back to notebook"
  tail -n 40 "$TAR_LOG" || true
  run_nb "21_tar_daily_supabase.ipynb" continue
fi

run_nb "21_tar_daily_report.ipynb" continue
run_nb "30_generate_bills_summary.ipynb" fail
run_nb "31_vat_purchase_report_excel.ipynb" fail
run_nb "32_vat_sales_report_excel.ipynb" fail
run_nb "33_ar_ap_report.ipynb" fail
run_nb "90_csv_to_supabase.ipynb" continue

echo
echo "ALL DONE."
