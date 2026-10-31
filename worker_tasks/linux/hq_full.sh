#!/usr/bin/env bash
# Linux stand-in for run_hq_parts9_full_pipeline.bat (HQ B).
source "$(dirname "$0")/common.sh"

echo "=========================================="
echo "HQ full pipeline (raw + notebooks)"
echo "Python: $PY"
echo "Repo: $REPO"
echo "Logs: $LOGDIR"
echo "=========================================="

KCW_API_DIR="${KCW_API_DIR:-$HOME/projects/kcw-api}"
KCW_API_PY="${KCW_API_PYTHON:-$KCW_API_DIR/.venv/bin/python}"
CURRENT_STEP="start"
STEP_LOG=""

notify_hq_daily() {
  if [[ ! -x "$KCW_API_PY" ]]; then
    echo "HQ daily LINE push skipped: missing $KCW_API_PY"
    return 0
  fi
  (
    cd "$KCW_API_DIR"
    "$KCW_API_PY" -m src.hq_daily_push "$@"
  ) || echo "HQ daily LINE push failed ($1)"
}

on_exit() {
  local code=$?
  trap - EXIT
  if [[ "$code" -eq 0 ]]; then
    notify_hq_daily ok || true
  else
    local -a args=(fail --code "$code" --step "$CURRENT_STEP")
    if [[ -n "${STEP_LOG:-}" ]]; then
      args+=(--log "$STEP_LOG")
    fi
    notify_hq_daily "${args[@]}" || true
  fi
  exit "$code"
}
trap on_exit EXIT

CURRENT_STEP="hq raw"
"$LINUX_DIR/hq_raw.sh"

CURRENT_STEP="00_archive_output.ipynb"
STEP_LOG="$LOGDIR/00_archive_output.log"
run_nb "00_archive_output.ipynb" fail
CURRENT_STEP="51_parts9_to_drive.ipynb"
STEP_LOG="$LOGDIR/51_parts9_to_drive.log"
run_nb "51_parts9_to_drive.ipynb" fail
echo
echo "Running: RV catch-up (CLI)"
CURRENT_STEP="rv catch-up"
RV_LOG="$LOGDIR/rv_catchup.log"
STEP_LOG="$RV_LOG"
if "$PY" -m src.kcw.pipeline rv --catch-up > "$RV_LOG" 2>&1; then
  echo "DONE: RV catch-up CLI"
else
  echo "CLI RV catch-up failed"
  tail -n 40 "$RV_LOG" || true
  exit 1
fi

echo "Running: RV report (CLI)"
CURRENT_STEP="rv report"
RV_REPORT_LOG="$LOGDIR/rv_report.log"
STEP_LOG="$RV_REPORT_LOG"
if "$PY" -m src.kcw.pipeline rv-report > "$RV_REPORT_LOG" 2>&1; then
  echo "DONE: RV report CLI"
else
  echo "CLI RV report failed"
  tail -n 40 "$RV_REPORT_LOG" || true
  exit 1
fi

echo
echo "Running: TAR catch-up (CLI)"
CURRENT_STEP="tar catch-up"
TAR_LOG="$LOGDIR/tar_catchup.log"
STEP_LOG="$TAR_LOG"
if "$PY" -m src.kcw.pipeline tar --catch-up > "$TAR_LOG" 2>&1; then
  echo "DONE: TAR catch-up CLI"
else
  echo "CLI TAR catch-up failed - falling back to notebook"
  tail -n 40 "$TAR_LOG" || true
  run_nb "21_tar_daily_supabase.ipynb" continue
fi

CURRENT_STEP="21_tar_daily_report.ipynb"
STEP_LOG="$LOGDIR/21_tar_daily_report.log"
run_nb "21_tar_daily_report.ipynb" continue

echo
echo "Running: VAT inventory rebuild (CLI)"
CURRENT_STEP="vat inventory"
VAT_INV_LOG="$LOGDIR/vat_inventory.log"
STEP_LOG="$VAT_INV_LOG"
if "$PY" -m src.kcw.pipeline vat-inventory > "$VAT_INV_LOG" 2>&1; then
  echo "DONE: VAT inventory"
else
  echo "CLI VAT inventory failed"
  tail -n 40 "$VAT_INV_LOG" || true
  exit 1
fi

CURRENT_STEP="30_generate_bills_summary.ipynb"
STEP_LOG="$LOGDIR/30_generate_bills_summary.log"
run_nb "30_generate_bills_summary.ipynb" fail
CURRENT_STEP="31_vat_purchase_report_excel.ipynb"
STEP_LOG="$LOGDIR/31_vat_purchase_report_excel.log"
run_nb "31_vat_purchase_report_excel.ipynb" fail
CURRENT_STEP="32_vat_sales_report_excel.ipynb"
STEP_LOG="$LOGDIR/32_vat_sales_report_excel.log"
run_nb "32_vat_sales_report_excel.ipynb" fail
CURRENT_STEP="33_ar_ap_report.ipynb"
STEP_LOG="$LOGDIR/33_ar_ap_report.log"
run_nb "33_ar_ap_report.ipynb" fail
CURRENT_STEP="90_csv_to_supabase.ipynb"
STEP_LOG="$LOGDIR/90_csv_to_supabase.log"
run_nb "90_csv_to_supabase.ipynb" continue

echo
echo "Running: FIFO movers (CLI)"
CURRENT_STEP="fifo movers"
FIFO_LOG="$LOGDIR/fifo_movers.log"
STEP_LOG="$FIFO_LOG"
if "$PY" -m src.kcw.pipeline fifo-movers --site hq > "$FIFO_LOG" 2>&1; then
  echo "DONE: FIFO movers"
else
  echo "CLI FIFO movers failed"
  tail -n 40 "$FIFO_LOG" || true
  exit 1
fi

echo
echo "ALL DONE."
