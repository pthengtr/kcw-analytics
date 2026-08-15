#!/usr/bin/env bash
# Linux stand-in for run_bank_statement_import.bat
source "$(dirname "$0")/common.sh"

echo "Daily bank sync"
"$LINUX_DIR/sync_brdet_bpdet.sh"

echo "Bank statement Drive -> Edge Function"
"$PY" "$REPO/scripts/upload_drive_bank_statements.py"

echo "Bank statement report"
"$PY" -m src.kcw.pipeline bank-statement-report
echo "DONE: daily bank sync"
