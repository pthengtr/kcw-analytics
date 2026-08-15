#!/usr/bin/env bash
# Linux stand-in for run_hq_parts9_to_drive_raw.bat (HQ A).
source "$(dirname "$0")/common.sh"

echo "=========================================="
echo "HQ A: PARTS9 -> Drive raw + daily Supabase"
echo "Python: $PY"
echo "Repo: $REPO"
echo "=========================================="

"$PY" -m src.kcw.pipeline extract --site hq
"$PY" -m src.kcw.pipeline upload-daily-raw
echo "DONE: HQ A"
