#!/usr/bin/env bash
# Linux stand-in for run_hq_parts9_to_drive_raw.bat (HQ A).
source "$(dirname "$0")/common.sh"

echo "=========================================="
echo "HQ A: SYP+HQ PARTS9 -> Drive raw + daily Supabase"
echo "Python: $PY"
echo "Repo: $REPO"
echo "=========================================="

# SYP extract from this box over Tailscale (kss-pc:1433). Do not wait for
# SYP Task Scheduler. kss-pc must be online or this step fails.
"$LINUX_DIR/syp_raw.sh"
"$PY" -m src.kcw.pipeline extract --site hq
"$PY" -m src.kcw.pipeline upload-daily-raw
echo "DONE: HQ A"
