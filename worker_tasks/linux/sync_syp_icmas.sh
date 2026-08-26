#!/usr/bin/env bash
# Linux stand-in for run_syp_icmas_sync.bat
source "$(dirname "$0")/common.sh"
echo "SYP ICMAS sync"
"$PY" -m src.kcw.pipeline sync-icmas --site syp
echo "DONE: sync_icmas (syp)"
