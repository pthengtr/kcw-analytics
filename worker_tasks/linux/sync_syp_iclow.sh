#!/usr/bin/env bash
# Linux stand-in for run_syp_iclow_sync.bat
source "$(dirname "$0")/common.sh"
echo "SYP ICLOW sync"
"$PY" -m src.kcw.pipeline sync-iclow --site syp
echo "DONE: sync_iclow (syp)"
