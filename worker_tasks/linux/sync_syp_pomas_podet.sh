#!/usr/bin/env bash
# Linux stand-in for run_syp_pomas_podet_sync.bat
source "$(dirname "$0")/common.sh"
echo "SYP POMAS/PODET sync"
"$PY" -m src.kcw.pipeline sync-pomas-podet --site syp
echo "DONE: sync_pomas_podet (syp)"
