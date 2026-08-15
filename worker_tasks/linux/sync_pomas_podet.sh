#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
echo "HQ POMAS/PODET sync"
"$PY" -m src.kcw.pipeline sync-pomas-podet --site hq
echo "DONE: sync_pomas_podet"
