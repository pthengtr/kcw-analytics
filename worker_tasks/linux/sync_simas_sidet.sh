#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
echo "HQ SIMAS/SIDET sync"
"$PY" -m src.kcw.pipeline sync-simas-sidet --site hq
echo "DONE: sync_simas_sidet"
