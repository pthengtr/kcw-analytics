#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
echo "HQ BRDET/BPDET sync"
"$PY" -m src.kcw.pipeline sync-brdet-bpdet
echo "DONE: sync_brdet_bpdet"
