#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
echo "HQ ICMAS sync"
"$PY" -m src.kcw.pipeline sync-icmas --site hq
echo "DONE: sync_icmas"
