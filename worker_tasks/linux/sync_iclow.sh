#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
echo "HQ ICLOW sync"
"$PY" -m src.kcw.pipeline sync-iclow --site hq
echo "DONE: sync_iclow"
