#!/usr/bin/env bash
# HQ → SYP ICMAS product-master sync (dry-run by default).
# Usage:
#   sync_icmas_master.sh              # dry-run
#   sync_icmas_master.sh --apply      # write SYP
#   sync_icmas_master.sh --bcode 01010044 --limit 10
source "$(dirname "$0")/common.sh"
echo "HQ→SYP ICMAS master sync"
"$PY" -m src.kcw.pipeline sync-icmas-master "$@"
echo "DONE: sync_icmas_master"
