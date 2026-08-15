#!/usr/bin/env bash
# Linux stand-in for run_online_sales_sync.bat (one notebook at a time).
source "$(dirname "$0")/common.sh"

FAILED=0
echo "Online sales sync"
run_nb "71_online_shopee.ipynb" fail || FAILED=1
run_nb "72_online_lazada.ipynb" fail || FAILED=1
run_nb "73_online_tiktok.ipynb" fail || FAILED=1

if [[ "$FAILED" -eq 0 ]]; then
  echo "ONLINE_SYNC_RESULT: ALL_OK"
  exit 0
fi
echo "ONLINE_SYNC_RESULT: DONE_WITH_FAILURES"
exit 1
