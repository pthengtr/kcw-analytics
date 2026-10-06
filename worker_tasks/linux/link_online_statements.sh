#!/usr/bin/env bash
# Build marketplace payout → order → TAD links from statement/online.
source "$(dirname "$0")/common.sh"

echo "Link online statements"
"$PY" -m src.kcw.pipeline link-online-statements
echo "ONLINE_STATEMENT_LINK: DONE"
