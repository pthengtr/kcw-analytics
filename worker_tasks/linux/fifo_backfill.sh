#!/usr/bin/env bash
# One-time FIFO history load. Not part of HQ full.
source "$(dirname "$0")/common.sh"

echo "=========================================="
echo "FIFO backfill (separate from HQ full)"
echo "Python: $PY"
echo "Repo: $REPO"
echo "=========================================="

"$PY" -m src.kcw.pipeline fifo-backfill --site hq

echo "ALL DONE."
