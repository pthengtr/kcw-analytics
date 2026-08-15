#!/usr/bin/env bash
# SYP PARTS9 -> Drive raw_syp_*.csv from this HQ box (Tailscale kss-pc:1433).
# Daily set: PODET, POMAS, PIDET, PIMAS, SIDET, SIMAS, ICMAS, ICLOW.
source "$(dirname "$0")/common.sh"

echo "=========================================="
echo "SYP A: PARTS9 (kss-pc) -> Drive raw"
echo "Python: $PY"
echo "Repo: $REPO"
echo "=========================================="

"$PY" -m src.kcw.pipeline extract --site syp
echo "DONE: SYP extract"
