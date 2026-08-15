#!/usr/bin/env bash
# Linux stand-in for run_hq_po_related_sync.bat
source "$(dirname "$0")/common.sh"

SKIP_INVENTORY=0
if [[ "${1:-}" == "--skip-inventory" ]]; then
  SKIP_INVENTORY=1
fi

echo "HQ PO-related sync (POMAS/PODET + ICLOW + SIDET/SIMAS)"
"$PY" -m src.kcw.pipeline sync-po-related --site hq
echo "DONE: PO/ICLOW/sales"

if [[ "$SKIP_INVENTORY" -eq 0 ]]; then
  echo "Inventory after PO-related"
  "$LINUX_DIR/sync_inventory.sh"
fi
echo "ALL DONE: HQ PO-related sync"
