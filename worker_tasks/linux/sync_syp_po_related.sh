#!/usr/bin/env bash
# Linux stand-in for run_syp_po_related_sync.bat
source "$(dirname "$0")/common.sh"

SKIP_INVENTORY=0
if [[ "${1:-}" == "--skip-inventory" ]]; then
  SKIP_INVENTORY=1
fi

echo "SYP PO-related sync (POMAS/PODET + ICLOW + inventory)"
"$PY" -m src.kcw.pipeline sync-po-related --site syp
echo "DONE: PO/ICLOW (syp)"

if [[ "$SKIP_INVENTORY" -eq 0 ]]; then
  echo "Inventory after PO-related"
  "$LINUX_DIR/sync_inventory.sh"
fi
echo "ALL DONE: SYP PO-related sync"
