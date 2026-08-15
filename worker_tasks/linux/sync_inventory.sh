#!/usr/bin/env bash
# Linux stand-in for run_inventory_sync.bat. Executed notebook stays in local logs/.
source "$(dirname "$0")/common.sh"
echo "Inventory sync (notebook 50)"
run_nb "50_parts9_to_supabase.ipynb" fail
echo "DONE: sync_inventory"
