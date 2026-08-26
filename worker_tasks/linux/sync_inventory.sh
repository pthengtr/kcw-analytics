#!/usr/bin/env bash
# Linux stand-in for run_inventory_sync.bat. Executed notebook stays in local logs/.
# Site: BRANCH / KCW_BRANCH → SYP uses PARTS9_SYP_* (kss-pc); HQ uses PARTS9_HQ_* (KSS).
source "$(dirname "$0")/common.sh"
BRANCH="${BRANCH:-$(_env_get BRANCH)}"
BRANCH="${BRANCH:-$(_env_get KCW_BRANCH)}"
BRANCH="$(printf '%s' "$BRANCH" | tr '[:lower:]' '[:upper:]')"
if [[ "$BRANCH" != "HQ" && "$BRANCH" != "SYP" ]]; then
  echo "Set BRANCH or KCW_BRANCH to HQ or SYP before inventory sync" >&2
  exit 1
fi
export BRANCH
export KCW_BRANCH="$BRANCH"
echo "Inventory sync (notebook 50) branch=$BRANCH"
run_nb "50_parts9_to_supabase.ipynb" fail
echo "DONE: sync_inventory"
