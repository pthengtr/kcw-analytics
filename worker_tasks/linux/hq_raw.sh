#!/usr/bin/env bash
# Linux stand-in for run_hq_parts9_to_drive_raw.bat (HQ A).
source "$(dirname "$0")/common.sh"

echo "=========================================="
echo "HQ A: SYP+HQ PARTS9 -> Drive raw + daily Supabase"
echo "Python: $PY"
echo "Repo: $REPO"
echo "=========================================="

# SYP extract from this box over Tailscale (kss-pc:1433). Do not wait for
# SYP Task Scheduler. kss-pc must be online Mon–Sat or this step fails
# (operator sees HQ B failed / summary not generated). Sundays kss-pc is
# powered off — skip SYP and reuse existing raw_syp_* on Drive so HQ B continues.
# Override: KCW_SKIP_SYP=1 always skip; KCW_FORCE_SYP=1 never skip (even Sunday).
_dow="$(TZ=Asia/Bangkok date +%u)" # 1=Mon … 7=Sun
if [[ "${KCW_FORCE_SYP:-0}" == "1" ]]; then
  "$LINUX_DIR/syp_raw.sh"
elif [[ "${KCW_SKIP_SYP:-0}" == "1" || "$_dow" == "7" ]]; then
  echo "SKIP SYP extract (Sunday / KCW_SKIP_SYP) — reusing existing raw_syp_* on Drive"
else
  "$LINUX_DIR/syp_raw.sh"
fi
"$PY" -m src.kcw.pipeline extract --site hq
"$PY" -m src.kcw.pipeline upload-daily-raw
echo "DONE: HQ A"
