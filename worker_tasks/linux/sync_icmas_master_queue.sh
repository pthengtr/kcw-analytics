#!/usr/bin/env bash
# Drain HQ ICMAS_MASTER_SYNC_QUEUE → push claimed BCODEs to SYP.
# Intended for frequent systemd timer (every 1–5 min) on hq-ubuntu-server.
source "$(dirname "$0")/common.sh"
echo "HQ→SYP ICMAS master queue drain"
"$PY" -m src.kcw.pipeline sync-icmas-master-queue "$@"
echo "DONE: sync_icmas_master_queue"
