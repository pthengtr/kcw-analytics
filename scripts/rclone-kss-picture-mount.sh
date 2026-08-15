#!/usr/bin/env bash
# Mount KAcc9/PARTS9/Picture. Host list matches PARTS9 SQL; rclone gets the resolved IP
# because its DNS cannot look up .local / NetBIOS the way ODBC can.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
HOST_IP="$("$REPO/.venv/bin/python" "$REPO/scripts/pick_kss_host.py" --port 445)"
echo "KSS SMB rclone host: $HOST_IP (probed TCP 445)"
MOUNT="${HOME}/mnt/kss/PARTS9/Picture"
mkdir -p "$MOUNT"
exec "${HOME}/.local/bin/rclone" mount "kss:KAcc9/PARTS9/Picture" "$MOUNT" \
  --config "${HOME}/.config/rclone/rclone.conf" \
  --smb-host "$HOST_IP" \
  --vfs-cache-mode writes \
  --dir-cache-time 10s \
  --umask 022
