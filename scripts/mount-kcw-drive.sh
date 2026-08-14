#!/usr/bin/env bash
# Mount Google Shared drives at ~/mnt/gdrive (rclone remote: kcw).
set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
MOUNT="${HOME}/mnt/gdrive"
mkdir -p "${MOUNT}"
if mountpoint -q "${MOUNT}"; then
  echo "Already mounted: ${MOUNT}"
  exit 0
fi
# Shared drive KCW-Data (not My Drive)
DATA="${MOUNT}/KCW-Data"
mkdir -p "${DATA}"
if mountpoint -q "${DATA}"; then
  echo "Already mounted: ${DATA}"
  ls "${DATA}" | head
  exit 0
fi
rclone mount "kcw,team_drive=0AJ5BTDhgit7-Uk9PVA:" "${DATA}" \
  --daemon \
  --vfs-cache-mode writes \
  --dir-cache-time 10s
echo "Mounted KCW-Data -> ${DATA}"
ls "${DATA}" | head
