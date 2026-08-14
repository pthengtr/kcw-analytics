#!/usr/bin/env bash
# Mount Google Shared drive KCW-Data at ~/mnt/gdrive/KCW-Data (rclone remote: kcw).
# Prefers the user systemd unit so the mount survives reboot (needs linger).
set -euo pipefail
export PATH="${HOME}/.local/bin:${PATH}"
DATA="${HOME}/mnt/gdrive/KCW-Data"
UNIT="rclone-kcw-data.service"
USER_UNIT="${HOME}/.config/systemd/user/${UNIT}"
REPO_UNIT="$(cd "$(dirname "$0")" && pwd)/rclone-kcw-data.service"

mkdir -p "${DATA}" "${HOME}/.config/systemd/user"

if [[ -f "${REPO_UNIT}" ]] && ! cmp -s "${REPO_UNIT}" "${USER_UNIT}" 2>/dev/null; then
  cp "${REPO_UNIT}" "${USER_UNIT}"
  systemctl --user daemon-reload
fi

if [[ -f "${USER_UNIT}" ]]; then
  systemctl --user enable "${UNIT}" >/dev/null
  if mountpoint -q "${DATA}"; then
    if systemctl --user is-active --quiet "${UNIT}"; then
      echo "Already mounted by ${UNIT}: ${DATA}"
      ls "${DATA}" | head
      exit 0
    fi
    echo "Unmounting leftover rclone daemon so systemd can own the mount..."
    fusermount3 -u "${DATA}" || fusermount -u "${DATA}"
  fi
  systemctl --user start "${UNIT}"
  echo "Mounted KCW-Data via ${UNIT} -> ${DATA}"
  ls "${DATA}" | head
  exit 0
fi

if mountpoint -q "${DATA}"; then
  echo "Already mounted: ${DATA}"
  ls "${DATA}" | head
  exit 0
fi

rclone mount "kcw,team_drive=0AJ5BTDhgit7-Uk9PVA:" "${DATA}" \
  --daemon \
  --vfs-cache-mode writes \
  --dir-cache-time 10s
echo "Mounted KCW-Data (rclone --daemon, not persistent) -> ${DATA}"
ls "${DATA}" | head
