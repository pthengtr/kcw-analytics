# Shared by worker_tasks/linux/*.sh. Source only; do not execute.
# Windows HQ-PC never runs these. Logs stay on this box (not Drive FUSE).

set -euo pipefail

LINUX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$LINUX_DIR/../.." && pwd)"
cd "$REPO"

_env_get() {
  local key="$1"
  local line
  line="$(grep -E "^${key}=" "$REPO/.env" 2>/dev/null | head -n 1 || true)"
  printf '%s' "${line#*=}"
}

PY="${KCW_ANALYTICS_PYTHON:-$(_env_get KCW_ANALYTICS_PYTHON)}"
PY="${PY:-$REPO/.venv/bin/python}"
LOGDIR="${KCW_LINUX_JOB_LOG_DIR:-$REPO/logs}"
mkdir -p "$LOGDIR"

if [[ ! -x "$PY" && ! -f "$PY" ]]; then
  echo "Missing python: $PY" >&2
  exit 1
fi

run_nb() {
  local nbname="$1"
  local mode="${2:-fail}"
  local nb="$REPO/notebooks/$nbname"
  local stem="${nbname%.ipynb}"
  local log="$LOGDIR/${stem}.log"

  echo
  echo "------------------------------------------"
  echo "Running: $nbname"
  echo "------------------------------------------"

  if [[ ! -f "$nb" ]]; then
    echo "FAILED: notebook not found $nb"
    [[ "$mode" == "fail" ]] && return 1
    return 0
  fi

  if "$PY" -m jupyter nbconvert \
      --to notebook \
      --execute \
      --ExecutePreprocessor.kernel_name=python3 \
      --output-dir "$LOGDIR" \
      --output "${stem}.executed" \
      "$nb" > "$log" 2>&1; then
    echo "DONE: $nbname"
    return 0
  fi

  echo "FAILED: $nbname"
  echo "Check log: $log"
  tail -n 40 "$log" || true
  [[ "$mode" == "fail" ]] && return 1
  echo "Continue even if error"
  return 0
}
