#!/usr/bin/env bash
set -euo pipefail
REPO="${HQ_KCW_ANALYTIC_DIR:-$HOME/projects/kcw-analytic}"
PY="${REPO}/.venv/bin/python"
cd "$REPO"
git fetch origin
git reset --hard origin/master
if [[ -x "$PY" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$PY" -r requirements.txt
  else
    "$PY" -m pip install -r requirements.txt
  fi
fi
echo "kcw-analytic updated at $(git rev-parse --short HEAD). Next worker job uses new scripts."
