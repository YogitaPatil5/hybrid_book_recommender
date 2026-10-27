#!/usr/bin/env bash
# BookMind Pro — one-command launcher for macOS / Linux:  ./run.sh  [--port 9000]
set -euo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "[BookMind] Virtual environment not found. Run ./scripts/setup.sh first." >&2
  exit 1
fi
if [[ ! -f artifacts/manifest.json ]]; then
  echo "[BookMind] Search index not built yet. Building it now (one-time)..."
  "$PY" -m bookmind.pipeline.cli download
  "$PY" -m bookmind.pipeline.cli all --classifier keyword
fi
exec "$PY" -m bookmind --open "$@"
