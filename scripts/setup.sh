#!/usr/bin/env bash
# One-shot local setup for Linux / macOS / WSL.
#   ./scripts/setup.sh            # dev install (tests, linters, pipeline extras)
#   ./scripts/setup.sh --runtime  # runtime-only install
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3}"

"$PYTHON" - <<'EOF'
import sys
if sys.version_info < (3, 10):
    sys.exit(f"Python 3.10+ is required, found {sys.version.split()[0]}")
EOF

echo "==> Creating virtual environment in .venv"
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

if [[ "$(uname -s)" == "Linux" ]]; then
  echo "==> Installing CPU-only PyTorch (skips ~2 GB of CUDA libraries)"
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
fi

if [[ "${1:-}" == "--runtime" ]]; then
  python -m pip install -r requirements.txt
else
  python -m pip install -r requirements-dev.txt
  [[ -d .git ]] && python -m pre_commit install || true
fi
python -m pip install -e . --no-deps

[[ -f .env ]] || cp .env.example .env
mkdir -p data artifacts logs

echo
echo "Done. Next steps (README Steps 4-8):"
echo "  source .venv/bin/activate"
echo "  python -m pytest"
echo "  python -m bookmind.pipeline.cli download"
echo "  python -m bookmind.pipeline.cli all --classifier keyword"
echo "  ./run.sh"
