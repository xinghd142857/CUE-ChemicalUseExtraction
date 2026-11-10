#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$APP_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [ ! -f config.local.json ]; then
  cp config.example.json config.local.json
fi

echo "Setup complete."
echo "App folder: $APP_DIR"
echo "Edit config.local.json if you need model paths."
echo "Start with: bash scripts/run_unix.sh"
