#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$APP_DIR"

if [ ! -x .venv/bin/python ]; then
  echo "Missing .venv. Run setup_unix.sh first." >&2
  exit 1
fi

exec .venv/bin/python -m chem_use_web "$@"
