#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required." >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker is required. Install Docker Desktop or Docker Engine first." >&2; exit 1; }

exec python3 gui/app.py "$@"
