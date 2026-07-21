#!/usr/bin/env bash
set -euo pipefail

if [ "${EUID:-$(id -u)}" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
  echo "Do not start the MicroICS GUI with sudo; run: bash run_gui.sh" >&2
  echo "If Docker access is denied, add your user to the docker group and sign in again." >&2
  exit 1
fi

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required." >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker is required. Install Docker Desktop or Docker Engine first." >&2; exit 1; }

if ! docker info >/dev/null 2>&1; then
  DOCKER_USER="${USER:-$(id -un)}"
  if command -v getent >/dev/null 2>&1 && command -v sg >/dev/null 2>&1 \
    && getent group docker | cut -d: -f4 | tr ',' '\n' | grep -Fxq "$DOCKER_USER" \
    && sg docker -c 'docker info >/dev/null 2>&1'; then
    printf -v GUI_ARGS ' %q' "$@"
    printf -v QUOTED_ROOT '%q' "$ROOT_DIR"
    echo "Refreshing Docker group membership for this shell." >&2
    exec sg docker -c "cd $QUOTED_ROOT && exec python3 gui/app.py$GUI_ARGS"
  fi
  echo "Docker is installed but is not available to this user." >&2
  echo "Start Docker Desktop/Engine, then run: bash run_gui.sh" >&2
  if [ -S /var/run/docker.sock ] && [ ! -r /var/run/docker.sock ]; then
    echo "On Linux, fix access with: sudo usermod -aG docker $USER" >&2
    echo "Then sign out and in before retrying." >&2
  fi
  exit 1
fi

exec python3 gui/app.py "$@"
