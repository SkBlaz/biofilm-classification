#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
command -v docker >/dev/null 2>&1 || { echo "Docker is required. Install Docker Desktop or Docker Engine first." >&2; exit 1; }

if ! docker info >/dev/null 2>&1; then
  echo "Docker is installed but its engine is not available. Start Docker and retry." >&2
  exit 1
fi

IMAGE_NAME="${MICROICS_IMAGE:-microics}"
PORT="${MICROICS_PORT:-8765}"

echo "Building $IMAGE_NAME..." >&2
docker build -t "$IMAGE_NAME" .
echo "MicroICS will be available at http://localhost:$PORT" >&2
exec docker run --rm -p "$PORT:8765" "$IMAGE_NAME"
