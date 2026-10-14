#!/usr/bin/env bash
set -Eeuo pipefail
LATIUM_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$LATIUM_REPO"
exec "${LATIUM_PYTHON:-python3}" -m src.submission "$@"
