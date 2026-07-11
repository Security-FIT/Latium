#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: jobs/setup_env.sh [ENV_PREFIX]

Create/update a Python environment for Latium. Run this from the repository in
an interactive MetaCentrum job with network access. ENV_PREFIX defaults to
$HOME/.conda/envs/latium.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${1:-$HOME/.conda/envs/latium}"

if command -v module >/dev/null 2>&1; then
  module add mambaforge
fi

if command -v mamba >/dev/null 2>&1; then
  if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    mamba create --yes --prefix "$ENV_PREFIX" python=3.11 pip
  fi
elif command -v conda >/dev/null 2>&1; then
  if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    conda create --yes --prefix "$ENV_PREFIX" python=3.11 pip
  fi
else
  echo "ERROR: mamba/conda is unavailable. On MetaCentrum run: module add mambaforge" >&2
  exit 1
fi

"$ENV_PREFIX/bin/python" -m pip install --upgrade pip wheel
"$ENV_PREFIX/bin/python" -m pip install -r "$ROOT/requirements.txt"

"$ENV_PREFIX/bin/python" - <<'PY'
import torch

print("torch", torch.__version__)
print("torch CUDA build", torch.version.cuda)
print("CUDA visible now", torch.cuda.is_available())
PY

echo
echo "Environment ready: $ENV_PREFIX"
echo "Set LATIUM_ENV=$ENV_PREFIX in $ROOT/jobs/local.env"
