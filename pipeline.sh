#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DEFAULT_MODELS=(
  gpt2-large
  gpt2-xl
  qwen3-4b
  qwen3-8b
  mistral-7b-v0.1
  mistral-7b-v0.3
  falcon-7b
  opt-6.7b
  gpt-j-6b
  deepseek-7b-base
)

MODELS=("${DEFAULT_MODELS[@]}")
N=50
START_IDX=30
N_PROMPTS=50
REMOTE_HOST=""
REMOTE_DIR="~/Latium"
CONDA_ENV="latium"
OUTPUT_ROOT="./pipeline_out"
SETUP_ENV=false
COMPUTE_COV=false
COV_SOURCE_HOST="metju@kubapc"
COV_SOURCE_DIR="~/data/covariance_matrices"
TMUX_SESSION_GPU=""
TMUX_SESSION_LOCAL=""
SYNC_INTERVAL=60
STATUS_ONLY=false

usage() {
  cat <<'EOF'
Usage: bash pipeline.sh --remote <user@host> [options]

Split pipeline defaults:
  models:     gpt2-large gpt2-xl qwen3-4b qwen3-8b mistral-7b-v0.1 mistral-7b-v0.3 falcon-7b opt-6.7b gpt-j-6b deepseek-7b-base
  n-tests:    50
  start-idx:  30

Options:
  --remote <host>            Remote SSH host for H100 work
  --remote-dir <path>        Remote repo dir (default: ~/Latium)
  --conda-env <name>         Remote conda env (default: latium)
  --setup-env                Bootstrap remote conda env + dependencies
  --compute-cov              Compute only missing covariance files remotely
  --cov-source-host <host>   Fallback covariance SSH host (default: metju@kubapc)
  --cov-source-dir <path>    Fallback covariance dir (default: ~/data/covariance_matrices)
  --n <int>                  Number of CounterFact cases (default: 50)
  --start-idx <int>          CounterFact start index (default: 30)
  --n-prompts <int>          Number of prompts per benchmark evaluation (default: 50)
  --output-dir <path>        Local/remote output root (default: ./pipeline_out)
  --models <m1 m2 ...>       Override model list
  --tmux-session-gpu <name>  Remote tmux session name
  --tmux-session-local <n>   Local tmux session name
  --sync-interval <sec>      Local poll interval in seconds (default: 60)
  --status                   Show current remote/local tmux pane output
  --structural               Accepted for compatibility; ignored
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote) REMOTE_HOST="$2"; shift 2 ;;
    --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --setup-env) SETUP_ENV=true; shift ;;
    --compute-cov) COMPUTE_COV=true; shift ;;
    --cov-source-host) COV_SOURCE_HOST="$2"; shift 2 ;;
    --cov-source-dir) COV_SOURCE_DIR="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    --start-idx) START_IDX="$2"; shift 2 ;;
    --n-prompts) N_PROMPTS="$2"; shift 2 ;;
    --output-dir) OUTPUT_ROOT="$2"; shift 2 ;;
    --tmux-session-gpu) TMUX_SESSION_GPU="$2"; shift 2 ;;
    --tmux-session-local) TMUX_SESSION_LOCAL="$2"; shift 2 ;;
    --sync-interval) SYNC_INTERVAL="$2"; shift 2 ;;
    --models)
      shift
      MODELS=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        MODELS+=("$1")
        shift
      done
      ;;
    --status) STATUS_ONLY=true; shift ;;
    --structural) shift ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$REMOTE_HOST" ]]; then
  echo "ERROR: --remote is required" >&2
  usage >&2
  exit 1
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "ERROR: no models selected" >&2
  exit 1
fi

RUN_NAME="n${N}_s${START_IDX}"
LOCAL_RUN_DIR="$REPO_ROOT/${OUTPUT_ROOT#./}/$RUN_NAME"
REMOTE_REL_RUN_DIR="./${OUTPUT_ROOT#./}/$RUN_NAME"
TMUX_SESSION_GPU="${TMUX_SESSION_GPU:-latium_gpu_${RUN_NAME}}"
TMUX_SESSION_LOCAL="${TMUX_SESSION_LOCAL:-latium_local_${RUN_NAME}}"
COV_CACHE_DIR="$LOCAL_RUN_DIR/archive/covariance_cache"

SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=30
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=6
)

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command not found: $1" >&2
    exit 1
  }
}

python_supports_pipeline() {
  local py_bin="$1"
  [[ -x "$py_bin" ]] || return 1
  "$py_bin" - <<'PY' >/dev/null 2>&1
from omegaconf import OmegaConf
PY
}

resolve_pipeline_python() {
  local candidates=()
  if [[ -n "${LATIUM_PIPELINE_PYTHON:-}" ]]; then
    candidates+=("$LATIUM_PIPELINE_PYTHON")
  fi
  if command -v python >/dev/null 2>&1; then
    candidates+=("$(command -v python)")
  fi
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi
  candidates+=(
    "$REPO_ROOT/.venv/bin/python"
    "$HOME/miniforge/envs/latium/bin/python"
    "$HOME/miniconda3/envs/latium/bin/python"
    "$HOME/miniforge/envs/llms/bin/python"
    "$HOME/miniconda3/envs/llms/bin/python"
    "$HOME/miniforge/bin/python"
    "$HOME/miniconda3/bin/python"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if python_supports_pipeline "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  echo "ERROR: unable to find a local Python interpreter with OmegaConf for pipeline.sh" >&2
  return 1
}

quote_cmd() {
  printf '%q ' "$@"
}

ssh_cmd() {
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$@"
}

resolve_remote_path() {
  local host="$1"
  local raw_path="$2"
  local home
  if [[ "$raw_path" == "~" ]]; then
    home="$(ssh "${SSH_OPTS[@]}" "$host" 'printf %s "$HOME"')"
    printf '%s\n' "$home"
  elif [[ "$raw_path" == "~/"* ]]; then
    home="$(ssh "${SSH_OPTS[@]}" "$host" 'printf %s "$HOME"')"
    printf '%s/%s\n' "$home" "${raw_path:2}"
  else
    printf '%s\n' "$raw_path"
  fi
}

model_cov_basename() {
  MODEL_KEY="$1" "$PIPELINE_PYTHON" - <<'PY'
from pathlib import Path
from omegaconf import OmegaConf
import os

model_key = os.environ["MODEL_KEY"]
cfg = OmegaConf.load(Path("src/config/model") / f"{model_key}.yaml")
explicit = str(getattr(cfg, "second_moment_path", "") or "").strip()
if explicit:
    print(Path(explicit).name)
else:
    print(f"{str(cfg.name).replace('/', '_')}_{int(cfg.layer)}")
PY
}

local_cov_candidate() {
  local basename="$1"
  local candidates=(
    "$REPO_ROOT/second_moment_stats/$basename"
    "$REPO_ROOT/data/second_moment_stats/$basename"
    "$REPO_ROOT/../reimagined/second_moment_stats/$basename"
    "$REPO_ROOT/../reimagined/data/second_moment_stats/$basename"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

show_status() {
  echo "== Local tmux: $TMUX_SESSION_LOCAL =="
  tmux capture-pane -t "$TMUX_SESSION_LOCAL" -p 2>/dev/null || echo "No active local session."
  echo
  echo "== Remote tmux: $TMUX_SESSION_GPU =="
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "tmux capture-pane -t '$TMUX_SESSION_GPU' -p 2>/dev/null" || echo "No active remote session."
}

require_cmd ssh
require_cmd rsync
require_cmd tmux
require_cmd python

PIPELINE_PYTHON="$(resolve_pipeline_python)"

if $STATUS_ONLY; then
  show_status
  exit 0
fi

REMOTE_DIR_ABS="$(resolve_remote_path "$REMOTE_HOST" "$REMOTE_DIR")"
REMOTE_RUN_DIR_ABS="$REMOTE_DIR_ABS/${REMOTE_REL_RUN_DIR#./}"
COV_SOURCE_DIR_ABS="$(resolve_remote_path "$COV_SOURCE_HOST" "$COV_SOURCE_DIR")"

mkdir -p "$LOCAL_RUN_DIR/archive" "$COV_CACHE_DIR"

echo "========================================"
echo " Latium Split Pipeline"
echo " models=${#MODELS[@]} | n=$N | start_idx=$START_IDX | compute_cov=$COMPUTE_COV"
echo " counterfact=iterating_per_model stride=$N"
echo " remote=$REMOTE_HOST:$REMOTE_DIR_ABS"
echo " remote_run=$REMOTE_RUN_DIR_ABS"
echo " local_run=$LOCAL_RUN_DIR"
echo " tmux_remote=$TMUX_SESSION_GPU | tmux_local=$TMUX_SESSION_LOCAL"
echo "========================================"

echo "[1/7] Ensuring remote directories exist ..."
ssh_cmd "mkdir -p $(printf '%q' "$REMOTE_DIR_ABS") $(printf '%q' "$REMOTE_RUN_DIR_ABS") $(printf '%q' "$REMOTE_DIR_ABS/second_moment_stats")"

echo "[2/7] Syncing repository to remote ..."
rsync -az --delete \
  -e "ssh ${SSH_OPTS[*]}" \
  --exclude='.git/' \
  --exclude='.pytest_cache/' \
  --exclude='__pycache__/' \
  --exclude='.venv/' \
  --exclude='venv/' \
  --exclude='analysis_out/' \
  --exclude='pipeline_out/' \
  --exclude='datasets/' \
  --exclude='outputs/' \
  --exclude='second_moment_stats/' \
  --exclude='data/second_moment_stats/' \
  ./ "$REMOTE_HOST:$REMOTE_DIR_ABS/"

if $SETUP_ENV; then
  echo "[3/7] Bootstrapping remote environment ..."
  {
    printf '%s\n' 'set -euo pipefail'
    printf 'REMOTE_DIR_ABS=%q\n' "$REMOTE_DIR_ABS"
    printf 'CONDA_ENV=%q\n' "$CONDA_ENV"
    cat <<'EOF'
if command -v sudo >/dev/null 2>&1; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -y >/dev/null
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y tmux rsync git wget curl >/dev/null
fi

if [[ ! -x "$HOME/miniconda3/bin/conda" ]]; then
  if command -v wget >/dev/null 2>&1; then
    wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  else
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
  fi
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi

# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
conda config --set auto_activate_base false >/dev/null 2>&1 || true

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda create -y -n "$CONDA_ENV" python=3.11 >/dev/null
fi

cd "$REMOTE_DIR_ABS"
conda run -n "$CONDA_ENV" python -m pip install --upgrade pip wheel >/dev/null
conda run -n "$CONDA_ENV" python -m pip install -r requirements.txt >/dev/null
conda run -n "$CONDA_ENV" python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio >/dev/null
conda run -n "$CONDA_ENV" python - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
print(torch.__version__)
print(torch.cuda.get_device_name(0))
PY
EOF
  } | ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" 'bash -s'
else
  echo "[3/7] Remote env setup skipped (--setup-env not set)"
fi

echo "[4/7] Staging covariance files ..."
for model in "${MODELS[@]}"; do
  basename="$(model_cov_basename "$model")"
  remote_target="$REMOTE_DIR_ABS/second_moment_stats/$basename"

  if ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "[ -f $(printf '%q' "$remote_target") ]"; then
    echo "  [$model] already present on remote: $basename"
    continue
  fi

  if local_path="$(local_cov_candidate "$basename" 2>/dev/null)"; then
    echo "  [$model] local stage -> $basename"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "$local_path" "$REMOTE_HOST:$remote_target"
    continue
  fi

  if rsync -az -e "ssh ${SSH_OPTS[*]}" "$COV_SOURCE_HOST:$COV_SOURCE_DIR_ABS/$basename" "$COV_CACHE_DIR/$basename" >/dev/null 2>&1; then
    echo "  [$model] kubapc fallback -> $basename"
    rsync -az -e "ssh ${SSH_OPTS[*]}" "$COV_CACHE_DIR/$basename" "$REMOTE_HOST:$remote_target"
    continue
  fi

  if $COMPUTE_COV; then
    echo "  [$model] no staged covariance found; remote runner will compute it"
    continue
  fi

  echo "ERROR: covariance not found for $model ($basename)" >&2
  exit 1
done

echo "[5/7] Launching remote GPU tmux session ..."
REMOTE_CMD=(bash scripts/remote_gpu_structural.sh --run-root "$REMOTE_REL_RUN_DIR" --n-tests "$N" --start-idx "$START_IDX" --n-prompts "$N_PROMPTS" --conda-env "$CONDA_ENV" --models "${MODELS[@]}")
if $COMPUTE_COV; then
  REMOTE_CMD=(bash scripts/remote_gpu_structural.sh --run-root "$REMOTE_REL_RUN_DIR" --n-tests "$N" --start-idx "$START_IDX" --n-prompts "$N_PROMPTS" --compute-cov --conda-env "$CONDA_ENV" --models "${MODELS[@]}")
fi
REMOTE_CMD_STR="$(quote_cmd "${REMOTE_CMD[@]}")"
HF_TOKEN_B64="$(printf '%s' "${HF_TOKEN:-}" | base64 -w0 2>/dev/null || printf '%s' "${HF_TOKEN:-}" | base64 | tr -d '\n')"
COV_SOURCE_NOTE="local->../reimagined->${COV_SOURCE_HOST}:${COV_SOURCE_DIR_ABS}"

{
  printf '%s\n' 'set -euo pipefail'
  printf 'REMOTE_DIR_ABS=%q\n' "$REMOTE_DIR_ABS"
  printf 'TMUX_SESSION_GPU=%q\n' "$TMUX_SESSION_GPU"
  printf 'CONDA_ENV=%q\n' "$CONDA_ENV"
  printf 'HF_TOKEN_B64=%q\n' "$HF_TOKEN_B64"
  printf 'COV_SOURCE_NOTE=%q\n' "$COV_SOURCE_NOTE"
  printf 'REMOTE_CMD_STR=%q\n' "$REMOTE_CMD_STR"
  cat <<'EOF'
cd "$REMOTE_DIR_ABS"
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null || true
fi
HF_TOKEN="$(printf '%s' "$HF_TOKEN_B64" | base64 -d 2>/dev/null || true)"
export HF_TOKEN
export COV_SOURCE_NOTE
tmux kill-session -t "$TMUX_SESSION_GPU" 2>/dev/null || true
tmux new-session -d -s "$TMUX_SESSION_GPU" "cd \"$REMOTE_DIR_ABS\" && $REMOTE_CMD_STR"
EOF
} | ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" 'bash -s'

echo "[6/7] Launching local post-hoc tmux session ..."
LOCAL_CMD=(
  bash scripts/local_posthoc_runner.sh
  --remote-host "$REMOTE_HOST"
  --remote-run-dir "$REMOTE_RUN_DIR_ABS"
  --local-run-dir "$LOCAL_RUN_DIR"
  --n-tests "$N"
  --start-idx "$START_IDX"
  --sync-interval "$SYNC_INTERVAL"
  --composite-window-sweep
  --success-file "$REPO_ROOT/success.txt"
  --models "${MODELS[@]}"
)
LOCAL_CMD_STR="$(quote_cmd "${LOCAL_CMD[@]}")"
tmux kill-session -t "$TMUX_SESSION_LOCAL" 2>/dev/null || true
tmux new-session -d -s "$TMUX_SESSION_LOCAL" "cd \"$REPO_ROOT\" && $LOCAL_CMD_STR"

echo "[7/7] Sessions started"
echo
echo "Remote attach: ssh ${REMOTE_HOST} -t tmux attach -t ${TMUX_SESSION_GPU}"
echo "Local attach:  tmux attach -t ${TMUX_SESSION_LOCAL}"
echo "Status:        bash pipeline.sh --remote ${REMOTE_HOST} --remote-dir ${REMOTE_DIR} --output-dir ${OUTPUT_ROOT} --n ${N} --start-idx ${START_IDX} --tmux-session-gpu ${TMUX_SESSION_GPU} --tmux-session-local ${TMUX_SESSION_LOCAL} --status"
echo "CounterFact:   first model starts at ${START_IDX}, then +${N} per model"
echo "Run root:      $LOCAL_RUN_DIR"
