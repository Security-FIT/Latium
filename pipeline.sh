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
ANALYSIS_PROFILE="paper"
SLICE_POLICY="iterating_per_model"
REMOTE_HOST=""
REMOTE_DIR="~/Latium"
CONDA_ENV="latium"
OUTPUT_ROOT="./pipeline_out"
RUN_NAME=""
SETUP_ENV=false
COMPUTE_COV=false
RUN_STRUCTURAL=false
BUNDLE_GRAPHS=false
BUNDLE_ROOT="final_n500_bundle"
COV_SOURCE_HOST="metju@kubapc"
COV_SOURCE_DIR="~/data/covariance_matrices"
TMUX_SESSION_GPU=""
TMUX_SESSION_LOCAL=""
SYNC_INTERVAL=60
STATUS_ONLY=false
CASE_INDEX_FILE=""
RAW_SPECTRAL_MAX_TOP_K=""
PROGRESS_INTERVAL=10
WORKER_ID=""

usage() {
  cat <<'EOF'
Usage: bash pipeline.sh [--remote <user@host>] [options]

Pipeline defaults:
  models:     gpt2-large gpt2-xl qwen3-4b qwen3-8b mistral-7b-v0.1 mistral-7b-v0.3 falcon-7b opt-6.7b gpt-j-6b deepseek-7b-base
  n-tests:    50
  start-idx:  30

Modes:
  default                      local ROME-only benchmark via rome_benchmark.py
  --structural                 structural_benchmark.py + local post-hoc graph processing
  --remote <host>              orchestrate the selected mode on a remote GPU host

Options:
  --remote <host>            Optional SSH host for remote GPU orchestration
  --remote-dir <path>        Remote repo dir (default: ~/Latium)
  --conda-env <name>         Remote conda env (default: latium)
  --setup-env                Bootstrap remote conda env + dependencies
  --compute-cov              Compute missing covariance files on the runner
  --cov-source-host <host>   Fallback covariance SSH host (default: metju@kubapc)
  --cov-source-dir <path>    Fallback covariance dir (default: ~/data/covariance_matrices)
  --n <int>                  Number of CounterFact cases (default: 50)
  --start-idx <int>          CounterFact start index (default: 30)
  --n-prompts <int>          Number of prompts per benchmark evaluation (default: 50)
  --slice-policy <mode>      CounterFact assignment: iterating_per_model or shared (default: iterating_per_model)
  --analysis-profile <name>  Structural benchmark payload: raw, paper, or full (default: paper)
  --posthoc-only             Alias for --analysis-profile paper
  --detection-only           Alias for --analysis-profile paper
  --output-dir <path>        Local/remote output root (default: ./pipeline_out)
  --run-name <name>          Run directory name under --output-dir
  --case-index-file <path>   Local JSON manifest of explicit CounterFact indices shared across models
  --raw-spectral-max-top-k   Stored raw spectral top-k cap for reusable structural JSON
  --progress-interval <int>  Worker progress update cadence in completed cases (default: 10)
  --worker-id <name>         Human-readable worker label stored in worker_progress.txt
  --bundle-graphs            After structural post-hoc, rebuild graphs from --bundle-root
  --bundle-root <path>       final_n500_bundle root for --bundle-graphs (default: final_n500_bundle)
  --rome-only                Explicitly select the default ROME-only mode
  --models <m1 m2 ...>       Override model list
  --tmux-session-gpu <name>  Remote tmux session name
  --tmux-session-local <n>   Local tmux session name
  --sync-interval <sec>      Local poll interval in seconds (default: 60)
  --status                   Show current remote/local tmux pane output
  --structural               Run structural benchmark instead of the ROME-only benchmark
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
    --slice-policy) SLICE_POLICY="$2"; shift 2 ;;
    --analysis-profile) ANALYSIS_PROFILE="$2"; shift 2 ;;
    --posthoc-only|--detection-only) ANALYSIS_PROFILE="paper"; shift ;;
    --output-dir) OUTPUT_ROOT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --case-index-file) CASE_INDEX_FILE="$2"; shift 2 ;;
    --raw-spectral-max-top-k) RAW_SPECTRAL_MAX_TOP_K="$2"; shift 2 ;;
    --progress-interval) PROGRESS_INTERVAL="$2"; shift 2 ;;
    --worker-id) WORKER_ID="$2"; shift 2 ;;
    --bundle-graphs) BUNDLE_GRAPHS=true; shift ;;
    --bundle-root) BUNDLE_ROOT="$2"; shift 2 ;;
    --rome-only) RUN_STRUCTURAL=false; shift ;;
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
    --structural) RUN_STRUCTURAL=true; shift ;;
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

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "ERROR: no models selected" >&2
  exit 1
fi

if [[ "$SLICE_POLICY" != "iterating_per_model" && "$SLICE_POLICY" != "shared" ]]; then
  echo "ERROR: unsupported --slice-policy: $SLICE_POLICY" >&2
  exit 1
fi

if ! $RUN_STRUCTURAL; then
  if [[ -n "$CASE_INDEX_FILE" ]]; then
    echo "ERROR: --case-index-file is only supported with --structural" >&2
    exit 1
  fi
  if [[ -n "$RAW_SPECTRAL_MAX_TOP_K" ]]; then
    echo "ERROR: --raw-spectral-max-top-k is only supported with --structural" >&2
    exit 1
  fi
  if $BUNDLE_GRAPHS; then
    echo "ERROR: --bundle-graphs is only supported with --structural" >&2
    exit 1
  fi
fi

if [[ -z "$REMOTE_HOST" && "$STATUS_ONLY" == true ]]; then
  echo "ERROR: --status requires --remote because local runs are synchronous" >&2
  exit 1
fi

if [[ -z "$REMOTE_HOST" && "$SETUP_ENV" == true ]]; then
  echo "ERROR: --setup-env requires --remote" >&2
  exit 1
fi

if [[ -z "$RUN_NAME" ]]; then
  if [[ -n "$CASE_INDEX_FILE" ]]; then
    RUN_NAME="n${N}_manifest"
  else
    RUN_NAME="n${N}_s${START_IDX}"
  fi
fi
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

sync_git_tracked_and_unignored() {
  git ls-files --cached --others --exclude-standard -z |
    while IFS= read -r -d '' relpath; do
      if [[ -e "$REPO_ROOT/$relpath" ]]; then
        printf '%s\0' "$relpath"
      fi
    done |
    rsync -0az \
      -e "ssh ${SSH_OPTS[*]}" \
      --files-from=- \
      ./ "$REMOTE_HOST:$REMOTE_DIR_ABS/"
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
import os
from src.model_config import second_moment_basename

print(second_moment_basename(os.environ["MODEL_KEY"]))
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

require_cmd python
if [[ -n "$REMOTE_HOST" ]]; then
  require_cmd ssh
  require_cmd rsync
  require_cmd tmux
fi

PIPELINE_PYTHON="$(resolve_pipeline_python)"

if [[ -n "$CASE_INDEX_FILE" ]]; then
  if [[ ! -f "$CASE_INDEX_FILE" ]]; then
    echo "ERROR: --case-index-file not found: $CASE_INDEX_FILE" >&2
    exit 1
  fi
  CASE_INDEX_FILE="$("$PIPELINE_PYTHON" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$CASE_INDEX_FILE")"
fi

if $STATUS_ONLY; then
  show_status
  exit 0
fi

LOCAL_SUCCESS_FILE="$LOCAL_RUN_DIR/success.txt"
mkdir -p "$LOCAL_RUN_DIR/archive" "$COV_CACHE_DIR"
if [[ -n "$CASE_INDEX_FILE" ]]; then
  mkdir -p "$LOCAL_RUN_DIR/manifest"
  install -m 0644 "$CASE_INDEX_FILE" "$LOCAL_RUN_DIR/manifest/$(basename "$CASE_INDEX_FILE")"
fi
if [[ -z "$WORKER_ID" ]]; then
  if [[ ${#MODELS[@]} -eq 1 ]]; then
    WORKER_ID="${MODELS[0]}"
  else
    WORKER_ID="$RUN_NAME"
  fi
fi

stage_local_covariances() {
  echo "Staging covariance files locally ..."
  mkdir -p "$REPO_ROOT/second_moment_stats" "$COV_CACHE_DIR"
  local model basename target local_path
  for model in "${MODELS[@]}"; do
    basename="$(model_cov_basename "$model")"
    target="$REPO_ROOT/second_moment_stats/$basename"

    if [[ -f "$target" ]]; then
      echo "  [$model] already present: $basename"
      continue
    fi

    if local_path="$(local_cov_candidate "$basename" 2>/dev/null)"; then
      echo "  [$model] local stage -> $basename"
      cp -f "$local_path" "$target"
      continue
    fi

    if $COMPUTE_COV; then
      echo "  [$model] no staged covariance found; runner will compute it"
      continue
    fi

    echo "ERROR: covariance not found for $model ($basename)" >&2
    echo "       Re-run with --compute-cov to build missing covariance files locally." >&2
    exit 1
  done
}

run_bundle_graphs_if_requested() {
  if $BUNDLE_GRAPHS; then
    bash scripts/bundle_graphs/run_all_graphs.sh --bundle-root "$BUNDLE_ROOT"
  fi
}

run_local_pipeline() {
  stage_local_covariances

  echo "========================================"
  echo " Latium Local Pipeline"
  echo " mode=$([ "$RUN_STRUCTURAL" = true ] && printf '%s' 'structural' || printf '%s' 'rome-only')"
  echo " models=${#MODELS[@]} | n=$N | start_idx=$START_IDX | compute_cov=$COMPUTE_COV"
  if [[ "$SLICE_POLICY" == "shared" ]]; then
    echo " counterfact=shared slice across all models"
  else
    echo " counterfact=iterating_per_model stride=$N"
  fi
  echo " run=$LOCAL_RUN_DIR"
  if [[ -n "$CASE_INDEX_FILE" ]]; then
    echo " case_manifest=$CASE_INDEX_FILE"
  fi
  if $BUNDLE_GRAPHS; then
    echo " bundle_root=$BUNDLE_ROOT"
  fi
  echo "========================================"

  if $RUN_STRUCTURAL; then
    local struct_rc=0
    local posthoc_rc=0
    local bundle_rc=0
    local structural_cmd=(
      bash scripts/remote_gpu_structural.sh
      --run-root "$LOCAL_RUN_DIR"
      --n-tests "$N"
      --start-idx "$START_IDX"
      --n-prompts "$N_PROMPTS"
      --slice-policy "$SLICE_POLICY"
      --analysis-profile "$ANALYSIS_PROFILE"
      --progress-file "$LOCAL_RUN_DIR/worker_progress.txt"
      --progress-interval "$PROGRESS_INTERVAL"
      --worker-id "$WORKER_ID"
      --conda-env "$CONDA_ENV"
      --models "${MODELS[@]}"
    )
    if $COMPUTE_COV; then
      structural_cmd+=(--compute-cov)
    fi
    if [[ -n "$CASE_INDEX_FILE" ]]; then
      structural_cmd+=(--case-index-file "$CASE_INDEX_FILE")
    fi
    if [[ -n "$RAW_SPECTRAL_MAX_TOP_K" ]]; then
      structural_cmd+=(--raw-spectral-max-top-k "$RAW_SPECTRAL_MAX_TOP_K")
    fi

    printf '[local][structural] command:'
    printf ' %q' "${structural_cmd[@]}"
    echo
    "${structural_cmd[@]}" || struct_rc=$?

    local posthoc_cmd=(
      bash scripts/local_posthoc_runner.sh
      --once
      --local-run-dir "$LOCAL_RUN_DIR"
      --n-tests "$N"
      --start-idx "$START_IDX"
      --slice-policy "$SLICE_POLICY"
      --sync-interval 1
      --composite-window-sweep
      --success-file "$LOCAL_SUCCESS_FILE"
      --models "${MODELS[@]}"
    )
    printf '[local][posthoc] command:'
    printf ' %q' "${posthoc_cmd[@]}"
    echo
    "${posthoc_cmd[@]}" || posthoc_rc=$?

    if [[ "$struct_rc" -eq 0 && "$posthoc_rc" -eq 0 ]]; then
      run_bundle_graphs_if_requested || bundle_rc=$?
    fi

    if [[ "$struct_rc" -ne 0 ]]; then
      exit "$struct_rc"
    fi
    if [[ "$posthoc_rc" -ne 0 ]]; then
      exit "$posthoc_rc"
    fi
    exit "$bundle_rc"
  fi

  local rome_cmd=(
    bash scripts/remote_gpu_rome.sh
    --run-root "$LOCAL_RUN_DIR"
    --n-tests "$N"
    --start-idx "$START_IDX"
    --slice-policy "$SLICE_POLICY"
    --conda-env "$CONDA_ENV"
    --models "${MODELS[@]}"
  )
  if $COMPUTE_COV; then
    rome_cmd+=(--compute-cov)
  fi
  printf '[local][rome] command:'
  printf ' %q' "${rome_cmd[@]}"
  echo
  exec "${rome_cmd[@]}"
}

if [[ -z "$REMOTE_HOST" ]]; then
  run_local_pipeline
fi

REMOTE_DIR_ABS="$(resolve_remote_path "$REMOTE_HOST" "$REMOTE_DIR")"
REMOTE_RUN_DIR_ABS="$REMOTE_DIR_ABS/${REMOTE_REL_RUN_DIR#./}"
COV_SOURCE_DIR_ABS="$(resolve_remote_path "$COV_SOURCE_HOST" "$COV_SOURCE_DIR")"
REMOTE_CASE_INDEX_FILE=""
REMOTE_CASE_INDEX_FILE_REL=""

TOTAL_STEPS=7
if ! $RUN_STRUCTURAL; then
  TOTAL_STEPS=6
fi

echo "========================================"
echo " Latium Split Pipeline"
echo " mode=$([ "$RUN_STRUCTURAL" = true ] && printf '%s' 'structural' || printf '%s' 'rome-only')"
echo " models=${#MODELS[@]} | n=$N | start_idx=$START_IDX | compute_cov=$COMPUTE_COV"
if [[ "$SLICE_POLICY" == "shared" ]]; then
  echo " counterfact=shared slice across all models"
else
  echo " counterfact=iterating_per_model stride=$N"
fi
echo " remote=$REMOTE_HOST:$REMOTE_DIR_ABS"
echo " remote_run=$REMOTE_RUN_DIR_ABS"
echo " local_run=$LOCAL_RUN_DIR"
if [[ -n "$CASE_INDEX_FILE" ]]; then
  echo " case_manifest=$CASE_INDEX_FILE"
fi
echo " tmux_remote=$TMUX_SESSION_GPU | tmux_local=$TMUX_SESSION_LOCAL"
echo "========================================"

echo "[1/${TOTAL_STEPS}] Ensuring remote directories exist ..."
ssh_cmd "mkdir -p $(printf '%q' "$REMOTE_DIR_ABS") $(printf '%q' "$REMOTE_RUN_DIR_ABS") $(printf '%q' "$REMOTE_DIR_ABS/second_moment_stats")"

echo "[2/${TOTAL_STEPS}] Syncing repository to remote ..."
sync_git_tracked_and_unignored

if $SETUP_ENV; then
  echo "[3/${TOTAL_STEPS}] Bootstrapping remote environment ..."
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
  echo "[3/${TOTAL_STEPS}] Remote env setup skipped (--setup-env not set)"
fi

echo "[4/${TOTAL_STEPS}] Staging covariance files ..."
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

if $RUN_STRUCTURAL && [[ -n "$CASE_INDEX_FILE" ]]; then
  echo "[4.5/${TOTAL_STEPS}] Staging shared case manifest ..."
  REMOTE_CASE_INDEX_FILE="$REMOTE_RUN_DIR_ABS/manifest/$(basename "$CASE_INDEX_FILE")"
  REMOTE_CASE_INDEX_FILE_REL="$REMOTE_REL_RUN_DIR/manifest/$(basename "$CASE_INDEX_FILE")"
  ssh_cmd "mkdir -p $(printf '%q' "$REMOTE_RUN_DIR_ABS/manifest")"
  rsync -az -e "ssh ${SSH_OPTS[*]}" "$CASE_INDEX_FILE" "$REMOTE_HOST:$REMOTE_CASE_INDEX_FILE"
fi

echo "[5/${TOTAL_STEPS}] Launching remote GPU tmux session ..."
if $RUN_STRUCTURAL; then
  REMOTE_CMD=(bash scripts/remote_gpu_structural.sh --run-root "$REMOTE_REL_RUN_DIR" --n-tests "$N" --start-idx "$START_IDX" --n-prompts "$N_PROMPTS" --slice-policy "$SLICE_POLICY" --analysis-profile "$ANALYSIS_PROFILE" --progress-file "$REMOTE_REL_RUN_DIR/worker_progress.txt" --progress-interval "$PROGRESS_INTERVAL" --worker-id "$WORKER_ID" --conda-env "$CONDA_ENV" --models "${MODELS[@]}")
  if $COMPUTE_COV; then
    REMOTE_CMD=(bash scripts/remote_gpu_structural.sh --run-root "$REMOTE_REL_RUN_DIR" --n-tests "$N" --start-idx "$START_IDX" --n-prompts "$N_PROMPTS" --slice-policy "$SLICE_POLICY" --analysis-profile "$ANALYSIS_PROFILE" --progress-file "$REMOTE_REL_RUN_DIR/worker_progress.txt" --progress-interval "$PROGRESS_INTERVAL" --worker-id "$WORKER_ID" --compute-cov --conda-env "$CONDA_ENV" --models "${MODELS[@]}")
  fi
  if [[ -n "$REMOTE_CASE_INDEX_FILE_REL" ]]; then
    REMOTE_CMD+=(--case-index-file "$REMOTE_CASE_INDEX_FILE_REL")
  fi
  if [[ -n "$RAW_SPECTRAL_MAX_TOP_K" ]]; then
    REMOTE_CMD+=(--raw-spectral-max-top-k "$RAW_SPECTRAL_MAX_TOP_K")
  fi
else
  REMOTE_CMD=(bash scripts/remote_gpu_rome.sh --run-root "$REMOTE_REL_RUN_DIR" --n-tests "$N" --start-idx "$START_IDX" --slice-policy "$SLICE_POLICY" --conda-env "$CONDA_ENV" --models "${MODELS[@]}")
  if $COMPUTE_COV; then
    REMOTE_CMD+=(--compute-cov)
  fi
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

if $RUN_STRUCTURAL; then
  echo "[6/${TOTAL_STEPS}] Launching local post-hoc tmux session ..."
  LOCAL_CMD=(
    bash scripts/local_posthoc_runner.sh
    --remote-host "$REMOTE_HOST"
    --remote-run-dir "$REMOTE_RUN_DIR_ABS"
    --local-run-dir "$LOCAL_RUN_DIR"
    --n-tests "$N"
    --start-idx "$START_IDX"
    --slice-policy "$SLICE_POLICY"
    --sync-interval "$SYNC_INTERVAL"
    --composite-window-sweep
    --success-file "$LOCAL_SUCCESS_FILE"
    --models "${MODELS[@]}"
  )
  LOCAL_CMD_STR="$(quote_cmd "${LOCAL_CMD[@]}")"
  if $BUNDLE_GRAPHS; then
    BUNDLE_CMD=(bash scripts/bundle_graphs/run_all_graphs.sh --bundle-root "$BUNDLE_ROOT")
    BUNDLE_CMD_STR="$(quote_cmd "${BUNDLE_CMD[@]}")"
    LOCAL_CMD_STR="$LOCAL_CMD_STR && $BUNDLE_CMD_STR"
  fi
  tmux kill-session -t "$TMUX_SESSION_LOCAL" 2>/dev/null || true
  tmux new-session -d -s "$TMUX_SESSION_LOCAL" "cd \"$REPO_ROOT\" && $LOCAL_CMD_STR"

  echo "[7/${TOTAL_STEPS}] Sessions started"
else
  echo "[6/${TOTAL_STEPS}] Remote session started"
fi
echo
echo "Remote attach: ssh ${REMOTE_HOST} -t tmux attach -t ${TMUX_SESSION_GPU}"
if $RUN_STRUCTURAL; then
  echo "Local attach:  tmux attach -t ${TMUX_SESSION_LOCAL}"
else
  echo "Local attach:  (not used in ROME-only mode)"
fi
echo "Status:        bash pipeline.sh --remote ${REMOTE_HOST} --remote-dir ${REMOTE_DIR} --output-dir ${OUTPUT_ROOT} --n ${N} --start-idx ${START_IDX} --tmux-session-gpu ${TMUX_SESSION_GPU} --tmux-session-local ${TMUX_SESSION_LOCAL} --status"
if [[ "$SLICE_POLICY" == "shared" ]]; then
  echo "CounterFact:   all models use shared slice ${START_IDX}-$((START_IDX + N - 1))"
else
  echo "CounterFact:   first model starts at ${START_IDX}, then +${N} per model"
fi
echo "Run root:      $LOCAL_RUN_DIR"
if $RUN_STRUCTURAL; then
  echo "Success file:  $LOCAL_SUCCESS_FILE"
  if $BUNDLE_GRAPHS; then
    echo "Bundle graphs: $BUNDLE_ROOT"
  fi
else
  echo "Remote logs:   $REMOTE_RUN_DIR_ABS/logs"
  echo "Remote JSON:   $REMOTE_RUN_DIR_ABS/rome"
fi
