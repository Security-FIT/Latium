#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=""
N_TESTS=10
START_IDX=0
SLICE_POLICY="iterating_per_model"
COMPUTE_COV=false
CONDA_ENV="${CONDA_ENV:-latium}"
COV_SOURCE_NOTE="${COV_SOURCE_NOTE:-local -> ../reimagined -> kubapc fallback}"
MODELS=()

usage() {
  cat <<'EOF'
Usage: scripts/remote_gpu_rome.sh --run-root <path> [options] --models <m1 m2 ...>

Options:
  --run-root <path>     Remote run root, e.g. ./pipeline_out/n1_s0
  --n-tests <int>       Number of CounterFact cases per model (default: 10)
  --start-idx <int>     CounterFact starting index (default: 0)
  --slice-policy <mode> CounterFact assignment: iterating_per_model or shared (default: iterating_per_model)
  --compute-cov         Compute missing covariance on this runner before benchmarking
  --conda-env <name>    Conda env to activate if available (default: latium)
  --models <...>        Model config keys
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --n-tests) N_TESTS="$2"; shift 2 ;;
    --start-idx) START_IDX="$2"; shift 2 ;;
    --slice-policy) SLICE_POLICY="$2"; shift 2 ;;
    --compute-cov) COMPUTE_COV=true; shift ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --models)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        MODELS+=("$1")
        shift
      done
      ;;
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

if [[ -z "$RUN_ROOT" ]]; then
  echo "ERROR: --run-root is required" >&2
  exit 1
fi

if [[ ${#MODELS[@]} -eq 0 ]]; then
  echo "ERROR: at least one model is required" >&2
  exit 1
fi

if [[ "$SLICE_POLICY" != "iterating_per_model" && "$SLICE_POLICY" != "shared" ]]; then
  echo "ERROR: unsupported --slice-policy: $SLICE_POLICY" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null || true
fi

mkdir -p "$RUN_ROOT/rome" "$RUN_ROOT/logs" "$RUN_ROOT/archive" second_moment_stats
rm -f "$RUN_ROOT/REMOTE_GPU_DONE" "$RUN_ROOT/REMOTE_GPU_FAILED"

GPU_NAME="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on this runner.")
print(torch.cuda.get_device_name(0))
PY
)"

MODELS_CSV="$(IFS=,; printf '%s' "${MODELS[*]}")"
STARTED_AT="$(date --iso-8601=seconds)"

write_status_json() {
  local phase="$1"
  local failed="$2"
  local finished_at="$3"
  STATUS_PHASE="$phase" \
  STATUS_FAILED="$failed" \
  STATUS_FINISHED_AT="$finished_at" \
  RUN_ROOT="$RUN_ROOT" \
  MODELS_CSV="$MODELS_CSV" \
  GPU_NAME="$GPU_NAME" \
  STARTED_AT="$STARTED_AT" \
  N_TESTS="$N_TESTS" \
  START_IDX="$START_IDX" \
  SLICE_POLICY="$SLICE_POLICY" \
  COMPUTE_COV="$COMPUTE_COV" \
  COV_SOURCE_NOTE="$COV_SOURCE_NOTE" \
  python - <<'PY'
import json
import os
from pathlib import Path

models = [item for item in os.environ.get("MODELS_CSV", "").split(",") if item]
n_tests = int(os.environ.get("N_TESTS", "0") or 0)
start_idx = int(os.environ.get("START_IDX", "0") or 0)
slice_policy = os.environ.get("SLICE_POLICY", "")
model_assignments = []
for index, model in enumerate(models):
    model_start = start_idx if slice_policy == "shared" else start_idx + index * n_tests
    model_assignments.append(
        {
            "model": model,
            "start_idx": model_start,
            "end_idx": model_start + n_tests - 1,
        }
    )

payload = {
    "hostname": os.uname().nodename,
    "gpu_name": os.environ.get("GPU_NAME", ""),
    "mode": "rome_only",
    "models": models,
    "n_tests": n_tests,
    "start_idx": start_idx,
    "slice_policy": slice_policy,
    "model_assignments": model_assignments,
    "compute_cov": os.environ.get("COMPUTE_COV", "").lower() == "true",
    "cov_source": os.environ.get("COV_SOURCE_NOTE", ""),
    "started_at": os.environ.get("STARTED_AT", ""),
    "finished_at": os.environ.get("STATUS_FINISHED_AT", ""),
    "phase": os.environ.get("STATUS_PHASE", ""),
    "failed": os.environ.get("STATUS_FAILED", "").lower() == "true",
}
Path(os.environ["RUN_ROOT"], "remote_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

expected_cov_basename() {
  MODEL_KEY="$1" python - <<'PY'
import os
from src.model_config import second_moment_basename

print(second_moment_basename(os.environ["MODEL_KEY"]))
PY
}

model_start_idx() {
  local model_index="$1"
  if [[ "$SLICE_POLICY" == "shared" ]]; then
    echo "$START_IDX"
  else
    echo $((START_IDX + model_index * N_TESTS))
  fi
}

write_status_json "running" "false" ""

overall_failed=0

for model_index in "${!MODELS[@]}"; do
  model="${MODELS[$model_index]}"
  model_start="$(model_start_idx "$model_index")"
  model_end=$((model_start + N_TESTS - 1))
  echo "[remote-gpu][rome] ===== $model | cases ${model_start}-${model_end} ====="
  log_file="$RUN_ROOT/logs/${model}.log"
  : > "$log_file"

  cov_basename="$(expected_cov_basename "$model")"
  cov_path="$REPO_ROOT/second_moment_stats/$cov_basename"
  if [[ ! -f "$cov_path" ]]; then
    if $COMPUTE_COV; then
      echo "[remote-gpu][rome] missing covariance for $model -> computing" | tee -a "$log_file"
      if ! PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m src.cli command=second-moment model="$model" model.device=cuda +model.cuda_mode=strict 2>&1 | tee -a "$log_file"; then
        echo "[remote-gpu][rome] covariance computation failed for $model" | tee -a "$log_file"
        overall_failed=1
        continue
      fi
    else
      echo "[remote-gpu][rome] missing covariance for $model and --compute-cov is disabled" | tee -a "$log_file"
      overall_failed=1
      continue
    fi
  fi

  cmd=(
    python rome_benchmark.py
    --models "$model"
    --n-tests "$N_TESTS"
    --start-idx "$model_start"
    --output-dir "$RUN_ROOT/rome"
  )

  printf '[remote-gpu][rome] command:' | tee -a "$log_file"
  printf ' %q' "${cmd[@]}" | tee -a "$log_file"
  echo | tee -a "$log_file"

  if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${cmd[@]}" 2>&1 | tee -a "$log_file"; then
    echo "[remote-gpu][rome] $model benchmark complete" | tee -a "$log_file"
  else
    echo "[remote-gpu][rome] $model benchmark failed" | tee -a "$log_file"
    overall_failed=1
  fi
done

FINISHED_AT="$(date --iso-8601=seconds)"
touch "$RUN_ROOT/REMOTE_GPU_DONE"
if [[ "$overall_failed" -ne 0 ]]; then
  touch "$RUN_ROOT/REMOTE_GPU_FAILED"
  write_status_json "done" "true" "$FINISHED_AT"
  exit 1
fi

write_status_json "done" "false" "$FINISHED_AT"
exit 0
