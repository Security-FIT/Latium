#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT=""
N_TESTS=50
START_IDX=30
N_PROMPTS=50
COMPUTE_COV=false
ANALYSIS_PROFILE="paper"
SLICE_POLICY="iterating_per_model"
CONDA_ENV="${CONDA_ENV:-latium}"
COV_SOURCE_NOTE="${COV_SOURCE_NOTE:-local -> ../reimagined -> kubapc fallback}"
CASE_INDEX_FILE=""
RAW_SPECTRAL_MAX_TOP_K=""
PROGRESS_FILE=""
PROGRESS_INTERVAL=10
WORKER_ID=""
MODELS=()

usage() {
  cat <<'EOF'
Usage: scripts/remote_gpu_structural.sh --run-root <path> [options] --models <m1 m2 ...>

Options:
  --run-root <path>     Remote run root, e.g. ./pipeline_out/n50_s30
  --n-tests <int>       Number of CounterFact cases per model (default: 50)
  --start-idx <int>     CounterFact starting index (default: 30)
  --n-prompts <int>     Number of prompts per benchmark evaluation (default: 50)
  --case-index-file <path>
                        Explicit CounterFact manifest shared across models
  --compute-cov         Compute missing covariance on this runner before benchmarking
  --slice-policy <mode> CounterFact assignment: iterating_per_model or shared (default: iterating_per_model)
  --analysis-profile <name>
                        Structural benchmark payload: raw, paper, or full (default: paper)
  --posthoc-only        Alias for --analysis-profile paper
  --detection-only      Alias for --analysis-profile paper
  --raw-spectral-max-top-k <int>
                        Optional stored raw spectral cap for post-hoc sweeps
  --progress-file <path>
                        Worker progress text file updated by structural_benchmark
  --progress-interval <int>
                        Progress update cadence in completed cases (default: 10)
  --worker-id <name>    Human-readable worker label for progress and status
  --conda-env <name>    Conda env to activate if available (default: latium)
  --models <...>        Model config keys
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --n-tests) N_TESTS="$2"; shift 2 ;;
    --start-idx) START_IDX="$2"; shift 2 ;;
    --n-prompts) N_PROMPTS="$2"; shift 2 ;;
    --case-index-file) CASE_INDEX_FILE="$2"; shift 2 ;;
    --compute-cov) COMPUTE_COV=true; shift ;;
    --slice-policy) SLICE_POLICY="$2"; shift 2 ;;
    --analysis-profile) ANALYSIS_PROFILE="$2"; shift 2 ;;
    --posthoc-only|--detection-only) ANALYSIS_PROFILE="paper"; shift ;;
    --raw-spectral-max-top-k) RAW_SPECTRAL_MAX_TOP_K="$2"; shift 2 ;;
    --progress-file) PROGRESS_FILE="$2"; shift 2 ;;
    --progress-interval) PROGRESS_INTERVAL="$2"; shift 2 ;;
    --worker-id) WORKER_ID="$2"; shift 2 ;;
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

mkdir -p "$RUN_ROOT/structural" "$RUN_ROOT/logs" "$RUN_ROOT/archive" second_moment_stats
rm -f "$RUN_ROOT/REMOTE_GPU_DONE" "$RUN_ROOT/REMOTE_GPU_FAILED"

GPU_NAME="$(python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available on this runner.")
print(torch.cuda.get_device_name(0))
PY
)"

echo "[remote-gpu] run_root=$RUN_ROOT"
echo "[remote-gpu] gpu=$GPU_NAME"
echo "[remote-gpu] slice_policy=$SLICE_POLICY"
if [[ "$GPU_NAME" != *H100* && "$GPU_NAME" != *H200* ]]; then
  echo "[remote-gpu] WARNING: GPU is not H100/H200" >&2
fi

MODELS_CSV="$(IFS=,; printf '%s' "${MODELS[*]}")"
STARTED_AT="$(date --iso-8601=seconds)"

model_start_idx() {
  local model_index="$1"
  if [[ "$SLICE_POLICY" == "shared" ]]; then
    echo "$START_IDX"
    return 0
  fi
  echo $((START_IDX + (model_index * N_TESTS)))
}

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
  CASE_INDEX_FILE="$CASE_INDEX_FILE" \
  WORKER_ID="$WORKER_ID" \
  python - <<'PY'
import json
import os
from pathlib import Path

models = [item for item in os.environ.get("MODELS_CSV", "").split(",") if item]
base_start = int(os.environ.get("START_IDX", "0") or 0)
n_tests = int(os.environ.get("N_TESTS", "0") or 0)
slice_policy = os.environ.get("SLICE_POLICY", "iterating_per_model")


def assignment_start(model_index: int) -> int:
    if slice_policy == "shared":
        return base_start
    return base_start + model_index * n_tests


model_assignments = [
    {
        "model": model,
        "start_idx": assignment_start(idx),
        "end_idx": assignment_start(idx) + max(0, n_tests - 1),
    }
    for idx, model in enumerate(models)
]
payload = {
    "hostname": os.uname().nodename,
    "gpu_name": os.environ.get("GPU_NAME", ""),
    "worker_id": os.environ.get("WORKER_ID", ""),
    "models": models,
    "n_tests": n_tests,
    "start_idx": base_start,
    "slice_policy": slice_policy,
    "model_assignments": model_assignments,
    "compute_cov": os.environ.get("COMPUTE_COV", "").lower() == "true",
    "cov_source": os.environ.get("COV_SOURCE_NOTE", ""),
    "case_index_file": os.environ.get("CASE_INDEX_FILE", ""),
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

write_failure_json() {
  local model_key="$1"
  local start_idx_used="$2"
  local error_message="$3"
  local out_path="$RUN_ROOT/structural/rome_structural_${model_key}_runner_failure_$(date +%Y-%m-%d_%H-%M-%S).json"
  MODEL_KEY="$model_key" \
  START_IDX_USED="$start_idx_used" \
  ERROR_MESSAGE="$error_message" \
  OUT_PATH="$out_path" \
  ANALYSIS_PROFILE="$ANALYSIS_PROFILE" \
  N_TESTS="$N_TESTS" \
  python - <<'PY'
import json
import os
from datetime import datetime
from pathlib import Path

payload = {
    "metadata": {
        "model": os.environ["MODEL_KEY"],
        "timestamp": datetime.now().isoformat(),
        "analysis_profile": os.environ.get("ANALYSIS_PROFILE", "paper"),
        "start_idx_used": int(os.environ.get("START_IDX_USED", "0") or 0),
        "end_idx_used": int(os.environ.get("START_IDX_USED", "0") or 0) + max(0, int(os.environ.get("N_TESTS", "0") or 0) - 1),
        "runner_generated_failure": True,
    },
    "error": os.environ["ERROR_MESSAGE"],
    "tests": [],
    "summary": {
        "total": int(os.environ.get("N_TESTS", "0") or 0),
        "successful": 0,
        "skipped": int(os.environ.get("N_TESTS", "0") or 0),
        "cases_total": int(os.environ.get("N_TESTS", "0") or 0),
        "cases_completed": 0,
        "cases_error": int(os.environ.get("N_TESTS", "0") or 0),
        "rome_success_count": 0,
        "rome_success_rate": 0.0,
        "detector_eligible_count": 0,
        "mean_efficacy_score": 0.0,
        "mean_efficacy_magnitude": 0.0,
        "mean_paraphrase_score": 0.0,
        "mean_neighborhood_score": 0.0,
        "mean_overall_score": 0.0,
    },
}
Path(os.environ["OUT_PATH"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(os.environ["OUT_PATH"])
PY
}

write_status_json "running" "false" ""

overall_failed=0
completed_models=()
failed_models=()

write_progress_state() {
  local status="$1"
  local model="$2"
  local current_progress="$3"
  local log_file="$4"
  local completed_csv="$5"
  local remaining_csv="$6"
  local failed_csv="$7"

  if [[ -z "$PROGRESS_FILE" ]]; then
    return 0
  fi

  PROGRESS_FILE="$PROGRESS_FILE" \
  WORKER_ID="$WORKER_ID" \
  MODEL_KEY="$model" \
  STATUS_VALUE="$status" \
  CURRENT_PROGRESS="$current_progress" \
  N_TESTS="$N_TESTS" \
  PROGRESS_INTERVAL="$PROGRESS_INTERVAL" \
  RUN_ROOT="$RUN_ROOT" \
  LOG_FILE="$log_file" \
  COMPLETED_MODELS="$completed_csv" \
  REMAINING_MODELS="$remaining_csv" \
  FAILED_MODELS="$failed_csv" \
  python - <<'PY'
from pathlib import Path
import os
from src.worker_progress import effective_progress_interval, write_worker_progress

write_worker_progress(
    Path(os.environ["PROGRESS_FILE"]),
    {
        "worker_id": os.environ.get("WORKER_ID", ""),
        "status": os.environ.get("STATUS_VALUE", ""),
        "current_model": os.environ.get("MODEL_KEY", ""),
        "current_model_progress": os.environ.get("CURRENT_PROGRESS", ""),
        "progress_interval": effective_progress_interval(
            int(os.environ.get("N_TESTS", "0") or 0),
            int(os.environ.get("PROGRESS_INTERVAL", "10") or 10),
        ),
        "completed_models": os.environ.get("COMPLETED_MODELS", ""),
        "remaining_models": os.environ.get("REMAINING_MODELS", ""),
        "failed_models": os.environ.get("FAILED_MODELS", ""),
        "current_run_root": os.environ.get("RUN_ROOT", ""),
        "current_log": os.environ.get("LOG_FILE", ""),
    },
    preserve_existing=True,
)
PY
}

for model_index in "${!MODELS[@]}"; do
  model="${MODELS[$model_index]}"
  model_start="$(model_start_idx "$model_index")"
  model_end=$((model_start + N_TESTS - 1))
  echo "[remote-gpu] ===== $model | cases ${model_start}-${model_end} ====="
  log_file="$RUN_ROOT/logs/${model}.log"
  : > "$log_file"
  remaining_models=("${MODELS[@]:$((model_index + 1))}")
  completed_csv="$(IFS=,; printf '%s' "${completed_models[*]}")"
  remaining_csv="$(IFS=,; printf '%s' "${remaining_models[*]}")"
  failed_csv="$(IFS=,; printf '%s' "${failed_models[*]}")"

  if [[ -n "$PROGRESS_FILE" ]]; then
    write_progress_state "running" "$model" "0/$N_TESTS" "$log_file" "$completed_csv" "$remaining_csv" "$failed_csv"
  fi

  cov_basename="$(expected_cov_basename "$model")"
  cov_path="$REPO_ROOT/second_moment_stats/$cov_basename"
  if [[ ! -f "$cov_path" ]]; then
    if $COMPUTE_COV; then
      echo "[remote-gpu] missing covariance for $model -> computing" | tee -a "$log_file"
      if ! PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m src.cli command=second-moment model="$model" model.device=cuda ++model.cuda_mode=strict 2>&1 | tee -a "$log_file"; then
        echo "[remote-gpu] covariance computation failed for $model" | tee -a "$log_file"
        write_failure_json "$model" "$model_start" "covariance computation failed"
        failed_models+=("$model")
        failed_csv="$(IFS=,; printf '%s' "${failed_models[*]}")"
        write_progress_state "model_failed" "$model" "$N_TESTS/$N_TESTS" "$log_file" "$completed_csv" "$remaining_csv" "$failed_csv"
        overall_failed=1
        continue
      fi
    else
      echo "[remote-gpu] missing covariance for $model and --compute-cov is disabled" | tee -a "$log_file"
      write_failure_json "$model" "$model_start" "missing staged covariance"
      failed_models+=("$model")
      failed_csv="$(IFS=,; printf '%s' "${failed_models[*]}")"
      write_progress_state "model_failed" "$model" "$N_TESTS/$N_TESTS" "$log_file" "$completed_csv" "$remaining_csv" "$failed_csv"
      overall_failed=1
      continue
    fi
  fi

  if [[ ! -f "$cov_path" ]]; then
    echo "[remote-gpu] expected covariance still missing for $model: $cov_path" | tee -a "$log_file"
    write_failure_json "$model" "$model_start" "expected covariance still missing after preflight"
    failed_models+=("$model")
    failed_csv="$(IFS=,; printf '%s' "${failed_models[*]}")"
    write_progress_state "model_failed" "$model" "$N_TESTS/$N_TESTS" "$log_file" "$completed_csv" "$remaining_csv" "$failed_csv"
    overall_failed=1
    continue
  fi

  cmd=(
    python structural_benchmark.py
    --models "$model"
    --n-tests "$N_TESTS"
    --start-idx "$model_start"
    --n-prompts "$N_PROMPTS"
    --analysis-profile "$ANALYSIS_PROFILE"
    --output-dir "$RUN_ROOT/structural"
    --fail-on-missing-second-moment
  )
  if [[ -n "$CASE_INDEX_FILE" ]]; then
    cmd+=(--case-index-file "$CASE_INDEX_FILE")
  fi
  if [[ -n "$RAW_SPECTRAL_MAX_TOP_K" ]]; then
    cmd+=(--raw-spectral-max-top-k "$RAW_SPECTRAL_MAX_TOP_K")
  fi
  if [[ -n "$PROGRESS_FILE" ]]; then
    cmd+=(--progress-file "$PROGRESS_FILE" --progress-interval "$PROGRESS_INTERVAL")
  fi
  if [[ -n "$WORKER_ID" ]]; then
    cmd+=(--worker-id "$WORKER_ID")
  fi

  printf '[remote-gpu] command:' | tee -a "$log_file"
  printf ' %q' "${cmd[@]}" | tee -a "$log_file"
  echo | tee -a "$log_file"

  if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${cmd[@]}" 2>&1 | tee -a "$log_file"; then
    echo "[remote-gpu] $model benchmark complete" | tee -a "$log_file"
    completed_models+=("$model")
    completed_csv="$(IFS=,; printf '%s' "${completed_models[*]}")"
    failed_csv="$(IFS=,; printf '%s' "${failed_models[*]}")"
    write_progress_state "model_complete" "$model" "$N_TESTS/$N_TESTS" "$log_file" "$completed_csv" "$remaining_csv" "$failed_csv"
  else
    echo "[remote-gpu] $model benchmark failed" | tee -a "$log_file"
    write_failure_json "$model" "$model_start" "structural benchmark command failed"
    failed_models+=("$model")
    failed_csv="$(IFS=,; printf '%s' "${failed_models[*]}")"
    write_progress_state "model_failed" "$model" "$N_TESTS/$N_TESTS" "$log_file" "$completed_csv" "$remaining_csv" "$failed_csv"
    overall_failed=1
  fi
done

FINISHED_AT="$(date --iso-8601=seconds)"
touch "$RUN_ROOT/REMOTE_GPU_DONE"
final_completed_csv="$(IFS=,; printf '%s' "${completed_models[*]}")"
final_failed_csv="$(IFS=,; printf '%s' "${failed_models[*]}")"
if [[ "$overall_failed" -ne 0 ]]; then
  touch "$RUN_ROOT/REMOTE_GPU_FAILED"
  write_progress_state "done_with_failures" "" "" "" "$final_completed_csv" "" "$final_failed_csv"
  write_status_json "done" "true" "$FINISHED_AT"
  exit 1
fi

write_progress_state "done" "" "" "" "$final_completed_csv" "" "$final_failed_csv"
write_status_json "done" "false" "$FINISHED_AT"
exit 0
