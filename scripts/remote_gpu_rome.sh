#!/usr/bin/env bash
# :copyright: 2025 Jakub Res
# :license: MIT
# :author: Matej Olexa <olexa.matej@gmail.com>
# :author: Jakub Res <iresj@fit.vut.cz>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_GPU_CONFIG="$REPO_ROOT/src/config/remote_gpu_rome/default.yaml"

remote_default() {
  python - "$REMOTE_GPU_CONFIG" "$1" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
key = sys.argv[2]
for line in path.read_text(encoding="utf-8").splitlines():
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or ":" not in stripped:
        continue
    name, value = stripped.split(":", 1)
    if name.strip() == key:
        print(value.strip().strip('"').strip("'"))
        raise SystemExit(0)
raise SystemExit(f"Missing {key} in {path}")
PY
}

RUN_ROOT=""
N_TESTS="$(remote_default n_tests)"
START_IDX="$(remote_default start_idx)"
SLICE_POLICY="$(remote_default slice_policy)"
COMPUTE_COV="$(remote_default compute_cov)"
CONDA_ENV="$(remote_default conda_env)"
COV_SOURCE_NOTE="$(remote_default cov_source_note)"
MODELS=()

usage() {
  cat <<'EOF'
Usage: scripts/remote_gpu_rome.sh --run-root <path> [options] --models <m1 m2 ...>

Options:
  --run-root <path>     Remote run root, e.g. ./pipeline_out/n1_s0
  --n-tests <int>       Number of CounterFact cases per model
  --start-idx <int>     CounterFact starting index
  --slice-policy <mode> CounterFact assignment: iterating_per_model or shared
  --compute-cov         Compute missing covariance on this runner before benchmarking
  --conda-env <name>    Conda env to activate if available
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

cd "$REPO_ROOT"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null || true
fi

mkdir -p "$RUN_ROOT/rome" "$RUN_ROOT/logs" "$RUN_ROOT/archive" data/second_moment_stats
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
  python - \
    "$RUN_ROOT" \
    "$MODELS_CSV" \
    "$GPU_NAME" \
    "$STARTED_AT" \
    "$N_TESTS" \
    "$START_IDX" \
    "$SLICE_POLICY" \
    "$COMPUTE_COV" \
    "$COV_SOURCE_NOTE" \
    "$phase" \
    "$failed" \
    "$finished_at" <<'PY'
import json
import socket
import sys
from pathlib import Path

(
    run_root,
    models_csv,
    gpu_name,
    started_at,
    n_tests_raw,
    start_idx_raw,
    slice_policy,
    compute_cov_raw,
    cov_source_note,
    phase,
    failed_raw,
    finished_at,
) = sys.argv[1:]
models = [item for item in models_csv.split(",") if item]
n_tests = int(n_tests_raw)
start_idx = int(start_idx_raw)
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
    "hostname": socket.gethostname(),
    "gpu_name": gpu_name,
    "mode": "rome_only",
    "models": models,
    "n_tests": n_tests,
    "start_idx": start_idx,
    "slice_policy": slice_policy,
    "model_assignments": model_assignments,
    "compute_cov": compute_cov_raw.lower() == "true",
    "cov_source": cov_source_note,
    "started_at": started_at,
    "finished_at": finished_at,
    "phase": phase,
    "failed": failed_raw.lower() == "true",
}
Path(run_root, "remote_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

expected_cov_basename() {
  python - "$1" <<'PY'
import sys
from src.common.model_config import second_moment_basename

print(second_moment_basename(sys.argv[1]))
PY
}

expected_cov_path() {
  python - "$1" <<'PY'
import sys
from pathlib import Path

from src.common.model_config import load_model_config, second_moment_basename

model_key = sys.argv[1]
cfg = load_model_config(model_key)
explicit = str(getattr(cfg, "second_moment_path", "") or "").strip()
if explicit:
    path = Path(explicit)
else:
    path = Path(str(getattr(cfg, "second_moment_dir"))) / second_moment_basename(model_key)
print(path)
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

  cov_path="$(expected_cov_path "$model")"
  if [[ "$cov_path" != /* ]]; then
    cov_path="$REPO_ROOT/$cov_path"
  fi
  if [[ ! -f "$cov_path" ]]; then
    if $COMPUTE_COV; then
      echo "[remote-gpu][rome] missing covariance for $model -> computing" | tee -a "$log_file"
      if ! PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m src command=second-moment model="$model" model.device=cuda ++model.cuda_mode=strict 2>&1 | tee -a "$log_file"; then
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
    "rome_benchmark.models=[$model]"
    "rome_benchmark.n_tests=$N_TESTS"
    "rome_benchmark.start_idx=$model_start"
    "rome_benchmark.output_dir=$RUN_ROOT/rome"
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
