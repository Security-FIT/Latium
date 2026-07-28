#!/usr/bin/env bash
set -Eeuo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage:
  jobs/rome_math_n50_cluster.sh [--dry-run] trace MODEL
  jobs/rome_math_n50_cluster.sh [--dry-run] covariance MODEL LAYER
  jobs/rome_math_n50_cluster.sh [--dry-run] smoke-a
  jobs/rome_math_n50_cluster.sh [--dry-run] smoke-b
  jobs/rome_math_n50_cluster.sh [--dry-run] n50 MODEL|all

Environment:
  LATIUM_MAIN_ROOT       Clean, current main checkout (trace/covariance stages)
  LATIUM_DETECTOR_ROOT   detector-simplification checkout (defaults to this repo)
  LATIUM_EXECUTION_BACKEND  pbs (default) or direct (dedicated GPU host only)
  LATIUM_ENV             Python virtual environment for direct execution
  ROME_MATH_N50_STATE    Append-only PBS job ledger

Run this driver inside the cluster tmux frontend session. Every computational
stage uses PBS by default. The direct backend is allowed only from an existing
tmux session on a dedicated compute host.
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DETECTOR_ROOT="${LATIUM_DETECTOR_ROOT:-$ROOT}"
MAIN_ROOT="${LATIUM_MAIN_ROOT:-}"
STATE="${ROME_MATH_N50_STATE:-$DETECTOR_ROOT/analysis_out/rome-math-n50-orchestration/job-ids.tsv}"
BACKEND="${LATIUM_EXECUTION_BACKEND:-pbs}"
DRY_RUN=0

MODELS=(
  gpt2-xl
  gpt-j-6b
  mistral-7b-v0.1
  mistral-7b-v0.3
  deepseek-7b-base
  falcon-7b
  opt-6.7b
  llama2-7b
  granite4-micro
)
SMOKE_MODELS=(gpt2-xl mistral-7b-v0.1)

declare -A RUN_IDS=(
  [gpt2-xl]=rome-math-n50-gpt2-xl-v1
  [gpt-j-6b]=rome-math-n50-gpt-j-6b-v1
  [mistral-7b-v0.1]=rome-math-n50-mistral-7b-v0.1-v1
  [mistral-7b-v0.3]=rome-math-n50-mistral-7b-v0.3-v1
  [deepseek-7b-base]=rome-math-n50-deepseek-7b-base-v1
  [falcon-7b]=rome-math-n50-falcon-7b-v1
  [opt-6.7b]=rome-math-n50-opt-6.7b-v1
  [llama2-7b]=rome-math-n50-llama2-7b-v1
  [granite4-micro]=rome-math-n50-granite4-micro-v1
)

contains_model() {
  local requested="$1"
  local model
  for model in "${MODELS[@]}"; do
    [[ "$model" == "$requested" ]] && return 0
  done
  return 1
}

short_name() {
  printf '%s' "$1" | tr -cd '[:alnum:]-' | cut -c1-13
}

record_job() {
  local stage="$1"
  local model="$2"
  local run_id="$3"
  local output="$4"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '%s\n' "$output"
    return 0
  fi
  local job_id
  job_id="$(printf '%s\n' "$output" | awk 'NF {print $1}' | tail -n1)"
  [[ -n "$job_id" ]] || die "qsub returned no job ID for $stage/$model"
  mkdir -p "$(dirname "$STATE")"
  if [[ ! -f "$STATE" ]]; then
    printf 'submitted_at\tstage\tmodel\trun_id\tpbs_job_id\n' > "$STATE"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$stage" "$model" "$run_id" "$job_id" >> "$STATE"
  printf '%s\n' "$output"
}

record_direct() {
  local stage="$1"
  local model="$2"
  local run_id="$3"
  [[ "$DRY_RUN" == 0 ]] || return 0
  mkdir -p "$(dirname "$STATE")"
  if [[ ! -f "$STATE" ]]; then
    printf 'submitted_at\tstage\tmodel\trun_id\tpbs_job_id\n' > "$STATE"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$stage" "$model" "$run_id" "direct:$(hostname -f):complete" >> "$STATE"
}

submit() {
  local root="$1"
  shift
  local args=("$root/jobs/submit.sh")
  local inserted=0
  local arg
  for arg in "$@"; do
    if [[ "$DRY_RUN" == 1 && "$arg" == "--" && "$inserted" == 0 ]]; then
      args+=(--dry-run)
      inserted=1
    fi
    args+=("$arg")
  done
  if [[ "$DRY_RUN" == 1 && "$inserted" == 0 ]]; then
    args+=(--dry-run)
  fi
  "${args[@]}"
}

run_direct() {
  local root="$1"
  shift
  [[ -n "${TMUX:-}" ]] || die "direct backend requires an existing tmux session"
  local python_bin
  if [[ -n "${LATIUM_ENV:-}" ]]; then
    python_bin="$LATIUM_ENV/bin/python"
  elif [[ -x "$DETECTOR_ROOT/.venv/bin/python" ]]; then
    python_bin="$DETECTOR_ROOT/.venv/bin/python"
  else
    die "direct backend requires LATIUM_ENV or detector .venv"
  fi
  [[ -x "$python_bin" ]] || die "Python environment is unavailable: $python_bin"
  local command=("$python_bin" -m src "$@")
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'direct command:'
    printf ' %q' "${command[@]}"
    printf '\n'
    return 0
  fi
  (
    cd "$root"
    "${command[@]}"
  )
}

require_dependency_ready() {
  local model="$1"
  python3 - "$DETECTOR_ROOT" "$model" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

root = Path(sys.argv[1])
model = sys.argv[2]
manifest = OmegaConf.load(root / "manifests/rome_math_ablation_n50_cluster.yaml")
records = {str(record.model): record for record in manifest.models}
record = records[model]
if str(record.dependency_decision) not in {"reuse_exact", "trace_and_recompute"}:
    raise SystemExit(f"dependency is not audited for {model}")
if str(record.covariance.status) != "verified":
    raise SystemExit(f"covariance is not verified for {model}")
path = root / str(record.covariance.path).removeprefix("./")
if not path.is_file() or path.stat().st_size <= 0:
    raise SystemExit(f"verified covariance is absent: {path}")
PY
}

validate_trace() {
  local model="$1"
  local layer="$2"
  local trace_root="$MAIN_ROOT/analysis_out/rome-math-n50-dependencies/$model/causal-trace"
  python3 - "$trace_root" "$layer" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
summaries = sorted(root.glob("*/summary.json"), key=lambda path: path.stat().st_mtime_ns)
if not summaries:
    raise SystemExit(f"no causal-trace summary under {root}")
path = summaries[-1]
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("confirmation_passed") is not True:
    raise SystemExit(f"causal trace did not pass confirmation: {path}")
if not isinstance(data.get("selected_trace_center"), int):
    raise SystemExit(f"selected_trace_center is not an integer: {path}")
if int(data["selected_trace_center"]) != expected:
    raise SystemExit(
        f"confirmed layer {data['selected_trace_center']} does not equal requested {expected}"
    )
if int(data.get("num_valid_facts", 0)) <= 0:
    raise SystemExit(f"causal trace has no valid facts: {path}")
plot = Path(str(data.get("plot", "")))
if not plot.is_file() or plot.stat().st_size <= 0:
    raise SystemExit(f"causal trace plot is missing or empty: {plot}")
print(path)
PY
}

submit_trace() {
  local model="$1"
  [[ -n "$MAIN_ROOT" && -d "$MAIN_ROOT/.git" ]] || die "LATIUM_MAIN_ROOT must name the main checkout"
  contains_model "$model" || die "unsupported model: $model"
  if [[ "$BACKEND" == direct ]]; then
    run_direct "$MAIN_ROOT" causal-trace \
      "model=$model" \
      "command.causal_trace.output_dir=analysis_out/rome-math-n50-dependencies/$model/causal-trace"
    record_direct trace "$model" dependency
  else
    local output
    output="$(submit "$MAIN_ROOT" causal-trace \
      --name "ct-$(short_name "$model")" \
      --mem 96gb --gpu-mem 40gb --scratch 100gb --walltime 48:00:00 -- \
      "model=$model" \
      "command.causal_trace.output_dir=analysis_out/rome-math-n50-dependencies/$model/causal-trace")"
    record_job trace "$model" dependency "$output"
  fi
}

submit_covariance() {
  local model="$1"
  local layer="$2"
  [[ -n "$MAIN_ROOT" && -d "$MAIN_ROOT/.git" ]] || die "LATIUM_MAIN_ROOT must name the main checkout"
  contains_model "$model" || die "unsupported model: $model"
  [[ "$layer" =~ ^[0-9]+$ ]] || die "layer must be a non-negative integer"
  [[ "$DRY_RUN" == 1 ]] || validate_trace "$model" "$layer"
  if [[ "$BACKEND" == direct ]]; then
    run_direct "$MAIN_ROOT" second-moment \
      "model=$model" \
      "model.layer=$layer" \
      model.second_moment_path=null \
      model.second_moment_target_samples=100000
    record_direct covariance "$model" "layer-$layer"
  else
    local output
    output="$(submit "$MAIN_ROOT" second-moment \
      --name "sm-$(short_name "$model")-l$layer" \
      --mem 96gb --gpu-mem 40gb --scratch 100gb --walltime 48:00:00 -- \
      "model=$model" \
      "model.layer=$layer" \
      model.second_moment_path=null \
      model.second_moment_target_samples=100000)"
    record_job covariance "$model" "layer-$layer" "$output"
  fi
}

submit_detector() {
  local model="$1"
  local run_id="$2"
  local count="$3"
  local stage="$4"
  require_dependency_ready "$model"
  local detector_args=(
    structural run
    "structural.run.models=[$model]"
    'structural.run.edit_methods=[rome]'
    "structural.run.n_tests=$count"
    structural.run.start_idx=0
    structural.run.output_dir=analysis_out
    "structural.run.run_id=$run_id"
    structural.run.fail_on_missing_second_moment=true
    structural.capture.profile=rome-math-ablation
    structural.analysis.preset=none
    structural.render.enabled=false
  )
  if [[ "$BACKEND" == direct ]]; then
    run_direct "$DETECTOR_ROOT" "${detector_args[@]}"
    record_direct "$stage" "$model" "$run_id"
  else
    local output
    output="$(submit "$DETECTOR_ROOT" custom \
      --name "$(short_name "$stage-$model")" \
      --mem 96gb --gpu-mem 40gb --scratch 150gb --walltime 48:00:00 -- \
      "${detector_args[@]}")"
    record_job "$stage" "$model" "$run_id" "$output"
  fi
}

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
[[ "$BACKEND" == pbs || "$BACKEND" == direct ]] || die "LATIUM_EXECUTION_BACKEND must be pbs or direct"
[[ $# -ge 1 ]] || { usage; exit 2; }
ACTION="$1"
shift

case "$ACTION" in
  trace)
    [[ $# -eq 1 ]] || die "trace requires MODEL"
    submit_trace "$1"
    ;;
  covariance)
    [[ $# -eq 2 ]] || die "covariance requires MODEL LAYER"
    submit_covariance "$1" "$2"
    ;;
  smoke-a|smoke-b)
    [[ $# -eq 0 ]] || die "$ACTION takes no model argument"
    repeat="${ACTION#smoke-}"
    for model in "${SMOKE_MODELS[@]}"; do
      submit_detector "$model" "rome-math-n50-smoke-$repeat-$model-v1" 2 "$ACTION"
    done
    ;;
  n50)
    [[ $# -eq 1 ]] || die "n50 requires MODEL or all"
    if [[ "$1" == all ]]; then
      for model in "${MODELS[@]}"; do
        submit_detector "$model" "${RUN_IDS[$model]}" 50 n50
      done
    else
      contains_model "$1" || die "unsupported model: $1"
      submit_detector "$1" "${RUN_IDS[$1]}" 50 n50
    fi
    ;;
  -h|--help)
    usage
    ;;
  *)
    die "unknown action: $ACTION"
    ;;
esac
