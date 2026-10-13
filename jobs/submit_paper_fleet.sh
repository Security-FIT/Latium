#!/usr/bin/env bash
set -Eeuo pipefail

# Submit one independent PBS worker per model. Workers share only the
# repository/output directory; each model writes its own state and logs.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${LATIUM_PAPER_FLEET_ROOT:-$ROOT/analysis_out/paper-fleet/$(date -u +%Y%m%dT%H%M%SZ)}"
N_TESTS="${LATIUM_N_TESTS:-50}"
COVARIANCE_SAMPLES="${LATIUM_COVARIANCE_SAMPLES:-100000}"
MEM="${LATIUM_PAPER_MEM:-96gb}"
GPU_MEM="${LATIUM_PAPER_GPU_MEM:-40gb}"
SCRATCH="${LATIUM_PAPER_SCRATCH:-100gb}"
WALLTIME="${LATIUM_PAPER_WALLTIME:-10:00:00}"
NCPUS="${LATIUM_PAPER_NCPUS:-8}"
QUEUE="${LATIUM_PAPER_QUEUE:-}"
WANDB_PROJECT="${WANDB_PROJECT:-latium}"
WANDB_GROUP="${WANDB_GROUP:-paper-fleet-$(basename "$RUN_ROOT")}"
DRY_RUN=0
SMOKE=0
SKIP_SECOND_MOMENT=0
COVARIANCE_ONLY=0
WORKFLOW=paper
CASE_FILE=""
CASE_START=0
CASE_STOP=""
TRACKING=none
NO_GRAPHS=0
WALLTIME_EXPLICIT=0

MODELS=(
  deepseek-7b-base
  falcon-7b
  gemma-4-12b
  gpt2-xl
  granite-4.1-8b
  granite4-micro
  llama2-7b
  ministral-3-8b
  mistral-7b-v0.1
  mistral-7b-v0.3
  olmo-3-1025-7b
  opt-6.7b
  qwen3-8b
)

usage() {
  cat <<'EOF'
Usage: jobs/submit_paper_fleet.sh [options]

Submits one MetaCentrum PBS job per model. Use --smoke first to submit only
the small granite4-micro smoke test.

Options:
  --smoke                       granite4-micro, n=1, covariance=100000
  --run-root PATH               shared output directory
  --models MODEL [MODEL ...]    replace the default model list
  --workflow paper|gram         choose full paper or minimal Gram workflow
  --case-index-file PATH        fixed shared cohort (required for gram)
  --case-start N                zero-based manifest position (default 0)
  --case-stop N                 exclusive manifest position
  --tracking none|wandb         Gram tracking (default none)
  --no-graphs                   Gram: save artifacts without rendering
  --n-tests N                   ROME/structural cases (default 50)
  --covariance-samples N        second-moment samples (default 100000)
  --covariance-only             compute/verify matrices, then stop
  --mem SIZE                    host memory (default 96gb)
  --gpu-mem SIZE                minimum VRAM (default 40gb; Gemma uses 64gb)
  --scratch SIZE                local scratch (default 100gb)
  --walltime HH:MM:SS           PBS walltime (default 10:00:00)
  --skip-second-moment          require existing covariance; never recompute it
  --reuse-covariance             alias for --skip-second-moment
  --queue QUEUE                 optional PBS queue
  --dry-run                     print qsub commands without submitting
EOF
}

die() { echo "ERROR: $*" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      SMOKE=1
      MODELS=(granite4-micro)
      N_TESTS=1
      COVARIANCE_SAMPLES=100000
      MEM=64gb
      GPU_MEM=24gb
      SCRATCH=50gb
      WALLTIME=10:00:00
      shift
      ;;
    --run-root) RUN_ROOT="${2:?missing value for --run-root}"; shift 2 ;;
    --models)
      shift
      MODELS=()
      while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done
      [[ ${#MODELS[@]} -gt 0 ]] || die "--models requires at least one model"
      ;;
    --workflow) WORKFLOW="${2:?missing workflow}"; shift 2 ;;
    --case-index-file) CASE_FILE="${2:?missing manifest}"; shift 2 ;;
    --case-start) CASE_START="${2:?missing start}"; shift 2 ;;
    --case-stop) CASE_STOP="${2:?missing stop}"; shift 2 ;;
    --tracking) TRACKING="${2:?missing tracking}"; shift 2 ;;
    --no-graphs) NO_GRAPHS=1; shift ;;
    --n-tests) N_TESTS="${2:?missing value for --n-tests}"; shift 2 ;;
    --covariance-samples) COVARIANCE_SAMPLES="${2:?missing value for --covariance-samples}"; shift 2 ;;
    --covariance-only) COVARIANCE_ONLY=1; shift ;;
    --mem) MEM="${2:?missing value for --mem}"; shift 2 ;;
    --gpu-mem) GPU_MEM="${2:?missing value for --gpu-mem}"; shift 2 ;;
    --scratch) SCRATCH="${2:?missing value for --scratch}"; shift 2 ;;
    --walltime) WALLTIME="${2:?missing value for --walltime}"; WALLTIME_EXPLICIT=1; shift 2 ;;
    --skip-second-moment|--reuse-covariance)
      SKIP_SECOND_MOMENT=1
      shift
      ;;
    --queue) QUEUE="${2:?missing value for --queue}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option '$1'" ;;
  esac
done

if [[ "$QUEUE" == gpu_long || "$QUEUE" == gpu_long@* ]]; then
  die "gpu_long is disabled for the paper fleet; omit --queue to use normal routing"
fi

[[ "$WORKFLOW" == paper || "$WORKFLOW" == gram ]] || die "unknown workflow"
[[ "$TRACKING" == none || "$TRACKING" == wandb ]] || die "unknown tracking provider"
RUN_ROOT="$(realpath -m "$RUN_ROOT")"
if [[ "$WORKFLOW" == gram ]]; then
  [[ -f "$CASE_FILE" ]] || die "Gram requires an existing --case-index-file"
  CASE_FILE="$(realpath "$CASE_FILE")"
  [[ "$CASE_START" =~ ^[0-9]+$ && "$N_TESTS" =~ ^[0-9]+$ ]] || die "case range must use integer positions"
  CASE_STOP="${CASE_STOP:-$((CASE_START + N_TESTS))}"
  [[ "$CASE_STOP" =~ ^[0-9]+$ ]] || die "case stop must be an integer"
  (( CASE_STOP > CASE_START )) || die "case stop must be greater than start"
  N_TESTS=$((CASE_STOP - CASE_START))
  if (( ! DRY_RUN )); then
    "${LATIUM_PYTHON:-python}" "$ROOT/jobs/paper_fleet.py" --workflow gram --prepare-only \
      --run-root "$RUN_ROOT" --models "${MODELS[@]}" --case-index-file "$CASE_FILE" \
      --case-start "$CASE_START" --case-stop "$CASE_STOP" --covariance-samples "$COVARIANCE_SAMPLES"
  fi
elif [[ -n "$CASE_FILE" || "$CASE_START" != 0 || -n "$CASE_STOP" ]]; then
  die "manifest range options require --workflow gram"
fi
mkdir -p "$RUN_ROOT" "$ROOT/jobs/logs/paper-fleet"

for model in "${MODELS[@]}"; do
  model_slug="${model//[^a-zA-Z0-9_.-]/_}"
  model_gpu_mem="$GPU_MEM"
  model_mem="$MEM"
  model_walltime="$WALLTIME"
  if [[ "$model" == gemma-4-12b ]]; then
    model_gpu_mem="${LATIUM_GEMMA_GPU_MEM:-64gb}"
    model_mem="${LATIUM_GEMMA_MEM:-128gb}"
    if (( ! WALLTIME_EXPLICIT )); then
      model_walltime="${LATIUM_GEMMA_WALLTIME:-$WALLTIME}"
    fi
  fi

  model_log="$ROOT/jobs/logs/paper-fleet/${model_slug}.$(date +%Y%m%d-%H%M%S).log"
  args=(
    --workflow "$WORKFLOW"
    --run-root "$RUN_ROOT"
    --models "$model"
    --n-tests "$N_TESTS"
    --covariance-samples "$COVARIANCE_SAMPLES"
    --wandb-project "$WANDB_PROJECT"
    --wandb-group "$WANDB_GROUP"
    --worker
  )
  if [[ "$WORKFLOW" == gram ]]; then
    args+=(--case-index-file "$CASE_FILE" --case-start "$CASE_START" --case-stop "$CASE_STOP" --tracking "$TRACKING")
    if (( NO_GRAPHS )); then args+=(--no-graphs); fi
  fi
  if (( SKIP_SECOND_MOMENT )); then
    args+=(--skip-second-moment)
  fi
  if (( COVARIANCE_ONLY )); then
    args+=(--covariance-only)
  fi
  args_b64="$(printf '%s\0' "${args[@]}" | base64 | tr -d '\n')"
  select="select=1:ncpus=$NCPUS:mem=$model_mem:scratch_local=$SCRATCH:ngpus=1:gpu_mem=$model_gpu_mem"
  job_prefix="latium-pf"
  if (( COVARIANCE_ONLY )); then job_prefix="latium-cov"; fi
  qsub_cmd=(qsub -N "$job_prefix-$model_slug" -j oe -o "$model_log" -l "$select" -l "walltime=$model_walltime")
  [[ -z "$QUEUE" ]] || qsub_cmd+=(-q "$QUEUE")
  qsub_cmd+=(-v "LATIUM_REPO_ROOT=$ROOT,LATIUM_EXPECT_GPU=1,LATIUM_RUNNER=paper-fleet,LATIUM_ARG_COUNT=${#args[@]},LATIUM_ARGS_B64=$args_b64" "$ROOT/jobs/run.pbs")

  if (( DRY_RUN )); then
    printf '[dry-run] %-20s %s; walltime=%s\n' "$model" "$select" "$model_walltime"
    printf '  qsub:'; printf ' %q' "${qsub_cmd[@]}"; printf '\n'
  else
    command -v qsub >/dev/null 2>&1 || die "qsub is unavailable; submit from a MetaCentrum frontend"
    job_id="$("${qsub_cmd[@]}")"
    printf 'submitted %-20s %s\n' "$model" "$job_id"
  fi
done

echo "Run root: $RUN_ROOT"
echo "W&B project/group: $WANDB_PROJECT / $WANDB_GROUP"
if (( SMOKE )); then echo "Smoke test: granite4-micro only"; fi
if (( COVARIANCE_ONLY )); then echo "Mode: covariance only"; fi
