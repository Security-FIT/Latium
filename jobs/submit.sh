#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  sed -n '1,130p' "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/README.md"
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ $# -gt 0 ]] || { usage; exit 2; }
case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

PRESET="$1"
shift

NCPUS=4
MEM=64gb
NGPUS=1
GPU_MEM=24gb
GPU_CAP=""
SCRATCH=50gb
WALLTIME=24:00:00
QUEUE=""
DEPEND=""
DRY_RUN=0
DEFAULT_ARGS=()
RUNNER=latium

case "$PRESET" in
  causal-trace)
    JOB_NAME=latium-trace
    DEFAULT_ARGS=(causal-trace model=gpt2-large
      command.causal_trace.output_dir=analysis_out/jobs/__PBS_JOBID__/causal-trace)
    ;;
  rome)
    JOB_NAME=latium-rome
    DEFAULT_ARGS=(rome-benchmark 'rome_benchmark.models=[gpt2-large]'
      rome_benchmark.output_dir=analysis_out/jobs/__PBS_JOBID__/rome)
    ;;
  detectors)
    JOB_NAME=latium-detect
    WALLTIME=36:00:00
    DEFAULT_ARGS=(structural run 'structural.run.models=[gpt2-large]'
      structural.run.n_tests=30 structural.capture.profile=rome-presence
      structural.analysis.preset=rome-presence structural.render.enabled=true
      structural.render.renderer_preset=rome-presence
      structural.run.output_dir=analysis_out/jobs
      structural.run.run_id=__PBS_JOBID__-detectors)
    ;;
  causal-rome-detection)
    JOB_NAME=latium-full-detect
    MEM=96gb
    GPU_MEM=40gb
    SCRATCH=100gb
    WALLTIME=72:00:00
    RUNNER=causal-rome-detection
    DEFAULT_ARGS=(--model gpt2-large --trace-facts 30 --detection-cases 30)
    ;;
  second-moment)
    JOB_NAME=latium-cov
    WALLTIME=48:00:00
    DEFAULT_ARGS=(second-moment model=gpt2-large)
    ;;
  analyze)
    JOB_NAME=latium-analyze
    NCPUS=8
    MEM=64gb
    NGPUS=0
    GPU_MEM=""
    SCRATCH=20gb
    WALLTIME=12:00:00
    DEFAULT_ARGS=(structural analyze structural.analysis.preset=paper)
    ;;
  custom)
    JOB_NAME=latium-custom
    DEFAULT_ARGS=()
    ;;
  *) die "unknown preset '$PRESET' (use causal-trace, rome, detectors, causal-rome-detection, second-moment, analyze, or custom)" ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ncpus) NCPUS="${2:?missing value for --ncpus}"; shift 2 ;;
    --mem) MEM="${2:?missing value for --mem}"; shift 2 ;;
    --ngpus) NGPUS="${2:?missing value for --ngpus}"; shift 2 ;;
    --gpu-mem) GPU_MEM="${2-}"; shift 2 ;;
    --gpu-cap) GPU_CAP="${2-}"; shift 2 ;;
    --scratch) SCRATCH="${2:?missing value for --scratch}"; shift 2 ;;
    --walltime) WALLTIME="${2:?missing value for --walltime}"; shift 2 ;;
    --queue) QUEUE="${2:?missing value for --queue}"; shift 2 ;;
    --name) JOB_NAME="${2:?missing value for --name}"; shift 2 ;;
    --afterok) DEPEND="${2:?missing value for --afterok}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; break ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown submission option '$1'; put Hydra arguments after --" ;;
  esac
done

ARGS=("${DEFAULT_ARGS[@]}" "$@")
[[ ${#ARGS[@]} -gt 0 ]] || die "custom requires a Latium command after --"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/jobs/logs"
LOG_FILE="$ROOT/jobs/logs/${JOB_NAME}.$(date +%Y%m%d-%H%M%S).$$.log"

SELECT="select=1:ncpus=$NCPUS:mem=$MEM:scratch_local=$SCRATCH"
if [[ "$NGPUS" != "0" ]]; then
  SELECT+=":ngpus=$NGPUS"
  [[ -z "$GPU_MEM" ]] || SELECT+=":gpu_mem=$GPU_MEM"
  [[ -z "$GPU_CAP" ]] || SELECT+=":gpu_cap=$GPU_CAP"
fi

ARGS_B64="$(printf '%s\0' "${ARGS[@]}" | base64 | tr -d '\n')"
QSUB=(qsub -N "$JOB_NAME" -j oe -o "$LOG_FILE" -l "$SELECT" -l "walltime=$WALLTIME")
[[ -z "$QUEUE" ]] || QSUB+=(-q "$QUEUE")
[[ -z "$DEPEND" ]] || QSUB+=(-W "depend=afterok:$DEPEND")
QSUB+=(-v "LATIUM_REPO_ROOT=$ROOT,LATIUM_EXPECT_GPU=$([[ "$NGPUS" == 0 ]] && echo 0 || echo 1),LATIUM_RUNNER=$RUNNER,LATIUM_ARGS_B64=$ARGS_B64")
QSUB+=("$ROOT/jobs/run.pbs")

if [[ "$DRY_RUN" == 1 ]]; then
  echo "preset: $PRESET"
  echo "resources: $SELECT; walltime=$WALLTIME${QUEUE:+; queue=$QUEUE}"
  printf 'command:'
  if [[ "$RUNNER" == "latium" ]]; then
    printf ' %q' python -m src "${ARGS[@]}"
  else
    printf ' %q' bash "$ROOT/jobs/causal_rome_detection.sh" "${ARGS[@]}"
  fi
  printf '\nqsub:'
  printf ' %q' "${QSUB[@]}"
  printf '\n'
  exit 0
fi

command -v qsub >/dev/null 2>&1 || die "qsub is unavailable; submit from a MetaCentrum frontend"
cd "$ROOT"
"${QSUB[@]}"
