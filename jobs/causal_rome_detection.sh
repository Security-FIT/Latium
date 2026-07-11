#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: jobs/causal_rome_detection.sh [options]

Run causal tracing, ensure ROME covariance is available, apply ROME edits,
capture all architecture-neutral detector inputs, analyze them, render the
detector explanation, and verify the resulting artifacts.

Options:
  --model KEY                    Model config key (default: gpt2-large)
  --trace-facts N                Accepted causal-trace facts (default: 30)
  --detection-cases N            CounterFact ROME cases (default: 30)
  --start-idx N                  First structural case (default: 0)
  --second-moment-samples N      Samples when covariance is missing (default: 100000)
  --output-root PATH             Pipeline output root
  --run-id ID                    Structural run ID below output root (default: detection)
  --skip-causal-trace            Resume without rerunning causal tracing
  --skip-second-moment           Fail instead of producing missing covariance
  --force                        Recompute structural artifacts
  --trace-override VALUE         Extra causal-trace Hydra override; repeatable
  --structural-override VALUE    Extra structural Hydra override; repeatable
  -h, --help                     Show this help
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL=gpt2-large
TRACE_FACTS=30
DETECTION_CASES=30
START_IDX=0
SECOND_MOMENT_SAMPLES=100000
JOB_SLUG="$(printf '%s' "${PBS_JOBID:-local}" | tr -c 'A-Za-z0-9._-' '_')"
OUTPUT_ROOT="analysis_out/jobs/${JOB_SLUG}-causal-rome-detection"
RUN_ID=detection
SKIP_TRACE=0
SKIP_SECOND_MOMENT=0
FORCE=0
TRACE_OVERRIDES=()
STRUCTURAL_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:?missing value for --model}"; shift 2 ;;
    --trace-facts) TRACE_FACTS="${2:?missing value for --trace-facts}"; shift 2 ;;
    --detection-cases) DETECTION_CASES="${2:?missing value for --detection-cases}"; shift 2 ;;
    --start-idx) START_IDX="${2:?missing value for --start-idx}"; shift 2 ;;
    --second-moment-samples)
      SECOND_MOMENT_SAMPLES="${2:?missing value for --second-moment-samples}"
      shift 2
      ;;
    --output-root) OUTPUT_ROOT="${2:?missing value for --output-root}"; shift 2 ;;
    --run-id) RUN_ID="${2:?missing value for --run-id}"; shift 2 ;;
    --skip-causal-trace) SKIP_TRACE=1; shift ;;
    --skip-second-moment) SKIP_SECOND_MOMENT=1; shift ;;
    --force) FORCE=1; shift ;;
    --trace-override) TRACE_OVERRIDES+=("${2:?missing value for --trace-override}"); shift 2 ;;
    --structural-override) STRUCTURAL_OVERRIDES+=("${2:?missing value for --structural-override}"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown pipeline option '$1'" ;;
  esac
done

[[ "$TRACE_FACTS" =~ ^[1-9][0-9]*$ ]] || die "--trace-facts must be positive"
[[ "$DETECTION_CASES" =~ ^[1-9][0-9]*$ ]] || die "--detection-cases must be positive"
[[ "$START_IDX" =~ ^[0-9]+$ ]] || die "--start-idx must be non-negative"
[[ "$SECOND_MOMENT_SAMPLES" =~ ^[1-9][0-9]*$ ]] || die "--second-moment-samples must be positive"

mkdir -p "$OUTPUT_ROOT"
TRACE_ROOT="$OUTPUT_ROOT/causal-trace"
RUN_ROOT="$OUTPUT_ROOT/$RUN_ID"

run_stage() {
  local name="$1"
  shift
  echo
  echo "===== $name ====="
  printf 'command:'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

if [[ "$SKIP_TRACE" == 0 ]]; then
  run_stage "causal tracing" \
    python -m src causal-trace \
    "model=$MODEL" \
    "command.causal_trace.output_dir=$TRACE_ROOT" \
    "command.causal_trace.num_valid_facts=$TRACE_FACTS" \
    "${TRACE_OVERRIDES[@]}"
elif ! find "$TRACE_ROOT" -name summary.json -type f -print -quit | grep -q .; then
  die "--skip-causal-trace requested but no summary exists below $TRACE_ROOT"
fi

VALIDATE_ARGS=(
  structural validate-cov
  "structural.run.models=[$MODEL]"
  structural.validate_cov.fail_missing=true
)
if ! python -m src "${VALIDATE_ARGS[@]}"; then
  [[ "$SKIP_SECOND_MOMENT" == 0 ]] || die "ROME covariance is missing and --skip-second-moment was requested"
  run_stage "second-moment covariance" \
    python -m src second-moment \
    "model=$MODEL" \
    "model.second_moment_target_samples=$SECOND_MOMENT_SAMPLES"
  run_stage "validate covariance" python -m src "${VALIDATE_ARGS[@]}"
fi

STRUCTURAL_ARGS=(
  structural run
  "structural.run.models=[$MODEL]"
  "structural.run.n_tests=$DETECTION_CASES"
  "structural.run.start_idx=$START_IDX"
  "structural.run.output_dir=$OUTPUT_ROOT"
  "structural.run.run_id=$RUN_ID"
  structural.run.fail_on_missing_second_moment=true
  structural.capture.profile=rome-presence
  structural.analysis.preset=rome-presence
  structural.render.enabled=true
  structural.render.renderer_preset=rome-presence
)
if [[ "$FORCE" == 1 ]]; then
  STRUCTURAL_ARGS+=(structural.run.force=true)
fi
STRUCTURAL_ARGS+=("${STRUCTURAL_OVERRIDES[@]}")
run_stage "ROME, detection, and rendering" python -m src "${STRUCTURAL_ARGS[@]}"

export LATIUM_PIPELINE_RUN_ROOT="$RUN_ROOT"
export LATIUM_PIPELINE_TRACE_ROOT="$TRACE_ROOT"
export LATIUM_PIPELINE_MODEL="$MODEL"
export LATIUM_PIPELINE_OUTPUT_ROOT="$OUTPUT_ROOT"
python - <<'PY'
import json
import os
from pathlib import Path

run_root = Path(os.environ["LATIUM_PIPELINE_RUN_ROOT"])
trace_root = Path(os.environ["LATIUM_PIPELINE_TRACE_ROOT"])
output_root = Path(os.environ["LATIUM_PIPELINE_OUTPUT_ROOT"])
manifest_path = run_root / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"Structural manifest is missing: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
records = list(manifest.get("artifacts", {}).values())
required_analyses = {
    "weighted-spectrum",
    "rome-presence-blind-peak",
    "rome-presence-blind-footprint",
    "rome-presence-delta",
}
complete_analyses = {
    str(record.get("producer"))
    for record in records
    if record.get("kind") == "analysis" and record.get("status") == "complete"
}
missing = sorted(required_analyses - complete_analyses)
if missing:
    raise SystemExit(f"Required analyses are not complete: {', '.join(missing)}")

render_path = run_root / "graphs" / "rome-detector-explainer" / "artifact.json"
if not render_path.is_file():
    raise SystemExit(f"Detector render artifact is missing: {render_path}")
render = json.loads(render_path.read_text(encoding="utf-8"))
if render.get("status") != "complete":
    raise SystemExit(f"Detector renderer did not complete: {render.get('error')}")

trace_summaries = sorted(trace_root.glob("*/summary.json"), key=lambda path: path.stat().st_mtime)
if not trace_summaries:
    raise SystemExit(f"Causal-trace summary is missing below {trace_root}")
trace_summary_path = trace_summaries[-1]
trace = json.loads(trace_summary_path.read_text(encoding="utf-8"))
if int(trace.get("num_valid_facts", 0)) <= 0:
    raise SystemExit(f"Causal tracing produced no valid facts: {trace_summary_path}")

summary = {
    "schema": "latium.causal_rome_detection_job.v1",
    "model": os.environ["LATIUM_PIPELINE_MODEL"],
    "causal_trace_summary": str(trace_summary_path),
    "causal_trace_selected_center": trace.get("selected_trace_center"),
    "causal_trace_confirmation_passed": trace.get("confirmation_passed"),
    "structural_run_root": str(run_root),
    "completed_analyses": sorted(complete_analyses),
    "detector_render_artifact": str(render_path),
    "detector_outputs": render.get("summary", {}).get("outputs", []),
}
summary_path = output_root / "pipeline-summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo
echo "Pipeline complete: $OUTPUT_ROOT"
