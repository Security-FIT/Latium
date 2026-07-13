#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: jobs/run_saved_models_pipeline.sh [options]

Run the causal-trace -> covariance -> ROME -> weighted/spectral detection
pipeline sequentially for every saved model. Completed model summaries are
skipped on resume, and one model failure does not prevent later models running.

Options:
  --model KEY                    Add one model; repeatable
  --edits N                      Override pipeline.structural.n_tests
  --trace-facts N                Override pipeline.causal_trace.num_valid_facts
  --start-idx N                  Override pipeline.structural.start_idx
  --second-moment-samples N      Override pipeline.covariance.target_samples
  --output-root PATH             Shared all-model output root
  --skip-causal-trace            Require/resume existing trace summaries
  --skip-second-moment           Require existing covariance files
  --force                        Recompute structural artifacts
  --fail-fast                    Stop after the first failed model
  --trace-override VALUE         Forwarded per-model trace override; repeatable
  --structural-override VALUE    Forwarded per-model structural override; repeatable
  -h, --help                     Show this help

Default saved-model keys: gpt2-medium, gpt2-large, gpt2-xl, qwen3-4b,
qwen3-8b. Counts default to the Hydra pipeline configuration.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODELS=()
EDITS=""
TRACE_FACTS=""
START_IDX=""
SECOND_MOMENT_SAMPLES=""
OUTPUT_ROOT="analysis_out/remote-all-models"
SKIP_TRACE=0
SKIP_SECOND_MOMENT=0
FORCE=0
FAIL_FAST=0
TRACE_OVERRIDES=()
STRUCTURAL_OVERRIDES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODELS+=("${2:?missing value for --model}"); shift 2 ;;
    --edits) EDITS="${2:?missing value for --edits}"; shift 2 ;;
    --trace-facts) TRACE_FACTS="${2:?missing value for --trace-facts}"; shift 2 ;;
    --start-idx) START_IDX="${2:?missing value for --start-idx}"; shift 2 ;;
    --second-moment-samples)
      SECOND_MOMENT_SAMPLES="${2:?missing value for --second-moment-samples}"
      shift 2
      ;;
    --output-root) OUTPUT_ROOT="${2:?missing value for --output-root}"; shift 2 ;;
    --skip-causal-trace) SKIP_TRACE=1; shift ;;
    --skip-second-moment) SKIP_SECOND_MOMENT=1; shift ;;
    --force) FORCE=1; shift ;;
    --fail-fast) FAIL_FAST=1; shift ;;
    --trace-override) TRACE_OVERRIDES+=("${2:?missing value for --trace-override}"); shift 2 ;;
    --structural-override) STRUCTURAL_OVERRIDES+=("${2:?missing value for --structural-override}"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown all-model pipeline option '$1'" ;;
  esac
done

[[ -z "$EDITS" || "$EDITS" =~ ^[1-9][0-9]*$ ]] || die "--edits must be positive"
[[ -z "$TRACE_FACTS" || "$TRACE_FACTS" =~ ^[1-9][0-9]*$ ]] || die "--trace-facts must be positive"
[[ -z "$START_IDX" || "$START_IDX" =~ ^[0-9]+$ ]] || die "--start-idx must be non-negative"
[[ -z "$SECOND_MOMENT_SAMPLES" || "$SECOND_MOMENT_SAMPLES" =~ ^[1-9][0-9]*$ ]] || die "--second-moment-samples must be positive"
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(gpt2-medium gpt2-large gpt2-xl qwen3-4b qwen3-8b)
fi

mkdir -p "$OUTPUT_ROOT/logs"
exec 9>"$OUTPUT_ROOT/.pipeline.lock"
flock -n 9 || die "another all-model pipeline already owns $OUTPUT_ROOT"

STATUS_FILE="$OUTPUT_ROOT/status.tsv"
SUMMARY_FILE="$OUTPUT_ROOT/all-model-summary.json"
printf 'timestamp\tmodel\tstatus\texit_code\toutput\n' >"$STATUS_FILE"

echo "All-model pipeline"
echo "models: ${MODELS[*]}"
echo "edits/model: ${EDITS:-Hydra default}"
echo "trace facts/model: ${TRACE_FACTS:-Hydra default}"
echo "output: $OUTPUT_ROOT"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

declare -a FAILED_MODELS=()
for model in "${MODELS[@]}"; do
  model_output="$OUTPUT_ROOT/$model"
  model_summary="$model_output/pipeline-summary.json"
  model_log="$OUTPUT_ROOT/logs/$model.log"
  if [[ "$FORCE" == 0 && -s "$model_summary" ]]; then
    printf '%s\t%s\tcomplete-resumed\t0\t%s\n' \
      "$(date --iso-8601=seconds)" "$model" "$model_output" >>"$STATUS_FILE"
    echo "Skipping completed model: $model"
    continue
  fi

  args=(
    --model "$model"
    --output-root "$model_output"
  )
  [[ -z "$TRACE_FACTS" ]] || args+=(--trace-facts "$TRACE_FACTS")
  [[ -z "$EDITS" ]] || args+=(--detection-cases "$EDITS")
  [[ -z "$START_IDX" ]] || args+=(--start-idx "$START_IDX")
  [[ -z "$SECOND_MOMENT_SAMPLES" ]] || args+=(--second-moment-samples "$SECOND_MOMENT_SAMPLES")
  [[ "$SKIP_TRACE" == 0 ]] || args+=(--skip-causal-trace)
  [[ "$SKIP_SECOND_MOMENT" == 0 ]] || args+=(--skip-second-moment)
  [[ "$FORCE" == 0 ]] || args+=(--force)
  for value in "${TRACE_OVERRIDES[@]}"; do
    args+=(--trace-override "$value")
  done
  for value in "${STRUCTURAL_OVERRIDES[@]}"; do
    args+=(--structural-override "$value")
  done

  echo
  echo "===== model: $model ====="
  printf 'command: bash jobs/causal_rome_detection.sh'
  printf ' %q' "${args[@]}"
  printf '\nlog: %s\n' "$model_log"
  set +e
  bash jobs/causal_rome_detection.sh "${args[@]}" 2>&1 | tee "$model_log"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -eq 0 && -s "$model_summary" ]]; then
    status=complete
  else
    status=failed
    FAILED_MODELS+=("$model")
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$(date --iso-8601=seconds)" "$model" "$status" "$rc" "$model_output" >>"$STATUS_FILE"
  if [[ $rc -ne 0 && "$FAIL_FAST" == 1 ]]; then
    break
  fi
done

export LATIUM_ALL_MODEL_OUTPUT_ROOT="$OUTPUT_ROOT"
LATIUM_ALL_MODEL_KEYS="$(IFS=,; echo "${MODELS[*]}")"
export LATIUM_ALL_MODEL_KEYS
python - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["LATIUM_ALL_MODEL_OUTPUT_ROOT"])
models = [value for value in os.environ["LATIUM_ALL_MODEL_KEYS"].split(",") if value]
rows = list(csv.DictReader((root / "status.tsv").open(encoding="utf-8"), delimiter="\t"))
latest = {row["model"]: row for row in rows}
results = []
for model in models:
    summary_path = root / model / "pipeline-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    status = latest.get(model, {}).get("status", "not-run")
    results.append({
        "model": model,
        "status": status,
        "pipeline_summary": str(summary_path) if summary is not None else None,
        "completed_analyses": summary.get("completed_analyses", []) if summary else [],
    })
payload = {
    "schema": "latium.all_saved_models_pipeline.v1",
    "models": results,
    "complete": all(item["status"].startswith("complete") for item in results),
}
(root / "all-model-summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
  echo "Failed models: ${FAILED_MODELS[*]}" >&2
  exit 1
fi
echo "All saved-model pipelines complete: $SUMMARY_FILE"
