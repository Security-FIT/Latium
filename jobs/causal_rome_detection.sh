#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: jobs/causal_rome_detection.sh [options]

Run causal tracing, persist its held-out-confirmed layer into the selected
model config, ensure matching ROME covariance is available, apply ROME edits,
capture all architecture-neutral detector inputs, analyze them, render every
requested graph, and verify the resulting artifacts.

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
MODEL_STATE_PATH="$OUTPUT_ROOT/model-state.json"
COVARIANCE_STATE_PATH="$OUTPUT_ROOT/covariance.json"
MODEL_CONFIG_SNAPSHOT="$OUTPUT_ROOT/model-config-after-causal-trace.yaml"

MODEL_LOCK_SLUG="$(printf '%s' "$MODEL" | tr -c 'A-Za-z0-9._-' '_')"
mkdir -p jobs/logs
exec 9>"jobs/logs/.${MODEL_LOCK_SLUG}.pipeline.lock"
flock -n 9 || die "another pipeline in this checkout is updating model config '$MODEL'"

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
    "${TRACE_OVERRIDES[@]}" \
    "model=$MODEL" \
    "command.causal_trace.output_dir=$TRACE_ROOT" \
    "command.causal_trace.num_valid_facts=$TRACE_FACTS" \
    command.causal_trace.overwrite_model_config_layer=true
elif ! find "$TRACE_ROOT" -name summary.json -type f -print -quit | grep -q .; then
  die "--skip-causal-trace requested but no summary exists below $TRACE_ROOT"
fi

export LATIUM_PIPELINE_MODEL="$MODEL"
export LATIUM_PIPELINE_TRACE_ROOT="$TRACE_ROOT"
export LATIUM_PIPELINE_MODEL_STATE="$MODEL_STATE_PATH"
export LATIUM_PIPELINE_MODEL_SNAPSHOT="$MODEL_CONFIG_SNAPSHOT"
echo
echo "===== causal layer handoff ====="
python - <<'PY'
import json
import os
import tempfile
from pathlib import Path

from yaml import compose
from yaml.nodes import MappingNode, ScalarNode


def scalar_node(document: MappingNode, key: str) -> ScalarNode | None:
    found = None
    for key_node, value_node in document.value:
        if isinstance(key_node, ScalarNode) and key_node.value == key:
            if found is not None:
                raise SystemExit(f"Model config has duplicate top-level {key!r} keys")
            if not isinstance(value_node, ScalarNode):
                raise SystemExit(f"Model config top-level {key!r} must be a scalar")
            found = value_node
    return found


def replace_scalar(path: Path, key: str, replacement: str) -> str | None:
    text = path.read_text(encoding="utf-8")
    document = compose(text)
    if not isinstance(document, MappingNode):
        raise SystemExit(f"Model config must be a YAML mapping: {path}")
    node = scalar_node(document, key)
    if node is None:
        return None
    updated = f"{text[:node.start_mark.index]}{replacement}{text[node.end_mark.index:]}"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, path.stat().st_mode)
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return None if node.tag.endswith(":null") else node.value


trace_root = Path(os.environ["LATIUM_PIPELINE_TRACE_ROOT"])
summaries = sorted(trace_root.glob("*/summary.json"), key=lambda path: path.stat().st_mtime_ns)
if not summaries:
    raise SystemExit(f"Causal-trace summary is missing below {trace_root}")
summary_path = summaries[-1]
trace = json.loads(summary_path.read_text(encoding="utf-8"))
selected = trace.get("selected_trace_center")
if selected is None or not bool(trace.get("confirmation_passed")):
    raise SystemExit(
        "Causal tracing did not produce a held-out-confirmed layer: "
        f"{trace.get('selection_failure_reason') or trace.get('failure_reason') or summary_path}"
    )
selected = int(selected)

configured_path = str(trace.get("model_config_path") or "").strip()
fallback_path = Path("src/config/model") / f"{os.environ['LATIUM_PIPELINE_MODEL']}.yaml"
configured_candidate = Path(configured_path) if configured_path else None
config_path = configured_candidate if configured_candidate is not None and configured_candidate.is_file() else fallback_path
if not config_path.is_file():
    raise SystemExit(f"Selected model config is unavailable: {config_path}")
config_path = config_path.resolve()

text = config_path.read_text(encoding="utf-8")
document = compose(text)
if not isinstance(document, MappingNode):
    raise SystemExit(f"Model config must be a YAML mapping: {config_path}")
layer_node = scalar_node(document, "layer")
if layer_node is None:
    raise SystemExit(f"Model config has no top-level layer: {config_path}")
observed_layer = int(layer_node.value)
layer_reapplied = observed_layer != selected
if layer_reapplied:
    replace_scalar(config_path, "layer", str(selected))

# Explicit covariance paths in model YAMLs are layer-specific. Clear one after
# selecting the causal layer so all later stages discover only a matrix whose
# filename is keyed to the selected layer.
previous_second_moment_path = replace_scalar(config_path, "second_moment_path", "null")
snapshot_path = Path(os.environ["LATIUM_PIPELINE_MODEL_SNAPSHOT"])
snapshot_path.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")

trace.update(
    {
        "pipeline_model_config_layer_verified": True,
        "pipeline_model_config_layer_reapplied": layer_reapplied,
        "pipeline_second_moment_path_cleared": previous_second_moment_path is not None,
        "pipeline_previous_second_moment_path": previous_second_moment_path,
        "pipeline_model_config_snapshot": str(snapshot_path),
    }
)
summary_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

state = {
    "model": os.environ["LATIUM_PIPELINE_MODEL"],
    "selected_layer": selected,
    "confirmation_passed": True,
    "trace_summary": str(summary_path),
    "model_config_path": str(config_path),
    "model_config_snapshot": str(snapshot_path),
    "configured_layer_before_trace": trace.get("previous_model_config_layer"),
    "model_config_layer_after_trace": selected,
    "layer_reapplied_for_resume": layer_reapplied,
    "previous_second_moment_path": previous_second_moment_path,
    "second_moment_path_after_trace": None,
}
state_path = Path(os.environ["LATIUM_PIPELINE_MODEL_STATE"])
state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
print(json.dumps(state, indent=2))
PY

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

export LATIUM_PIPELINE_COVARIANCE_STATE="$COVARIANCE_STATE_PATH"
echo
echo "===== record selected-layer covariance ====="
python - <<'PY'
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from src.common.paths import resolve_project_path

state = json.loads(Path(os.environ["LATIUM_PIPELINE_MODEL_STATE"]).read_text(encoding="utf-8"))
base = OmegaConf.load("src/config/model_base/default.yaml")
selected_config = OmegaConf.load(state["model_config_path"])
cfg = OmegaConf.merge(base, selected_config)
layer = int(cfg.layer)
if layer != int(state["selected_layer"]):
    raise SystemExit(f"Covariance model layer {layer} != causal layer {state['selected_layer']}")
if str(getattr(cfg, "second_moment_path", "") or "").strip():
    raise SystemExit("Full pipeline requires model.second_moment_path=null after causal tracing")

directory = resolve_project_path(Path(str(cfg.second_moment_dir)))
model_id = str(cfg.name).replace("/", "_")
files = sorted(directory.glob(f"{model_id}_{layer}_*_*.pt"))
files += sorted(directory.glob(f"{model_id}_{layer}_*_*.npz"))
if not files:
    raise SystemExit(f"No saved covariance matches model={cfg.name} layer={layer} in {directory}")
for path in files:
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"Covariance file is missing or empty: {path}")

payload = {
    "model": os.environ["LATIUM_PIPELINE_MODEL"],
    "resolved_model": str(cfg.name),
    "selected_layer": layer,
    "second_moment_dir": str(directory),
    "files": [str(path) for path in files],
}
path = Path(os.environ["LATIUM_PIPELINE_COVARIANCE_STATE"])
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

STRUCTURAL_ARGS=(
  structural run
  "${STRUCTURAL_OVERRIDES[@]}"
  "structural.run.models=[$MODEL]"
  "structural.run.n_tests=$DETECTION_CASES"
  "structural.run.start_idx=$START_IDX"
  "structural.run.output_dir=$OUTPUT_ROOT"
  "structural.run.run_id=$RUN_ID"
  structural.run.fail_on_missing_second_moment=true
  structural.capture.profile=rome-presence
  "structural.capture.enable=[spectral]"
  structural.analysis.preset=rome-presence
  "structural.analysis.enable=[spectral]"
  structural.render.enabled=true
  structural.render.renderer_preset=rome-presence
  "structural.render.enable=[detector,detector-signals]"
)
if [[ "$FORCE" == 1 ]]; then
  STRUCTURAL_ARGS+=(structural.run.force=true)
fi
run_stage "ROME, detection, and rendering" python -m src "${STRUCTURAL_ARGS[@]}"

export LATIUM_PIPELINE_RUN_ROOT="$RUN_ROOT"
export LATIUM_PIPELINE_OUTPUT_ROOT="$OUTPUT_ROOT"
python - <<'PY'
import json
import os
from pathlib import Path

from src.results import RunArtifactReader

run_root = Path(os.environ["LATIUM_PIPELINE_RUN_ROOT"])
trace_root = Path(os.environ["LATIUM_PIPELINE_TRACE_ROOT"])
output_root = Path(os.environ["LATIUM_PIPELINE_OUTPUT_ROOT"])
model_state_path = Path(os.environ["LATIUM_PIPELINE_MODEL_STATE"])
covariance_state_path = Path(os.environ["LATIUM_PIPELINE_COVARIANCE_STATE"])
manifest_path = run_root / "manifest.json"
if not manifest_path.is_file():
    raise SystemExit(f"Structural manifest is missing: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
records = list(manifest.get("artifacts", {}).values())
model_state = json.loads(model_state_path.read_text(encoding="utf-8"))
covariance = json.loads(covariance_state_path.read_text(encoding="utf-8"))
selected_layer = int(model_state["selected_layer"])

rome_executions = [
    record
    for record in records
    if record.get("kind") == "execution" and record.get("producer") == "rome"
]
if not rome_executions or any(record.get("status") != "complete" for record in rome_executions):
    raise SystemExit("ROME execution artifacts are missing or incomplete")
reader = RunArtifactReader(run_root)
execution_layers = {
    int(reader.load(str(record["artifact_id"])).get("summary", {}).get("target_layer", -1))
    for record in rome_executions
}
if execution_layers != {selected_layer}:
    raise SystemExit(f"ROME execution layers {sorted(execution_layers)} != causal layer {selected_layer}")

required_captures = {"weighted-spectrum", "rome-update", "spectral"}
edited_capture_records = [
    record
    for record in records
    if record.get("kind") == "capture"
    and record.get("edit_method") == "rome"
    and record.get("producer") in required_captures
]
failed_captures = [
    str(record.get("artifact_id"))
    for record in edited_capture_records
    if record.get("status") != "complete"
]
if failed_captures:
    raise SystemExit(f"Edited capture artifacts are incomplete: {', '.join(sorted(failed_captures))}")
complete_captures = {
    str(record.get("producer"))
    for record in edited_capture_records
}
missing_captures = sorted(required_captures - complete_captures)
if missing_captures:
    raise SystemExit(f"Required edited captures are not complete: {', '.join(missing_captures)}")

required_analyses = {
    "spectral",
    "weighted-spectrum",
    "rome-presence-blind-peak",
    "rome-presence-blind-footprint",
    "rome-presence-delta",
}
analysis_records = [
    record
    for record in records
    if record.get("kind") == "analysis"
    and record.get("edit_method") == "rome"
    and record.get("producer") in required_analyses
]
failed_analyses = [
    str(record.get("artifact_id"))
    for record in analysis_records
    if record.get("status") != "complete"
]
if failed_analyses:
    raise SystemExit(f"Detector analysis artifacts are incomplete: {', '.join(sorted(failed_analyses))}")
complete_analyses = {
    str(record.get("producer"))
    for record in analysis_records
}
missing = sorted(required_analyses - complete_analyses)
if missing:
    raise SystemExit(f"Required analyses are not complete: {', '.join(missing)}")

required_renderers = {
    "rome-detector-explainer",
    "rome-success",
    "detector-window",
    "detector",
    "detector-signals",
}
render_artifacts = {}
for renderer in sorted(required_renderers):
    render_path = run_root / "graphs" / renderer / "artifact.json"
    if not render_path.is_file():
        raise SystemExit(f"Required render artifact is missing: {render_path}")
    render = json.loads(render_path.read_text(encoding="utf-8"))
    if render.get("status") != "complete":
        raise SystemExit(f"Renderer {renderer} did not complete: {render.get('error')}")
    outputs = list(render.get("summary", {}).get("outputs", []))
    if not outputs:
        raise SystemExit(f"Renderer {renderer} produced no outputs")
    for output in outputs:
        output_path = Path(output)
        output_path = output_path if output_path.is_absolute() else run_root / output_path
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise SystemExit(f"Renderer output is missing or empty: {output_path}")
    render_artifacts[renderer] = {
        "artifact": str(render_path),
        "outputs": outputs,
    }

trace_summaries = sorted(trace_root.glob("*/summary.json"), key=lambda path: path.stat().st_mtime)
if not trace_summaries:
    raise SystemExit(f"Causal-trace summary is missing below {trace_root}")
trace_summary_path = trace_summaries[-1]
trace = json.loads(trace_summary_path.read_text(encoding="utf-8"))
if int(trace.get("num_valid_facts", 0)) <= 0:
    raise SystemExit(f"Causal tracing produced no valid facts: {trace_summary_path}")
if not bool(trace.get("confirmation_passed")) or int(trace.get("selected_trace_center", -1)) != selected_layer:
    raise SystemExit("Saved causal trace is not confirmed or does not match the pipeline layer")
trace_plot = Path(str(trace.get("plot", "")))
if not trace_plot.is_file() or trace_plot.stat().st_size <= 0:
    raise SystemExit(f"Causal-trace graph is missing or empty: {trace_plot}")

summary = {
    "schema": "latium.causal_rome_detection_job.v2",
    "model": os.environ["LATIUM_PIPELINE_MODEL"],
    "selected_layer": selected_layer,
    "causal_trace_summary": str(trace_summary_path),
    "causal_trace_plot": str(trace_plot),
    "causal_trace_selected_center": trace.get("selected_trace_center"),
    "causal_trace_confirmation_passed": trace.get("confirmation_passed"),
    "model_state": str(model_state_path),
    "model_config_path": model_state.get("model_config_path"),
    "model_config_snapshot": model_state.get("model_config_snapshot"),
    "covariance_state": str(covariance_state_path),
    "covariance_files": covariance.get("files", []),
    "structural_run_root": str(run_root),
    "structural_manifest": str(manifest_path),
    "rome_execution_artifacts": sorted(str(record["artifact_id"]) for record in rome_executions),
    "completed_captures": sorted(complete_captures),
    "completed_analyses": sorted(complete_analyses),
    "render_artifacts": render_artifacts,
}
summary_path = output_root / "pipeline-summary.json"
summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo
echo "Pipeline complete: $OUTPUT_ROOT"
