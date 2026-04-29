#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$(basename "$SCRIPT_DIR")" == "scripts_for_graphs" ]]; then
  IMPORT_ROOT="$SCRIPT_DIR"
  BUNDLE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  GRAPH_SCRIPT_DIR="$SCRIPT_DIR/scripts/bundle_graphs"
elif [[ "$(basename "$SCRIPT_DIR")" == "bundle_graphs" && "$(basename "$(dirname "$(dirname "$SCRIPT_DIR")")")" == "scripts_for_graphs" ]]; then
  IMPORT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
  BUNDLE_ROOT="$(cd "$IMPORT_ROOT/.." && pwd)"
  GRAPH_SCRIPT_DIR="$SCRIPT_DIR"
else
  IMPORT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
  BUNDLE_ROOT="$IMPORT_ROOT/final_n500_bundle"
  GRAPH_SCRIPT_DIR="$SCRIPT_DIR"
fi

PYTHON_BIN="${PYTHON:-python}"
WINDOW_RADIUS=1
RUN_FINALIZE=true
RUN_FINAL_PER_MODEL=true
PAPER_DIR=""
PAPER_IMG_DETECTION=""
PAPER_IMG_ARTIFACTS=""

usage() {
  cat <<'EOF'
Usage: run_all_graphs.sh [options]

Options:
  --bundle-root <path>       final_n500_bundle root (default: auto-detected)
  --python <path>            Python executable (default: python or $PYTHON)
  --window-radius <int>      Detector layer window radius (default: 1)
  --skip-finalize            Do not refresh bundle indices/scripts_for_graphs
  --skip-final-per-model     Do not rebuild per-model paper/detector graphs
  --paper-dir <path>         Optional paper root for cohort figure copies
  --paper-img-detection <p>  Optional paper img/detection copy target dir for artifact grids
  --paper-img-artifacts <p>  Optional paper img/artifacts copy target dir for artifact grids
  -h, --help                 Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-root) BUNDLE_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --window-radius) WINDOW_RADIUS="$2"; shift 2 ;;
    --skip-finalize) RUN_FINALIZE=false; shift ;;
    --skip-final-per-model) RUN_FINAL_PER_MODEL=false; shift ;;
    --paper-dir) PAPER_DIR="$2"; shift 2 ;;
    --paper-img-detection) PAPER_IMG_DETECTION="$2"; shift 2 ;;
    --paper-img-artifacts) PAPER_IMG_ARTIFACTS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ ! -d "$BUNDLE_ROOT" ]]; then
  echo "ERROR: bundle root does not exist: $BUNDLE_ROOT" >&2
  exit 1
fi
BUNDLE_ROOT="$(cd "$BUNDLE_ROOT" && pwd)"

export PYTHONPATH="$IMPORT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-codex}"

run_py() {
  echo "[bundle-graphs] $*"
  "$PYTHON_BIN" "$@"
}

run_final_per_model() {
  local final_root="$BUNDLE_ROOT/data/final_n500_models_only"
  if [[ ! -d "$final_root" ]]; then
    echo "ERROR: missing final model data root: $final_root" >&2
    exit 1
  fi

  while IFS= read -r run_root; do
    local model
    model="$(basename "$run_root")"
    run_py "$IMPORT_ROOT/pipeline_posthoc.py" \
      --local-run-dir "$run_root" \
      --models "$model" \
      --n-tests 500 \
      --start-idx 0 \
      --slice-policy shared \
      --success-file "$run_root/success.txt"

    mkdir -p "$BUNDLE_ROOT/graphs/final_per_model/$model"
    if [[ -d "$run_root/paper_graphs" ]]; then
      rm -rf "$BUNDLE_ROOT/graphs/final_per_model/$model"
      mkdir -p "$BUNDLE_ROOT/graphs/final_per_model"
      cp -a "$run_root/paper_graphs" "$BUNDLE_ROOT/graphs/final_per_model/$model"
    fi
  done < <(find "$final_root" -mindepth 1 -maxdepth 1 -type d | sort)
}

if $RUN_FINAL_PER_MODEL; then
  run_final_per_model
fi

run_py "$GRAPH_SCRIPT_DIR/render_bundle_rome_success_metrics.py" --bundle-root "$BUNDLE_ROOT"
run_py "$GRAPH_SCRIPT_DIR/render_bundle_fleet_summary.py" --bundle-root "$BUNDLE_ROOT"
run_py "$GRAPH_SCRIPT_DIR/render_bundle_prefixtest_graphs.py" --bundle-root "$BUNDLE_ROOT"

WINDOWED_DIR="$BUNDLE_ROOT/graphs/windowed_detector"
mkdir -p "$WINDOWED_DIR"
run_py "$IMPORT_ROOT/detector/composite_detector_v2.py" \
  "$BUNDLE_ROOT/data/fleet_single_edit_runs/fleet_out_remote" \
  --windowed-detector \
  --latest-only \
  --window-radius "$WINDOW_RADIUS" \
  --json-out "$WINDOWED_DIR/fleet_windowed_detector.json" \
  --csv-out "$WINDOWED_DIR/fleet_windowed_detector.csv"
run_py "$GRAPH_SCRIPT_DIR/render_bundle_windowed_detector_summary.py" --bundle-root "$BUNDLE_ROOT"

run_py "$GRAPH_SCRIPT_DIR/render_bundle_detector_stacked_variants.py" --bundle-root "$BUNDLE_ROOT"
run_py "$GRAPH_SCRIPT_DIR/render_final_n500_detector_layer_window.py" --bundle-root "$BUNDLE_ROOT" --window-radius "$WINDOW_RADIUS"
run_py "$GRAPH_SCRIPT_DIR/export_qwen_paper_graph_jsons.py" --bundle-root "$BUNDLE_ROOT"

cohort_args=(--bundle-root "$BUNDLE_ROOT")
if [[ -n "$PAPER_DIR" ]]; then
  cohort_args+=(--paper-dir "$PAPER_DIR")
fi
run_py "$GRAPH_SCRIPT_DIR/render_cohort_envelope_figure.py" "${cohort_args[@]}"

fp_args=(--bundle-root "$BUNDLE_ROOT")
if [[ -n "$PAPER_DIR" ]]; then
  fp_args+=(--paper-img "$PAPER_DIR/img/detection/cohort_fp_combined.pdf")
fi
run_py "$GRAPH_SCRIPT_DIR/render_cohort_fp_figure.py" "${fp_args[@]}"

artifact_args=(--bundle-root "$BUNDLE_ROOT")
if [[ -n "$PAPER_IMG_DETECTION" ]]; then
  artifact_args+=(--paper-img-detection "$PAPER_IMG_DETECTION")
fi
if [[ -n "$PAPER_IMG_ARTIFACTS" ]]; then
  artifact_args+=(--paper-img-artifacts "$PAPER_IMG_ARTIFACTS")
fi
run_py "$GRAPH_SCRIPT_DIR/render_artifacts/render_artifacts_grid.py" "${artifact_args[@]}"

if $RUN_FINALIZE; then
  run_py "$GRAPH_SCRIPT_DIR/finalize_n500_bundle.py" --bundle-root "$BUNDLE_ROOT"
fi

echo "[bundle-graphs] complete: $BUNDLE_ROOT/graphs"
