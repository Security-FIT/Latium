#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: jobs/run_rome_math_ablation.sh smoke|full [additional Hydra overrides...]

Runs the opt-in, versioned M0--M3 recapture. It does not replace the
production detector and does not run the final missing 40-case protocol.
EOF
}

[[ $# -ge 1 ]] || { usage; exit 2; }
MODE="$1"
shift

case "$MODE" in
  smoke)
    MODELS='[gpt2-medium,falcon-7b]'
    CASES=2
    RUN_ID=rome-math-ablation-smoke-v1
    ;;
  full)
    MODELS='[gpt-j-6b,gpt2-medium,gpt2-xl,llama2-7b,mistral-7b-v0.1,qwen3-8b,deepseek-7b-base,falcon-7b,opt-6.7b]'
    CASES=100
    RUN_ID=rome-math-ablation-full-v1
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "ERROR: mode must be smoke or full" >&2
    usage >&2
    exit 2
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m src structural run \
  "structural.run.models=$MODELS" \
  structural.run.edit_methods='[rome]' \
  "structural.run.n_tests=$CASES" \
  structural.run.start_idx=0 \
  structural.run.output_dir=analysis_out \
  "structural.run.run_id=$RUN_ID" \
  structural.capture.profile=rome-math-ablation \
  structural.analysis.preset=none \
  structural.render.enabled=false \
  "$@"

echo "Recapture complete: analysis_out/$RUN_ID"
echo "Calibrate one B1 cutoff on development families before evaluation."
echo "Then run: python3 scripts/evaluate_rome_math_ablation.py --blind-cutoff CUTOFF"
