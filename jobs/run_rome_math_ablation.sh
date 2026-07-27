#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: jobs/run_rome_math_ablation.sh smoke-a|smoke-b|development [additional Hydra overrides...]

Runs the opt-in, versioned M0--M3 recapture. It does not replace the
production detector, expose held-out families, or run the final missing
40-case protocol.
EOF
}

[[ $# -ge 1 ]] || { usage; exit 2; }
MODE="$1"
shift

case "$MODE" in
  smoke-a)
    MODELS='[gpt2-medium,mistral-7b-v0.1]'
    CASES=2
    RUN_ID=rome-math-ablation-smoke-a-v1
    ;;
  smoke-b)
    MODELS='[gpt2-medium,mistral-7b-v0.1]'
    CASES=2
    RUN_ID=rome-math-ablation-smoke-b-v1
    ;;
  development)
    MODELS='[gpt-j-6b,gpt2-medium,gpt2-xl,llama2-7b,mistral-7b-v0.1,qwen3-8b]'
    CASES=100
    RUN_ID=rome-math-ablation-development-v1
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "ERROR: mode must be smoke-a, smoke-b, or development" >&2
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
  structural.run.fail_on_missing_second_moment=true \
  structural.capture.profile=rome-math-ablation \
  structural.analysis.preset=none \
  structural.render.enabled=false \
  "$@"

echo "Recapture complete: analysis_out/$RUN_ID"
if [[ "$MODE" == smoke-b ]]; then
  echo "Compare smoke-a and smoke-b selected layers and scores before development capture."
elif [[ "$MODE" == development ]]; then
  echo "Freeze candidate math and one global B1 cutoff before any held-out capture."
fi
