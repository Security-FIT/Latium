#!/usr/bin/env bash
set -euo pipefail

# Opt-in remote runner for the no-reference G0-G3 Gram experiment.
# It refuses to start while any compute process is visible on the GPU.

trial_root="${LATIUM_TRIAL_ROOT:-/home/ubuntu/Latium-simple-gram-trial}"
env_file="${LATIUM_ENV_FILE:-/home/ubuntu/Latium-detector/jobs/local.env}"
output_root="${SIMPLE_GRAM_OUTPUT_ROOT:-analysis_out/rome-simple-gram-n20-v1}"
n_tests="${SIMPLE_GRAM_N_TESTS:-20}"
start_idx="${SIMPLE_GRAM_START_IDX:-0}"
models="${SIMPLE_GRAM_MODELS:-gpt2-xl gpt-j-6b mistral-7b-v0.3 falcon-7b olmo-3-1025-7b granite4-micro}"

if [[ ! -d "${trial_root}" ]]; then
  echo "Missing trial checkout: ${trial_root}" >&2
  exit 2
fi
if [[ -f "${env_file}" ]]; then
  # shellcheck disable=SC1090
  source "${env_file}"
fi
if [[ -n "${LATIUM_ENV:-}" ]]; then
  export PATH="${LATIUM_ENV}/bin:${PATH}"
fi
if [[ -n "${LATIUM_CACHE_ROOT:-}" ]]; then
  export HF_HOME="${LATIUM_CACHE_ROOT}"
  export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
  export HF_DATASETS_CACHE="${HF_HOME}/datasets"
fi
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

gpu_processes="$(
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>/dev/null || true
)"
if [[ -n "${gpu_processes//[[:space:]]/}" ]]; then
  echo "GPU is occupied; refusing to start the simple-Gram experiment:" >&2
  echo "${gpu_processes}" >&2
  exit 75
fi

cd "${trial_root}"
mkdir -p "${output_root}/logs"
ledger="${output_root}/ledger.tsv"
if [[ ! -f "${ledger}" ]]; then
  printf 'model\trun_id\tstatus\texit_code\tstarted_at\tfinished_at\n' >"${ledger}"
fi

for model in ${models}; do
  run_id="rome-simple-gram-n${n_tests}-${model}-v1"
  log="${output_root}/logs/${model}.log"
  if awk -F '\t' -v model="${model}" \
    '$1 == model && $3 == "complete" { found=1 } END { exit !found }' \
    "${ledger}"; then
    continue
  fi

  gpu_processes="$(
    nvidia-smi \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>/dev/null || true
  )"
  if [[ -n "${gpu_processes//[[:space:]]/}" ]]; then
    echo "GPU became occupied before ${model}; stopping without interference." \
      | tee -a "${log}"
    echo "${gpu_processes}" | tee -a "${log}"
    exit 75
  fi

  started_at="$(date --iso-8601=seconds)"
  printf '\n===== %s %s =====\n' "${model}" "${started_at}" | tee -a "${log}"
  set +e
  python -m src structural run \
    "structural.run.models=[${model}]" \
    'structural.run.edit_methods=[rome]' \
    "structural.run.n_tests=${n_tests}" \
    "structural.run.start_idx=${start_idx}" \
    structural.run.output_dir=analysis_out \
    "structural.run.run_id=${run_id}" \
    structural.run.fail_on_missing_second_moment=true \
    structural.capture.profile=none \
    'structural.capture.enable=[simple-gram-experiment]' \
    structural.analysis.preset=none \
    structural.render.enabled=false 2>&1 | tee -a "${log}"
  code=${PIPESTATUS[0]}
  set -e
  finished_at="$(date --iso-8601=seconds)"
  status=failed
  if [[ "${code}" -eq 0 ]]; then
    status=complete
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${model}" "${run_id}" "${status}" "${code}" \
    "${started_at}" "${finished_at}" >>"${ledger}"
  if [[ "${code}" -ne 0 ]]; then
    exit "${code}"
  fi
done
