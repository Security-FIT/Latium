#!/usr/bin/env bash
set -euo pipefail

# Re-capture matched non-ROME edits with the opt-in simple-Gram candidates.
# This script never generates ROME edits and refuses to start on a busy GPU.

trial_root="${LATIUM_TRIAL_ROOT:-/home/ubuntu/Latium-simple-gram-trial}"
env_file="${LATIUM_ENV_FILE:-/home/ubuntu/Latium-detector/jobs/local.env}"
magnitude_dir="${SIMPLE_GRAM_MAGNITUDE_DIR:-${trial_root}/analysis_out/magnitude-sources}"
output_dir="${SIMPLE_GRAM_NEGATIVE_OUTPUT_DIR:-${trial_root}/analysis_out/rome-simple-gram-hard-negatives-v1}"
count="${SIMPLE_GRAM_NEGATIVE_COUNT:-10}"
models="${SIMPLE_GRAM_NEGATIVE_MODELS:-gpt2-xl mistral-7b-v0.1 falcon-7b olmo-3-1025-7b granite4-micro}"

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

check_idle() {
  local gpu_processes relevant_processes
  gpu_processes="$(
    nvidia-smi \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>/dev/null || true
  )"
  relevant_processes="$(
    ps -eo pid=,args= | awk -v self="$$" '
      $1 != self &&
      $0 ~ /[p]ython/ &&
      $0 ~ /(Latium|latium|-m src|rome|qwen|gemma|structural|generate_)/ {
        print
      }
    '
  )"
  if [[ -n "${gpu_processes//[[:space:]]/}" ]] ||
    [[ -n "${relevant_processes//[[:space:]]/}" ]]; then
    echo "Cluster is occupied; refusing to start simple-Gram hard negatives:" >&2
    [[ -z "${gpu_processes//[[:space:]]/}" ]] ||
      echo "${gpu_processes}" >&2
    [[ -z "${relevant_processes//[[:space:]]/}" ]] ||
      echo "${relevant_processes}" >&2
    return 75
  fi
}

check_idle
cd "${trial_root}"
mkdir -p "${output_dir}/logs"
ledger="${output_dir}/ledger.tsv"
if [[ ! -f "${ledger}" ]]; then
  printf 'model\tstatus\texit_code\tstarted_at\tfinished_at\n' >"${ledger}"
fi

for model in ${models}; do
  if awk -F '\t' -v model="${model}" \
    '$1 == model && $2 == "complete" { found=1 } END { exit !found }' \
    "${ledger}"; then
    continue
  fi
  check_idle
  source_path="${magnitude_dir}/single-checkpoint-hard-negatives-dev-${model}-v1.json"
  if [[ ! -f "${source_path}" ]]; then
    echo "Missing generation-only magnitude source: ${source_path}" >&2
    exit 2
  fi

  output_path="${output_dir}/${model}.json"
  log="${output_dir}/logs/${model}.log"
  started_at="$(date --iso-8601=seconds)"
  set +e
  python scripts/generate_simple_gram_hard_negatives.py \
    --model "${model}" \
    --magnitude-source "${source_path}" \
    --count "${count}" \
    --output "${output_path}" 2>&1 | tee -a "${log}"
  code=${PIPESTATUS[0]}
  set -e
  finished_at="$(date --iso-8601=seconds)"
  status=failed
  if [[ "${code}" -eq 0 ]]; then
    status=complete
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "${model}" "${status}" "${code}" "${started_at}" "${finished_at}" \
    >>"${ledger}"
  if [[ "${code}" -ne 0 ]]; then
    exit "${code}"
  fi
done
