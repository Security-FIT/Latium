#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${1:-}"
REMOTE_REPO_INPUT="${2:-}"
REMOTE_BRANCH="${3:-optim}"
CONDA_ENV="${4:-latium}"

if [[ -z "$REMOTE_HOST" ]]; then
  echo "Usage: $0 <user@host> [remote_repo_path] [remote_branch] [conda_env]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/src" ]]; then
  REPO_ROOT="$SCRIPT_DIR"
elif [[ -d "$SCRIPT_DIR/../src" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  echo "ERROR: Could not determine repository root from script location." >&2
  exit 1
fi

LOCAL_DEST="$REPO_ROOT/data/second_moment_stats"
RUN_TS="$(date +%Y-%m-%d_%H-%M-%S)"
MANIFEST="$REPO_ROOT/analysis_out/covariance_manifest_${RUN_TS}.csv"

mkdir -p "$LOCAL_DEST"
mkdir -p "$REPO_ROOT/analysis_out"

SSH_OPTS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ConnectTimeout=20
)

ssh_cmd() {
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "$@"
}

scp_cmd() {
  scp "${SSH_OPTS[@]}" "$@"
}

echo "[1/9] Resolving remote paths..."
REMOTE_HOME="$(ssh_cmd 'printf %s "$HOME"')"
REMOTE_REPO="${REMOTE_REPO_INPUT:-$REMOTE_HOME/Latium}"
echo "Remote repo path: $REMOTE_REPO"

echo "[2/9] Ensuring remote NVIDIA runtime is available..."
GPU_LINE="$(ssh_cmd "set -e; if ! command -v nvidia-smi >/dev/null 2>&1; then sudo DEBIAN_FRONTEND=noninteractive apt-get update -y >/dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-headless-535 nvidia-utils-535 >/dev/null; fi; sudo modprobe nvidia >/dev/null 2>&1 || true; nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | head -n1")"
if [[ -z "$GPU_LINE" ]]; then
  echo "ERROR: Could not read GPU info from remote host." >&2
  exit 1
fi

echo "Remote GPU: $GPU_LINE"
if [[ "$GPU_LINE" != *A100* ]]; then
  echo "WARNING: Remote GPU is not A100. Script will continue, but runtime may be slower."
fi

echo "[3/9] Bootstrapping remote conda environment..."
ssh_cmd "set -e; if [ ! -x \"\$HOME/miniconda3/bin/conda\" ]; then if command -v wget >/dev/null 2>&1; then wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; else sudo DEBIAN_FRONTEND=noninteractive apt-get update -y >/dev/null; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y curl >/dev/null; curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh; fi; bash /tmp/miniconda.sh -b -p \$HOME/miniconda3; fi; source \$HOME/miniconda3/etc/profile.d/conda.sh; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true; conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true; conda config --set auto_activate_base false >/dev/null 2>&1 || true; if ! conda env list | awk '{print \$1}' | grep -qx '${CONDA_ENV}'; then conda create -y -n ${CONDA_ENV} python=3.11 >/dev/null; fi"

echo "[4/9] Preparing remote repository..."
ssh_cmd "set -e; if [ ! -d '${REMOTE_REPO}/.git' ]; then git clone -b '${REMOTE_BRANCH}' git@github.com:Security-FIT/Latium.git '${REMOTE_REPO}' || git clone -b '${REMOTE_BRANCH}' https://github.com/Security-FIT/Latium.git '${REMOTE_REPO}'; fi; cd '${REMOTE_REPO}'; git fetch origin; git checkout '${REMOTE_BRANCH}'; if git diff --quiet && git diff --cached --quiet; then git pull --ff-only origin '${REMOTE_BRANCH}' || true; else echo 'Remote repo has local modifications; skipping git pull'; fi"

echo "[5/9] Installing remote Python dependencies..."
ssh_cmd "set -e; source \$HOME/miniconda3/etc/profile.d/conda.sh; cd '${REMOTE_REPO}'; conda run -n '${CONDA_ENV}' python -m pip install --upgrade pip wheel >/dev/null; conda run -n '${CONDA_ENV}' python -m pip install -r requirements.txt >/dev/null; conda run -n '${CONDA_ENV}' python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio >/dev/null"

echo "[6/9] Syncing required local files to remote..."
scp_cmd "$REPO_ROOT/src/rome/common.py" "$REMOTE_HOST:${REMOTE_REPO}/src/rome/common.py"
scp_cmd "$REPO_ROOT/src/config/model/mistral-7b-v0.3.yaml" "$REMOTE_HOST:${REMOTE_REPO}/src/config/model/mistral-7b-v0.3.yaml"
scp_cmd "$REPO_ROOT/src/config/model/qwen3-8b.yaml" "$REMOTE_HOST:${REMOTE_REPO}/src/config/model/qwen3-8b.yaml"
scp_cmd "$REPO_ROOT/src/config/model/llama2-7b.yaml" "$REMOTE_HOST:${REMOTE_REPO}/src/config/model/llama2-7b.yaml"

echo "[7/9] Validating CUDA in remote Python environment..."
ssh_cmd "set -e; source \$HOME/miniconda3/etc/profile.d/conda.sh; conda run -n '${CONDA_ENV}' python -c \"import torch; assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'; x=torch.randn(1024,1024,device='cuda'); y=(x@x).mean().item(); print('torch', torch.__version__); print('cuda_device', torch.cuda.get_device_name(0)); print('cuda_matmul_mean', y)\""

echo "model,remote_artifact,local_artifact" > "$MANIFEST"

validate_local_artifact() {
  local local_artifact="$1"
  if /bin/python -c "import torch" >/dev/null 2>&1; then
    /bin/python -c "import torch; p=r'$local_artifact'; m=torch.load(p, map_location='cpu'); print('validated', p, tuple(m.shape), m.dtype, bool(torch.isfinite(m).all().item()))"
  else
    echo "Local torch not available in /bin/python. Skipping tensor validation for $local_artifact"
  fi
}

run_and_pull() {
  local model_key="$1"
  local filename_prefix="$2"
  local log_name="cov_${model_key}_${RUN_TS}.log"

  local remote_artifact
  remote_artifact="$(ssh_cmd "find '${REMOTE_REPO}/second_moment_stats' '${REMOTE_REPO}/data/second_moment_stats' -maxdepth 1 -type f -name '${filename_prefix}*SM_Method.WIKIPEDIA_100000.pt' 2>/dev/null | sort | tail -n1")"

  if [[ -n "$remote_artifact" ]]; then
    echo "[SKIP] Found existing remote covariance for $model_key: $remote_artifact"
    local local_artifact_existing="$LOCAL_DEST/$(basename "$remote_artifact")"
    if [[ ! -f "$local_artifact_existing" ]]; then
      echo "[COPY] Pulling existing artifact to local: $remote_artifact"
      scp_cmd "$REMOTE_HOST:$remote_artifact" "$LOCAL_DEST/"
    fi
    echo "$model_key,$remote_artifact,$local_artifact_existing" | tee -a "$MANIFEST"
    validate_local_artifact "$local_artifact_existing"
    return
  fi

  echo "[RUN] Starting covariance for $model_key"
  ssh_cmd "set -e; cd '${REMOTE_REPO}'; mkdir -p second_moment_stats data/second_moment_stats; source \$HOME/miniconda3/etc/profile.d/conda.sh; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True conda run -n '${CONDA_ENV}' python -m src.cli command=second-moment model='${model_key}' model.device=cuda +model.cuda_mode=strict 2>&1 | tee '${REMOTE_REPO}/$log_name'"

  remote_artifact="$(ssh_cmd "find '${REMOTE_REPO}/second_moment_stats' '${REMOTE_REPO}/data/second_moment_stats' -maxdepth 1 -type f -name '${filename_prefix}*SM_Method.WIKIPEDIA_100000.pt' 2>/dev/null | sort | tail -n1")"

  if [[ -z "$remote_artifact" ]]; then
    echo "ERROR: No covariance artifact found for $model_key" >&2
    exit 1
  fi

  echo "[COPY] Pulling artifact to local: $remote_artifact"
  scp_cmd "$REMOTE_HOST:$remote_artifact" "$LOCAL_DEST/"

  local local_artifact="$LOCAL_DEST/$(basename "$remote_artifact")"
  echo "$model_key,$remote_artifact,$local_artifact" | tee -a "$MANIFEST"
  validate_local_artifact "$local_artifact"
}

run_and_pull "mistral-7b-v0.3" "mistralai_Mistral-7B-v0.3_14_"
run_and_pull "qwen3-8b" "Qwen_Qwen3-8B_10_"
run_and_pull "llama2-7b" "NousResearch_Llama-2-7b-hf_19_"

echo "[9/9] All covariance runs finished successfully."
echo "Local artifacts directory: $LOCAL_DEST"
echo "Manifest: $MANIFEST"
