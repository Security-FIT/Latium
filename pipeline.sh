#!/usr/bin/env bash
# ==========================================================================
# pipeline.sh — Simple end-to-end benchmark pipeline (local or remote)
#
# Usage:
#   bash pipeline.sh [options]
#
# Examples:
#   # ROME-only, N=20, no covariance, no setup (default)
#   bash pipeline.sh
#
#   # Structural + ROME, N=30, with env setup on cluster
#   bash pipeline.sh --structural --n 30 --setup-env
#
#   # Compute covariance first, then ROME-only N=10
#   bash pipeline.sh --compute-cov --n 10
#
#   # Run remotely via SSH
#   bash pipeline.sh --remote ubuntu@132.145.129.234
# ==========================================================================
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
COMPUTE_COV=false
N=20
STRUCTURAL=false
SETUP_ENV=false
REMOTE_HOST=""
REMOTE_DIR="~/Latium"
CONDA_ENV="latium"
OUTPUT_DIR="./pipeline_out"
LOG_DIR="./pipeline_out/logs"

MODELS=(
    gpt2-xl
    granite4-micro
    gpt2-medium
    gpt2-large
    qwen3-4b
    falcon-7b
    deepseek-7b-base
    mistral-7b-v0.1
    mistral-7b-v0.3
    gpt-j-6b
    opt-6.7b
    llama2-7b
    qwen3-8b
)

# ── Parse arguments ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --compute-cov)   COMPUTE_COV=true; shift ;;
        --n)             N="$2"; shift 2 ;;
        --structural)    STRUCTURAL=true; shift ;;
        --setup-env)     SETUP_ENV=true; shift ;;
        --remote)        REMOTE_HOST="$2"; shift 2 ;;
        --remote-dir)    REMOTE_DIR="$2"; shift 2 ;;
        --conda-env)     CONDA_ENV="$2"; shift 2 ;;
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --models)        # read models until next flag or end
                         shift; MODELS=()
                         while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                             MODELS+=("$1"); shift
                         done ;;
        -h|--help)
            echo "Usage: bash pipeline.sh [options]"
            echo ""
            echo "Options:"
            echo "  --compute-cov        Compute covariance on cluster (default: skip, use existing)"
            echo "  --n <int>            Number of test edits per model (default: 20)"
            echo "  --structural         Run structural benchmark (default: ROME-only)"
            echo "  --setup-env          Set up conda env on cluster from scratch"
            echo "  --remote <host>      SSH host (e.g. ubuntu@132.145.129.234)"
            echo "  --remote-dir <path>  Remote repo path (default: ~/Latium)"
            echo "  --conda-env <name>   Conda env name (default: latium)"
            echo "  --output-dir <path>  Output directory (default: ./pipeline_out)"
            echo "  --models <m1 m2 ..>  Override model list"
            echo "  -h, --help           Show this help"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# ── Banner ────────────────────────────────────────────────────────────────
echo "========================================"
echo " Latium Pipeline"
echo " N=$N | structural=$STRUCTURAL | compute_cov=$COMPUTE_COV"
echo " setup_env=$SETUP_ENV | models=${#MODELS[@]}"
if [[ -n "$REMOTE_HOST" ]]; then
    echo " remote=$REMOTE_HOST:$REMOTE_DIR"
else
    echo " remote=(local)"
fi
echo " output=$OUTPUT_DIR"
echo " started: $(date)"
echo "========================================"

# ── Helper: run a command locally or remotely ─────────────────────────────
run_cmd() {
    if [[ -n "$REMOTE_HOST" ]]; then
        ssh -o StrictHostKeyChecking=accept-new "$REMOTE_HOST" \
            "source \$HOME/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true; \
             conda activate $CONDA_ENV 2>/dev/null || true; \
             cd $REMOTE_DIR; \
             export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; \
             $*"
    else
        (
            export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            eval "$*"
        )
    fi
}

# ── Step 0: Environment setup (optional) ──────────────────────────────────
if $SETUP_ENV; then
    echo ""
    echo "[SETUP] Setting up environment on ${REMOTE_HOST:-localhost}..."

    if [[ -n "$REMOTE_HOST" ]]; then
        # Sync repo to remote
        echo "[SETUP] Syncing repo..."
        rsync -avz --progress \
            --exclude='venv/' --exclude='.git/' --exclude='__pycache__/' \
            --exclude='models/' --exclude='runs/' --exclude='outputs/' \
            --exclude='.pytest_cache/' --exclude='datasets/' \
            --exclude='analysis_output/' --exclude='pipeline_out/' \
            ./ "$REMOTE_HOST:$REMOTE_DIR/"

        # Sync covariance files
        echo "[SETUP] Syncing covariance matrices..."
        rsync -avz --progress \
            second_moment_stats/*.pt \
            "$REMOTE_HOST:$REMOTE_DIR/second_moment_stats/" 2>/dev/null || true

        # Install conda + env + deps
        echo "[SETUP] Installing conda env and deps..."
        ssh -o StrictHostKeyChecking=accept-new "$REMOTE_HOST" "
            set -e
            # Install miniconda if missing
            if [ ! -x \"\$HOME/miniconda3/bin/conda\" ]; then
                wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
                bash /tmp/miniconda.sh -b -p \$HOME/miniconda3
            fi
            source \$HOME/miniconda3/etc/profile.d/conda.sh
            conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true

            # Create env if missing
            if ! conda env list | awk '{print \$1}' | grep -qx '$CONDA_ENV'; then
                conda create -y -n $CONDA_ENV python=3.11
            fi

            # Install deps
            conda activate $CONDA_ENV
            cd $REMOTE_DIR
            pip install --upgrade pip wheel -q
            pip install -r requirements.txt -q
            pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio -q
            echo '[SETUP] Done.'
        "
    else
        echo "[SETUP] Local mode — skipping remote setup. Make sure your env is active."
    fi

    echo "[SETUP] Complete."
fi

# ── Step 1: Covariance (optional) ─────────────────────────────────────────
if $COMPUTE_COV; then
    echo ""
    echo "[COV] Computing covariance matrices..."
    for model in "${MODELS[@]}"; do
        echo "[COV] $model"
        run_cmd "python -m src.cli command=second-moment model=$model model.device=cuda +model.cuda_mode=strict" \
            2>&1 | tee "$LOG_DIR/cov_${model}.log" || {
                echo "[COV] WARNING: covariance failed for $model"
            }
    done
    echo "[COV] Done."
else
    echo ""
    echo "[COV] Skipping covariance (using existing files from second_moment_stats/)"
fi

# ── Step 2: Run benchmarks ────────────────────────────────────────────────
TOTAL=${#MODELS[@]}
PASSED=0
FAILED=0
FAILED_MODELS=""

echo ""
if $STRUCTURAL; then
    echo "[BENCH] Running STRUCTURAL benchmark (N=$N) for $TOTAL models..."
else
    echo "[BENCH] Running ROME-only benchmark (N=$N) for $TOTAL models..."
fi

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    IDX=$((i + 1))
    echo ""
    echo "[$IDX/$TOTAL] ======== $MODEL ========"
    echo "Started at $(date)"

    LOG_FILE="$LOG_DIR/${MODEL}.log"

    if $STRUCTURAL; then
        CMD="python structural_benchmark.py \
            --models $MODEL \
            --n-tests $N \
            --start-idx 0 \
            --n-prompts 20 \
            --spectral-top-k 50 \
            --spectral-neighbor-layers 1 \
            --spectral-rolling-window 5 \
            --local-windows 3,5,7 \
            --analysis-profile full \
            --output-dir $OUTPUT_DIR"
    else
        CMD="python rome_benchmark.py \
            --models $MODEL \
            --n-tests $N \
            --start-idx 0 \
            --output-dir $OUTPUT_DIR"
    fi

    if run_cmd "$CMD" 2>&1 | tee "$LOG_FILE"; then
        echo "[$IDX/$TOTAL] $MODEL — DONE ($(date))"
        PASSED=$((PASSED + 1))
    else
        echo "[$IDX/$TOTAL] $MODEL — FAILED ($(date))"
        FAILED=$((FAILED + 1))
        FAILED_MODELS="$FAILED_MODELS $MODEL"
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo " Pipeline Complete — $(date)"
echo " Mode: $(if $STRUCTURAL; then echo 'structural'; else echo 'rome-only'; fi)"
echo " N=$N"
echo " Passed: $PASSED / $TOTAL"
echo " Failed: $FAILED / $TOTAL"
if [[ -n "$FAILED_MODELS" ]]; then
    echo " Failed:$FAILED_MODELS"
fi
echo " Output: $OUTPUT_DIR"
echo "========================================"
