#!/usr/bin/env bash
set -euo pipefail

# Remote runner for prefixtest/experiment.py.
#
# Defaults:
#   REMOTE_HOST=ubuntu@154.54.100.221
#   REMOTE_DIR=/home/ubuntu/reimagined
#   MODEL=Qwen/Qwen3-8B
#   CASE_IDX=0
#
# Examples:
#   ./prefixtest/run_remote.sh
#   ./prefixtest/run_remote.sh gpt2-large 3
#   ./prefixtest/run_remote.sh --status
#   ./prefixtest/run_remote.sh --fetch
#   REMOTE_HOST=user@host REMOTE_DIR=/workspace/reimagined ./prefixtest/run_remote.sh

REMOTE_HOST="${REMOTE_HOST:-ubuntu@154.54.100.221}"
REMOTE_DIR="${REMOTE_DIR:-/home/ubuntu/reimagined}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TMUX_SESSION="${TMUX_SESSION:-prefixtest}"
MODEL_DEFAULT="Qwen/Qwen3-8B"
CASE_IDX_DEFAULT="0"
RUN_NAMES="${RUN_NAMES:-}"
RUN_NAMES_B64="$(printf '%s' "$RUN_NAMES" | base64 -w0 2>/dev/null || printf '%s' "$RUN_NAMES" | base64 | tr -d '\n')"

SM_FILE="second_moment_stats/Qwen_Qwen3-8B_10_SM_Method.WIKIPEDIA_100000.pt"
SSH_OPTS="-o ConnectTimeout=30 -o ServerAliveInterval=15 -o ServerAliveCountMax=6 -o StrictHostKeyChecking=accept-new"

remote_home_dir() {
    case "$REMOTE_HOST" in
        root@*) printf "/root" ;;
        *) printf "/home/%s" "${REMOTE_HOST##*@}" | sed "s#^/home/\([^@]*\)\$#/home/${REMOTE_HOST%@*}#" ;;
    esac
}

remote_workdir() {
    printf "%s" "$REMOTE_DIR"
}

remote_shell_path() {
    local dir
    dir="$(remote_workdir)"
    printf "%q" "$dir"
}

ensure_local_output_dir() {
    mkdir -p "$SCRIPT_DIR/artifacts"
}

fetch_results() {
    ensure_local_output_dir
    echo "==> Fetching results from ${REMOTE_HOST}:${REMOTE_DIR}/analysis_out ..."
    local max_retries=5
    local attempt=1
    while [ "$attempt" -le "$max_retries" ]; do
        echo "  Attempt ${attempt}/${max_retries} ..."
        if rsync -az --partial --append-verify --timeout=180 \
            -e "ssh ${SSH_OPTS}" \
            --info=progress2 \
            "${REMOTE_HOST}:${REMOTE_DIR}/analysis_out/" \
            "$SCRIPT_DIR/artifacts/"
        then
            echo "==> Results saved to $SCRIPT_DIR/artifacts/"
            return 0
        fi
        if [ "$attempt" -lt "$max_retries" ]; then
            echo "  Transfer failed, retrying in 15s ..."
            sleep 15
        fi
        attempt=$((attempt + 1))
    done
    echo "Fetch failed after ${max_retries} attempts." >&2
    return 1
}

show_status() {
    local status
    status="$(ssh ${SSH_OPTS} "$REMOTE_HOST" "tmux capture-pane -t ${TMUX_SESSION} -p 2>/dev/null" || true)"
    if [ -n "$status" ]; then
        printf '%s\n' "$status"
    else
        echo "No active session."
    fi
}

if [ "${1:-}" = "--fetch" ]; then
    fetch_results
    exit 0
fi

if [ "${1:-}" = "--status" ]; then
    show_status
    exit 0
fi

MODEL="${1:-$MODEL_DEFAULT}"
CASE_IDX="${2:-$CASE_IDX_DEFAULT}"

echo "==> Target remote: ${REMOTE_HOST}"
echo "==> Remote dir:    ${REMOTE_DIR}"
echo "==> Model:         ${MODEL}"
echo "==> Case idx:      ${CASE_IDX}"
if [ -n "$RUN_NAMES" ]; then
    echo "==> Run names:     ${RUN_NAMES}"
fi

echo "==> Preparing remote directories ..."
ssh ${SSH_OPTS} "$REMOTE_HOST" "mkdir -p $(printf "%q" "$REMOTE_DIR")/analysis_out $(printf "%q" "$REMOTE_DIR")/second_moment_stats"

echo "==> Uploading source files ..."
rsync -az --info=progress2 \
    -e "ssh ${SSH_OPTS}" \
    --include='src/***' \
    --include='structural_benchmark.py' \
    --include='prefixtest/experiment.py' \
    --include='requirements.txt' \
    --exclude='*' \
    "$REPO_DIR/" "${REMOTE_HOST}:${REMOTE_DIR}/"

echo "==> Syncing second-moment file ..."
rsync -az --partial --append-verify --timeout=180 --info=progress2 \
    -e "ssh ${SSH_OPTS}" \
    "$REPO_DIR/$SM_FILE" \
    "${REMOTE_HOST}:${REMOTE_DIR}/${SM_FILE}"

echo "==> Installing dependencies on remote ..."
ssh ${SSH_OPTS} "$REMOTE_HOST" REMOTE_DIR="$REMOTE_DIR" bash <<'SETUP_EOF'
set -euo pipefail
cd "$REMOTE_DIR"

if [ -d .venv ] && [ ! -f .venv/bin/activate ]; then
    rm -rf .venv
fi

if [ ! -d .venv ]; then
    if ! python3 -m venv .venv; then
        if command -v sudo >/dev/null 2>&1; then
            sudo apt-get update -y
            sudo apt-get install -y python3-venv
            rm -rf .venv
            python3 -m venv .venv
        else
            echo "python3 -m venv failed and sudo is unavailable" >&2
            exit 1
        fi
    fi
    echo "Created venv"
fi

source .venv/bin/activate
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -q

echo "==> Python: $(python --version)"
echo "==> torch:  $(python -c 'import torch; print(torch.__version__, "cuda=" + str(torch.cuda.is_available()))')"
SETUP_EOF

echo "==> Launching experiment in tmux session '${TMUX_SESSION}' ..."
ssh ${SSH_OPTS} "$REMOTE_HOST" REMOTE_DIR="$REMOTE_DIR" MODEL="$MODEL" CASE_IDX="$CASE_IDX" TMUX_SESSION="$TMUX_SESSION" RUN_NAMES_B64="$RUN_NAMES_B64" bash <<'REMOTE_EOF'
set -euo pipefail
cd "$REMOTE_DIR"
RUN_NAMES="$(printf '%s' "$RUN_NAMES_B64" | base64 -d)"
RUN_SCRIPT="$REMOTE_DIR/run_prefixtest_tmux.sh"

tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

cat > "$RUN_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$REMOTE_DIR"
source .venv/bin/activate
echo "Starting prefixtest at \\$(date)"
echo "Model: $MODEL  Case: $CASE_IDX"
cmd=(python prefixtest/experiment.py --model "$MODEL" --case-idx $CASE_IDX --output-dir ./analysis_out --spectral-top-k 50 --spectral-neighbor-layers 1)
if [ -n "$RUN_NAMES" ]; then
    cmd+=(--run-names)
    for name in $RUN_NAMES; do
        cmd+=("\$name")
    done
fi
printf 'Command: '
printf '%q ' "\${cmd[@]}"
echo
"\${cmd[@]}"
status=\$?
echo
echo "===================================="
if [ \$status -eq 0 ]; then
    echo "  DONE at \\$(date)"
else
    echo "  FAILED with exit \$status at \\$(date)"
fi
echo "===================================="
ls -lh ./analysis_out/gpt2_prefix_template_variability_* 2>/dev/null || true
EOF

chmod +x "$RUN_SCRIPT"
tmux new-session -d -s "$TMUX_SESSION" "$RUN_SCRIPT"

echo "Experiment launched."
REMOTE_EOF

echo
echo "================================================================"
echo "  Experiment is running in tmux on the remote."
echo
echo "  Monitor live: ssh ${REMOTE_HOST} -t tmux attach -t ${TMUX_SESSION}"
echo "  Status:       ./prefixtest/run_remote.sh --status"
echo "  Fetch:        ./prefixtest/run_remote.sh --fetch"
echo "================================================================"
