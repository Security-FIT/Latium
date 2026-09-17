#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${LATIUM_TMUX_SESSION:-latium-paper-n10}"
RUN_ROOT="${LATIUM_FLEET_ROOT:-$ROOT/analysis_out/paper-fleet/$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON="${LATIUM_PYTHON:-$ROOT/.venv/bin/python}"
N_TESTS="${LATIUM_N_TESTS:-50}"

if [[ ! -x "$PYTHON" ]]; then
  printf 'Missing executable environment: %s\n' "$PYTHON" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  printf 'tmux session already exists: %s\n' "$SESSION"
  printf 'Attach with: tmux attach -t %s\n' "$SESSION"
  exit 0
fi

mkdir -p "$RUN_ROOT"
tmux new-session -d -s "$SESSION" -n fleet bash -lc \
  "cd $(printf '%q' "$ROOT") && $(printf '%q' "$PYTHON") $(printf '%q' "$ROOT/jobs/paper_fleet.py") --run-root $(printf '%q' "$RUN_ROOT") --n-tests $(printf '%q' "$N_TESTS"); rc=\$?; printf '\\nFleet launcher exited with status %s.\n' \"\$rc\"; exec bash"

printf 'Started tmux session: %s\n' "$SESSION"
printf 'Run root: %s\n' "$RUN_ROOT"
printf 'Attach with: tmux attach -t %s\n' "$SESSION"
printf 'Fleet log: %s/fleet.log\n' "$RUN_ROOT"
