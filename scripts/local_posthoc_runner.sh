#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python_supports_posthoc() {
	local py_bin="$1"
	[[ -x "$py_bin" ]] || return 1
	"$py_bin" - <<'PY' >/dev/null 2>&1
import matplotlib
import numpy
import pyparsing
import scipy
PY
}

resolve_posthoc_python() {
	local candidates=()
	if [[ -n "${LATIUM_POSTHOC_PYTHON:-}" ]]; then
		candidates+=("$LATIUM_POSTHOC_PYTHON")
	fi
	if command -v python >/dev/null 2>&1; then
		candidates+=("$(command -v python)")
	fi
	if command -v python3 >/dev/null 2>&1; then
		candidates+=("$(command -v python3)")
	fi
	candidates+=(
		"$REPO_ROOT/.venv/bin/python"
		"$HOME/miniforge/envs/latium/bin/python"
		"$HOME/miniconda3/envs/latium/bin/python"
		"$HOME/miniforge/envs/llms/bin/python"
		"$HOME/miniconda3/envs/llms/bin/python"
		"$HOME/miniforge/bin/python"
		"$HOME/miniconda3/bin/python"
	)

	local candidate
	for candidate in "${candidates[@]}"; do
		if python_supports_posthoc "$candidate"; then
			printf '%s\n' "$candidate"
			return 0
		fi
	done

	echo "Unable to find a Python interpreter with matplotlib/numpy/scipy/pyparsing for pipeline_posthoc.py" >&2
	return 1
}

POSTHOC_PYTHON="$(resolve_posthoc_python)"
if [[ "${1:-}" == "--once" ]]; then
	shift
	exec "$POSTHOC_PYTHON" pipeline_posthoc.py "$@"
fi

exec "$POSTHOC_PYTHON" pipeline_posthoc.py --watch "$@"
