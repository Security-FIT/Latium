
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

PROGRESS_KEY_ORDER = [
    "worker_id",
    "host",
    "tmux_session",
    "status",
    "current_model",
    "current_model_progress",
    "progress_interval",
    "completed_models",
    "remaining_models",
    "failed_models",
    "current_run_root",
    "current_log",
    "last_update",
]


def _stringify_progress_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    return str(value)


def read_worker_progress(path: str | Path) -> dict[str, str]:
    progress_path = Path(path)
    if not progress_path.exists():
        return {}

    data: dict[str, str] = {}
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if "|" not in line:
            continue
        key, value = line.split("|", 1)
        data[key.strip()] = value.strip()
    return data


def write_worker_progress(
    path: str | Path,
    updates: Mapping[str, Any],
    *,
    preserve_existing: bool = True,
) -> Path:
    progress_path = Path(path)
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    current = read_worker_progress(progress_path) if preserve_existing else {}
    for key, value in updates.items():
        current[str(key)] = _stringify_progress_value(value)
    current["last_update"] = datetime.now().isoformat()

    ordered_keys = list(PROGRESS_KEY_ORDER)
    for key in current:
        if key not in ordered_keys:
            ordered_keys.append(key)

    lines = [f"{key} | {current.get(key, '')}" for key in ordered_keys]
    progress_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return progress_path


def effective_progress_interval(total_cases: int, requested_interval: int) -> int:
    total = max(0, int(total_cases))
    interval = max(1, int(requested_interval))
    if total and total < interval:
        return 1
    return interval
