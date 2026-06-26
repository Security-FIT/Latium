"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def ensure_parent_dir(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def load_json(path: str | Path) -> dict:
    with Path(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    target = ensure_parent_dir(path)
    target.write_text(json.dumps(to_serializable(payload), indent=indent), encoding='utf-8')
    return target


def load_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    target = ensure_parent_dir(path)
    with target.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(to_serializable(row)))
            handle.write('\n')
    return target


def to_serializable(obj: Any):
    obj_module = type(obj).__module__
    if (obj_module == 'torch' or obj_module.startswith('torch.')) and hasattr(obj, 'tolist'):
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj
