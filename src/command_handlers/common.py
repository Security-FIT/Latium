"""
Shared helpers for Hydra command handlers.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.common.config import plain as _plain
from src.common.io import to_serializable, write_json


def path_or_none(value: Any) -> Path | None:
    raw = _plain(value)
    if raw in (None, ""):
        return None
    return Path(str(raw))


def write_or_print(payload: dict[str, object], path: Path | None) -> None:
    if path is not None:
        write_json(path, payload)
    else:
        print(json.dumps(to_serializable(payload), indent=2))


__all__ = ["path_or_none", "write_or_print"]
