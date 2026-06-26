"""
Filesystem path helpers for project-relative runtime artifacts.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def ensure_dir(value: str | Path) -> Path:
    path = resolve_project_path(value)
    path.mkdir(parents=True, exist_ok=True)
    return path


def non_conflicting_path(path: str | Path) -> Path:
    target = Path(path)
    if not target.exists():
        return target
    for index in range(1, 10_000):
        candidate = target.with_name(f"{target.stem}_{index:04d}{target.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not find an available filename near {target}")


__all__ = [
    "PROJECT_ROOT",
    "ensure_dir",
    "non_conflicting_path",
    "resolve_project_path",
]
