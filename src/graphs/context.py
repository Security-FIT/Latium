"""
Typed context helpers for graph renderers.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class RendererUnavailableError(RuntimeError):
    """Raised when declared renderer inputs are present but unusable."""


class RenderExecutionError(RuntimeError):
    """Raised after one or more renderer failures have been recorded."""


@dataclass(frozen=True)
class RenderContext:
    run_root: Path
    output_dir: Path
    manifest: Mapping[str, Any]
    executions: tuple[dict[str, Any], ...] = ()
    captures: dict[str, tuple[dict[str, Any], ...]] | None = None
    analyses: dict[str, tuple[dict[str, Any], ...]] | None = None
    options: Mapping[str, Any] | None = None
    warnings: tuple[str, ...] = ()

    @property
    def flat_captures(self) -> list[dict[str, Any]]:
        captures = self.captures or {}
        return [payload for values in captures.values() for payload in values]

    @property
    def flat_analyses(self) -> list[dict[str, Any]]:
        analyses = self.analyses or {}
        return [payload for values in analyses.values() for payload in values]


__all__ = ["RenderContext", "RenderExecutionError", "RendererUnavailableError"]
