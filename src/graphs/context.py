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


@dataclass(frozen=True)
class RenderContext:
    run_root: Path
    output_dir: Path
    manifest: Mapping[str, Any]
    executions: tuple[dict[str, Any], ...] = ()
    captures: dict[str, tuple[dict[str, Any], ...]] | None = None
    analyses: dict[str, tuple[dict[str, Any], ...]] | None = None
    options: Mapping[str, Any] | None = None
    style_preset: str = "default"
    warnings: tuple[str, ...] = ()

    @property
    def flat_captures(self) -> list[dict[str, Any]]:
        captures = self.captures or {}
        return [payload for values in captures.values() for payload in values]

    @property
    def flat_analyses(self) -> list[dict[str, Any]]:
        analyses = self.analyses or {}
        return [payload for values in analyses.values() for payload in values]

    def as_mapping(self) -> dict[str, Any]:
        return {
            "run_root": self.run_root,
            "output_dir": self.output_dir,
            "manifest": dict(self.manifest),
            "executions": list(self.executions),
            "captures": self.flat_captures,
            "analyses": self.flat_analyses,
            "captures_by_producer": {key: list(value) for key, value in (self.captures or {}).items()},
            "analyses_by_producer": {key: list(value) for key, value in (self.analyses or {}).items()},
            "options": dict(self.options or {}),
            "style_preset": self.style_preset,
            "warnings": list(self.warnings),
        }


__all__ = ["RenderContext", "RendererUnavailableError"]
