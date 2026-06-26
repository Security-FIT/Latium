"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.results.naming import safe_slug


def _segment(value: str, name: str) -> str:
    identifier = str(value).strip()
    candidate = Path(identifier)
    if not identifier or candidate.is_absolute() or len(candidate.parts) != 1 or identifier in {".", ".."}:
        raise ValueError(f"{name} must be one path-safe segment, got {value!r}")
    return safe_slug(identifier)


@dataclass(frozen=True)
class RunLayout:
    root: Path

    @classmethod
    def from_output(cls, output_dir: str | Path, run_id: str) -> "RunLayout":
        return cls(Path(output_dir) / _segment(run_id, "run_id"))

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def plans(self) -> Path:
        return self.root / "plans"

    @property
    def graphs(self) -> Path:
        return self.root / "graphs"

    def plan_root(self, model: str, plan_id: str) -> Path:
        return self.plans / _segment(model, "model") / _segment(plan_id, "plan_id")

    def baseline_root(self, model: str, plan_id: str) -> Path:
        return self.plan_root(model, plan_id) / "baseline"

    def method_root(self, model: str, plan_id: str, edit_method: str) -> Path:
        return self.plan_root(model, plan_id) / "methods" / _segment(edit_method, "edit_method")

    def execution_path(
        self,
        model: str,
        plan_id: str,
        *,
        edit_method: str | None,
    ) -> Path:
        root = (
            self.baseline_root(model, plan_id) if edit_method is None else self.method_root(model, plan_id, edit_method)
        )
        return root / "execution.json"

    def capture_path(
        self,
        model: str,
        plan_id: str,
        capture_id: str,
        *,
        edit_method: str | None,
    ) -> Path:
        root = (
            self.baseline_root(model, plan_id) if edit_method is None else self.method_root(model, plan_id, edit_method)
        )
        return root / "captures" / f"{_segment(capture_id, 'capture_id')}.json"

    def analysis_path(
        self,
        model: str,
        plan_id: str,
        edit_method: str,
        category: str,
        analysis_id: str,
        analysis_config_hash: str,
    ) -> Path:
        return (
            self.method_root(model, plan_id, edit_method)
            / "analysis"
            / _segment(category, "category")
            / _segment(analysis_id, "analysis_id")
            / f"{_segment(analysis_config_hash, 'analysis_config_hash')}.json"
        )

    def render_path(self, renderer: str, name: str) -> Path:
        return self.graphs / _segment(renderer, "renderer") / name

    def relative(self, path: str | Path) -> str:
        return str(Path(path).relative_to(self.root))

    def ensure(self) -> "RunLayout":
        self.root.mkdir(parents=True, exist_ok=True)
        return self
