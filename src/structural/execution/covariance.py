"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.paths import resolve_project_path

CONFIG_DIR = Path(__file__).resolve().parents[2] / 'config'
MODEL_CONFIG_DIR = CONFIG_DIR / 'model'


def available_model_names() -> list[str]:
    return sorted(p.stem for p in MODEL_CONFIG_DIR.glob('*.yaml') if p.name != 'boilerplate.yaml')


def find_second_moment_files(model_cfg: Any) -> tuple[list[Path], Path]:
    raw_dir = Path(getattr(model_cfg, 'second_moment_dir', './second_moment_stats'))
    sm_dir = resolve_project_path(raw_dir)

    explicit = str(getattr(model_cfg, 'second_moment_path', '') or '').strip()
    if explicit:
        path = resolve_project_path(explicit)
        return ([path] if path.exists() else []), sm_dir

    model_id = str(getattr(model_cfg, 'name', '')).replace('/', '_')
    layer = int(getattr(model_cfg, 'layer', -1))
    pt_files = sorted(sm_dir.glob(f'{model_id}_{layer}_*_*.pt'))
    npz_files = sorted(sm_dir.glob(f'{model_id}_{layer}_*_*.npz'))
    return (pt_files + npz_files), sm_dir
