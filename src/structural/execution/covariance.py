"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'
MODEL_CONFIG_DIR = CONFIG_DIR / 'model'


def load_model_config(model_name: str) -> Any:
    from src.common.model_config import load_model_config as resolve_model_config

    return resolve_model_config(model_name)


def available_model_names() -> list[str]:
    return sorted(p.stem for p in MODEL_CONFIG_DIR.glob('*.yaml') if p.name != 'boilerplate.yaml')


def find_second_moment_files(model_cfg: Any) -> tuple[list[Path], Path]:
    raw_dir = Path(getattr(model_cfg, 'second_moment_dir', './second_moment_stats'))
    sm_dir = raw_dir if raw_dir.is_absolute() else (Path(__file__).resolve().parents[2] / raw_dir).resolve()

    explicit = str(getattr(model_cfg, 'second_moment_path', '') or '').strip()
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = (Path(__file__).resolve().parents[2] / path).resolve()
        return ([path] if path.exists() else []), sm_dir

    model_id = str(getattr(model_cfg, 'name', '')).replace('/', '_')
    layer = int(getattr(model_cfg, 'layer', -1))
    pt_files = sorted(sm_dir.glob(f'{model_id}_{layer}_*_*.pt'))
    npz_files = sorted(sm_dir.glob(f'{model_id}_{layer}_*_*.npz'))
    return (pt_files + npz_files), sm_dir


def inspect_model_covariance(model_name: str) -> dict[str, object]:
    cfg = load_model_config(model_name)
    files, sm_dir = find_second_moment_files(cfg)
    return {
        'model': model_name,
        'resolved_name': str(getattr(cfg, 'name', model_name)),
        'layer': int(getattr(cfg, 'layer', -1)),
        'second_moment_dir': str(sm_dir),
        'files': [str(path) for path in files],
        'ok': bool(files),
    }


def inspect_models_covariance(models: Sequence[str]) -> list[dict[str, object]]:
    return [inspect_model_covariance(model_name) for model_name in models]


def summarize_covariance(models: Sequence[str]) -> dict[str, object]:
    statuses = inspect_models_covariance(models)
    return {
        'ok': all(bool(status['ok']) for status in statuses),
        'models': statuses,
    }
