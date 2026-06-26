"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

from src.structural.execution.covariance import available_model_names, find_second_moment_files

if TYPE_CHECKING:  # pragma: no cover
    from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).resolve().parents[1] / 'config'
MODEL_CONFIG_DIR = CONFIG_DIR / 'model'

FC_TEMPLATE_MAP = {
    'c_proj': 'c_fc',
    'fc_out': 'fc_in',
    'down_proj': 'up_proj',
    'output_linear': 'input_linear',
}


def get_fc_template(layer_name_template: str) -> Optional[str]:
    for proj_key, fc_key in FC_TEMPLATE_MAP.items():
        if proj_key in layer_name_template:
            return layer_name_template.replace(proj_key, fc_key)
    return None


def load_model_config(model_name: str) -> Any:
    from src.common.model_config import load_model_config as resolve_model_config

    return resolve_model_config(model_name)


def build_cfg(
    model_name: str,
    *,
    runtime: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> Any:
    from omegaconf import OmegaConf

    model_cfg = load_model_config(model_name)
    return OmegaConf.create(
        {
            'seed': int(seed),
            'runtime': dict(runtime or {}),
            'model': model_cfg,
            'generation': OmegaConf.load(CONFIG_DIR / 'generation' / 'generation.yaml'),
            'dataset_facts': OmegaConf.load(CONFIG_DIR / 'dataset_facts' / 'counterfact.yaml'),
            'dataset_sm': OmegaConf.load(CONFIG_DIR / 'dataset_sm' / 'wikitext.yaml'),
        }
    )


__all__ = [
    'MODEL_CONFIG_DIR',
    'FC_TEMPLATE_MAP',
    'available_model_names',
    'find_second_moment_files',
    'get_fc_template',
    'load_model_config',
    'build_cfg',
]
