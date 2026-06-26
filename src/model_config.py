"""
Compatibility facade for model configuration helpers.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from src.common.model_config import (
    MODEL_BASE_CONFIG,
    MODEL_CONFIG_DIR,
    MODEL_FLEET_CONFIG,
    canonical_model_name,
    fleet_model_key,
    load_model_config,
    sanitize_model_id,
    second_moment_basename,
)

__all__ = [
    "MODEL_BASE_CONFIG",
    "MODEL_CONFIG_DIR",
    "MODEL_FLEET_CONFIG",
    "canonical_model_name",
    "fleet_model_key",
    "load_model_config",
    "sanitize_model_id",
    "second_moment_basename",
]
