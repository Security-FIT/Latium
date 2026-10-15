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
    canonical_model_name,
    load_model_config,
)

__all__ = [
    "MODEL_BASE_CONFIG",
    "MODEL_CONFIG_DIR",
    "canonical_model_name",
    "load_model_config",
]
