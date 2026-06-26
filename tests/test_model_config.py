"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from src.model_config import canonical_model_name, load_model_config


def test_model_config_resolves_huggingface_names_from_hydra_yaml() -> None:
    assert canonical_model_name("eleutherai/gpt-j-6b") == "gpt-j-6b"
    assert canonical_model_name("Qwen/Qwen3-4B") == "qwen3-4b"

    cfg = load_model_config("qwen/qwen3-4b")

    assert cfg.name == "Qwen/Qwen3-4B"
    assert cfg.models_dir == "../models"
    assert cfg.save_to_local is True
