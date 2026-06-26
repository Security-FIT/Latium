"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import OmegaConf

from src.model_config import canonical_model_name, load_model_config
from src.structural.execution.covariance import find_second_moment_files
from src.structural.execution.weights import CONFIG_DIR


ROOT = Path(__file__).resolve().parents[1]


def test_model_config_resolves_huggingface_names_from_hydra_yaml() -> None:
    assert canonical_model_name("eleutherai/gpt-j-6b") == "gpt-j-6b"
    assert canonical_model_name("Qwen/Qwen3-4B") == "qwen3-4b"

    cfg = load_model_config("qwen/qwen3-4b")

    assert cfg.name == "Qwen/Qwen3-4B"
    assert cfg.models_dir == "../models"
    assert cfg.save_to_local is True


def test_structural_config_and_covariance_paths_resolve_from_project_root() -> None:
    cfg = load_model_config("gpt2-large")
    _, second_moment_dir = find_second_moment_files(cfg)

    assert CONFIG_DIR == ROOT / "src" / "config"
    assert second_moment_dir == (ROOT / "data" / "second_moment_stats").resolve()


def test_wikitext_sm_config_uses_namespaced_hub_id() -> None:
    cfg = OmegaConf.load(ROOT / "src" / "config" / "dataset_sm" / "wikitext.yaml")

    assert cfg.name == "Salesforce/wikitext"
    assert cfg.config_name == "wikitext-103-raw-v1"


def test_second_moment_model_overrides_compose() -> None:
    with hydra.initialize_config_dir(config_dir=str(ROOT / "src" / "config"), version_base=None):
        cfg = hydra.compose(
            config_name="latium",
            overrides=[
                "command=second_moment",
                "model=gpt2-large",
                "model.second_moment_target_samples=5000",
                "model.second_moment_max_length=256",
                "model.second_moment_min_text_length=20",
                "model.second_moment_clear_cache_every=3",
            ],
        )

    assert cfg.model.second_moment_target_samples == 5000
    assert cfg.model.second_moment_max_length == 256
    assert cfg.model.second_moment_min_text_length == 20
    assert cfg.model.second_moment_clear_cache_every == 3
