"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from pathlib import Path

import hydra
import pytest
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


@pytest.mark.parametrize(
    ("model_key", "layer", "covariance_name"),
    [
        ("granite4-micro", 10, "ibm-granite_granite-4.0-micro_10_SM_Method.WIKIPEDIA_100000.pt"),
        ("deepseek-r1-llama3-8b", 5, "deepseek-ai_DeepSeek-R1-Distill-Llama-8B_5_SM_Method.WIKIPEDIA_100000.pt"),
        ("deepseek-7b-base", 5, "deepseek-ai_deepseek-llm-7b-base_5_SM_Method.WIKIPEDIA_100000.pt"),
        ("gpt2-xl", 16, "gpt2-xl_16_SM_Method.WIKIPEDIA_100000.pt"),
        ("llama2-7b", 5, "NousResearch_Llama-2-7b-hf_5_SM_Method.WIKIPEDIA_100000.pt"),
        ("mistral-7b-v0.3", 6, "mistralai_Mistral-7B-v0.3_6_SM_Method.WIKIPEDIA_100000.pt"),
        ("granite-4.1-8b", 16, "ibm-granite_granite-4.1-8b-base_16_SM_Method.WIKIPEDIA_100000.pt"),
        ("ministral-3-8b", 5, "mistralai_Ministral-3-8B-Base-2512_5_SM_Method.WIKIPEDIA_100000.pt"),
        ("olmo-3-1025-7b", 6, "allenai_Olmo-3-1025-7B_6_SM_Method.WIKIPEDIA_100000.pt"),
    ],
)
def test_validated_rome_configs_reference_selected_layer_covariance(
    model_key: str,
    layer: int,
    covariance_name: str,
) -> None:
    cfg = load_model_config(model_key)

    assert cfg.layer == layer
    assert Path(cfg.second_moment_path).name == covariance_name


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
