"""Public selection and notebook configuration for the opt-in ROME method."""

import hydra
from omegaconf import OmegaConf

from src.common.model_config import CONFIG_ROOT, load_model_config
from src.editing.registry import get_edit_method
from src.editing.rome_t4.api import build_rome_config


def test_rome_t4_composes_through_the_edit_command() -> None:
    with hydra.initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base=None):
        cfg = hydra.compose(
            config_name="latium",
            overrides=[
                "command=edit", "edit_method=rome_t4", "model=qwen3.5-4b",
                "+model.v_batch_size=4", "+model.k_batch_size=8",
                "+model.second_moment_device=cpu", "+model.insert_kv_in_place=true",
            ],
        )

    assert get_edit_method(cfg.edit_method.identifier).identifier == "rome_t4"
    assert cfg.model.v_batch_size == 4
    assert cfg.model.k_batch_size == 8
    assert cfg.model.second_moment_device == "cpu"
    assert cfg.model.insert_kv_in_place is True


def test_notebook_preset_is_local_and_explicit_overrides_win(tmp_path) -> None:
    baseline = OmegaConf.to_container(load_model_config("qwen3.5-4b"))
    path = tmp_path / "covariance.pt"
    cfg = build_rome_config(
        device="cpu", second_moment_path=path, seed=13,
        overrides={"model.v_N": 7, "model.v_batch_size": 2, "generation.k_N": 9},
    )

    assert cfg.model.name == baseline["name"]
    assert cfg.model.device == "cpu"
    assert cfg.seed == 13
    assert cfg.model.prefix_mode == "static"
    from src.editing.rome_t4.prompts import DEFAULT_SYSTEM_PROMPT

    assert cfg.model.rome_prompt_mode == "chat"
    assert cfg.model.rome_system_prompt == DEFAULT_SYSTEM_PROMPT
    assert cfg.model.epochs == 15
    assert cfg.model.v_batch_size == 2
    assert cfg.model.v_N == cfg.generation.v_N == 7
    assert cfg.model.k_N == 20
    assert cfg.generation.k_N == 9
    assert cfg.model.second_moment_path == str(path)
    assert cfg.model.second_moment_dir == str(tmp_path)
    assert cfg.runtime.second_moment_allow_autocompute is False
    assert OmegaConf.to_container(load_model_config("qwen3.5-4b")) == baseline


def test_base_model_selection_does_not_apply_notebook_preset() -> None:
    cfg = build_rome_config("qwen3.5-4b", device="cpu")
    baseline = load_model_config("qwen3.5-4b")
    assert cfg.model.prefix_mode == baseline.prefix_mode
    assert cfg.model.epochs == baseline.epochs
    assert "v_batch_size" not in cfg.model
    assert "rome_prompt_mode" not in cfg.model
