"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import hydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.manual_rome import fact_tuple_from_manual_config


def test_manual_fact_tuple_normalizes_targets() -> None:
    cfg = OmegaConf.create(
        {
            "manual": {
                "prompt": "{} was born in",
                "subject": "Ada Lovelace",
                "target_new": "Paris",
                "target_true": "London",
            }
        }
    )

    assert fact_tuple_from_manual_config(cfg) == (
        "{} was born in",
        "Ada Lovelace",
        " Paris",
        " London",
    )


def test_counterfact_path_uses_requested_rewrite(tmp_path) -> None:
    path = tmp_path / "counterfact.json"
    path.write_text(
        json.dumps(
            [
                {
                    "case_id": 7,
                    "requested_rewrite": {
                        "prompt": "{} is located in",
                        "subject": "The Eiffel Tower",
                        "target_new": {"str": "Rome"},
                        "target_true": {"str": "Paris"},
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create({"manual": {"counterfact_path": str(path), "case_id": 7}})

    assert fact_tuple_from_manual_config(cfg) == (
        "{} is located in",
        "The Eiffel Tower",
        " Rome",
        " Paris",
    )


def test_missing_manual_fields_raise_clear_error() -> None:
    cfg = OmegaConf.create({"manual": {"prompt": "{} is located in"}})

    with pytest.raises(ValueError, match="manual.subject, manual.target_new"):
        fact_tuple_from_manual_config(cfg)


def test_hydra_manual_rome_defaults_provide_a_fact() -> None:
    with hydra.initialize_config_dir(config_dir=str(ROOT / "src" / "config"), version_base=None):
        cfg = hydra.compose(config_name="latium", overrides=["command=manual_rome", "model=gpt2-large"])

    assert fact_tuple_from_manual_config(cfg) == (
        "{} is located in",
        "The Eiffel Tower",
        " Rome",
        " Paris",
    )
