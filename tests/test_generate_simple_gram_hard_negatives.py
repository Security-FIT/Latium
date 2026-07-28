from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.generate_simple_gram_hard_negatives import (
    _matched_update,
    _magnitude_targets,
)
from src.structural.detectors.weighted_spectrum import hidden_gram


@pytest.mark.parametrize("shape", ((5, 8), (8, 5)))
def test_matched_rank_one_update_matches_hidden_gram_magnitude(
    shape: tuple[int, int],
) -> None:
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(*shape, generator=generator)

    edited, evidence = _matched_update(
        weight,
        rank=1,
        target_magnitude=0.08,
        generator=generator,
    )
    difference = edited - weight
    gram_change = hidden_gram(
        edited,
        normalize=False,
    ) - hidden_gram(weight, normalize=False)
    actual = float(
        torch.linalg.matrix_norm(gram_change, ord="fro")
    ) / float(
        torch.linalg.matrix_norm(
            hidden_gram(weight, normalize=False),
            ord="fro",
        )
    )

    assert int(torch.linalg.matrix_rank(difference)) == 1
    assert evidence["actual_change_magnitude"] == pytest.approx(actual)
    assert actual == pytest.approx(0.08, rel=1e-5)
    assert evidence["relative_match_error"] < 1e-5


def test_prior_bundle_can_supply_generation_magnitudes(tmp_path: Path) -> None:
    source = tmp_path / "prior.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "rome-single-checkpoint-hard-negatives-v1",
                "records": [
                    {
                        "source_rome_case_id": "4",
                        "update": {"target_change_magnitude": 0.12},
                    },
                    {
                        "source_rome_case_id": "4",
                        "update": {"target_change_magnitude": 0.12},
                    },
                    {
                        "source_rome_case_id": "7",
                        "update": {
                            "first": {"target_change_magnitude": 0.09}
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert _magnitude_targets(source) == [
        {"case_id": "4", "change_magnitude": 0.12},
        {"case_id": "7", "change_magnitude": 0.09},
    ]
