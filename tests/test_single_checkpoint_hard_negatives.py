from __future__ import annotations

import pytest
import torch

from scripts.generate_single_checkpoint_hard_negatives import _matched_update
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
    gram_change = hidden_gram(edited, normalize=False) - hidden_gram(
        weight,
        normalize=False,
    )
    actual = float(torch.linalg.matrix_norm(gram_change, ord="fro")) / float(
        torch.linalg.matrix_norm(hidden_gram(weight, normalize=False), ord="fro")
    )

    assert int(torch.linalg.matrix_rank(difference)) == 1
    assert evidence["actual_change_magnitude"] == pytest.approx(actual)
    assert actual == pytest.approx(0.08, rel=1e-5)
    assert evidence["relative_match_error"] < 1e-5
