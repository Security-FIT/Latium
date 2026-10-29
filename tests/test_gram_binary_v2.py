"""Deferred validation for the research extensions. Not executed at implementation time."""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from src.structural.detectors import gram_binary
from src.structural.detectors.gram_covariance import CovarianceManifest, Metric, manifest_signature
from src.structural.detectors.gram_mdl_codec import GramCodec
from src.structural.detectors.gram_mdl_v2 import BACKGROUNDS, BackgroundFamily, describe_adaptive_corrections
from src.structural.detectors.gram_peel import peel_in_gram_space
from src.structural.detectors.gram_research import bi_cross_validation, dyson_equalize
from src.structural.detectors.rome_layer_localizer import hidden_gram


def test_adaptive_correction_reconstructs_original_source_in_both_views():
    generator = torch.Generator().manual_seed(81)
    weight = torch.randn(6, 14, generator=generator)
    metric = torch.diag(torch.linspace(0.5, 2.0, 14))
    current = {"ordinary": hidden_gram(weight), "covariance": Metric(metric).gram(weight.double())}
    reference = torch.eye(6) / 6
    direction = torch.randn(6, generator=generator)
    peel = peel_in_gram_space(current["ordinary"], reference, direction, layer=2)
    codec = GramCodec(6)
    selected, choices = describe_adaptive_corrections(current, peel.direction, peel.mixer, codec)
    for view in current:
        np.testing.assert_array_equal(selected.views[view].reconstruct(codec), codec.quantize(current[view]))
    assert selected.bits == min(item["parameter_bits"] + item["reconstruction_bits"] for item in choices)
    assert selected.parameter_bits == selected.record()["precision_bits"] + sum(selected.parameter_blocks.values())


def test_local_recoding_matches_full_joint_background_reencoding():
    rng = np.random.default_rng(91)
    source = {
        view: [rng.integers(-100, 100, size=15, dtype=np.int64) for _ in range(9)]
        for view in ("ordinary", "covariance")
    }
    family = BackgroundFamily(lambda view, pos: source[view][pos], tuple(source), 9)
    replacement = {view: rng.integers(-10, 10, size=15, dtype=np.int64) for view in source}
    local = family.replaced(3, replacement)
    rebuilt = BackgroundFamily(lambda view, pos: replacement[view] if pos == 3 else source[view][pos], tuple(source), 9)
    assert local == {model: rebuilt.baseline[model].bits for model in BACKGROUNDS}


def test_dyson_normalization_is_equivariant_to_positive_global_scale():
    value = torch.randn(6, 14, dtype=torch.float64, generator=torch.Generator().manual_seed(55))
    first, second = dyson_equalize(value), dyson_equalize(3.0 * value)
    torch.testing.assert_close(first.matrix, second.matrix, rtol=1e-10, atol=1e-10)
    assert first.record["calibrated_for_llm_weights"] is False
    assert first.record["lss"]["calibrated_for_llm_weights"] is False


def test_held_out_rank_one_prediction_uses_observed_bridges():
    left = torch.arange(1, 9, dtype=torch.float64)
    right = torch.arange(1, 15, dtype=torch.float64)
    result = bi_cross_validation(torch.outer(left, right))
    assert result["one_rank_loss"] == pytest.approx(0.0, abs=1e-7)
    assert result["zero_rank_loss"] > 0
    assert result["held_out_entries"] == 8 * 14
    assert result["used_full_data_preprocessing"] is False


def test_covariance_and_explicit_precision_views_agree():
    weight = torch.randn(6, 14, dtype=torch.float64, generator=torch.Generator().manual_seed(16))
    covariance = torch.diag(torch.linspace(0.5, 2.0, 14, dtype=torch.float64))
    first = Metric(covariance, "covariance").gram(weight)
    second = Metric(torch.linalg.inv(covariance), "precision").gram(weight)
    torch.testing.assert_close(first, second)


def test_manifest_requires_complete_coverage_and_changes_content_identity(tmp_path):
    path = tmp_path / "metrics.json"
    metric_path = tmp_path / "metric.pt"
    torch.save(torch.eye(6), metric_path)
    manifest = {"version": 1, "kind": "precision", "layers": {"0": {"path": "metric.pt"}}}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="every scanned layer"):
        CovarianceManifest(path, [0, 1, 2])
    first = manifest_signature(path)
    torch.save(torch.eye(6) * 2, metric_path)
    assert manifest_signature(path) != first


def test_v2_replays_all_layers_without_diagnostics_and_rejects_modified_prices():
    grams = {layer: torch.eye(4) / 4 for layer in range(5)}
    evidence = gram_binary.profile_grams(grams)
    result = gram_binary.detect_from_evidence(evidence)
    assert evidence["eligible_layers"] == list(grams)
    assert evidence["excluded_layers"] == []
    assert result["edited"] == (result["edit_bits"] < result["null_bits"])
    assert result["method"] == "gram-peel-mdl-v2"
    json.dumps(result, allow_nan=False)
    altered = deepcopy(evidence)
    candidate = next(iter(altered["candidates"].values()))
    generated = next(item for item in candidate["generators"].values() if item["status"] == "complete")
    generated["precision_options"][0]["parameter_bits"] += 1
    with pytest.raises(ValueError, match="parameter cost"):
        gram_binary.detect_from_evidence(altered)
