"""Paid model selection, exact reconstruction and single-checkpoint integration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import hydra
import numpy as np
import pytest
import torch

from src.structural.analysis.detector_methods import analyze_gram_binary
from src.structural.analysis.runtime import AnalysisContext
from src.structural.capture.producers import CaptureContext, capture_gram_binary
from src.structural.capture.registry import CAPTURES
from src.structural.detectors import gram_binary_v1 as gram_binary
from src.structural.detectors.gram_mdl_codec import GramCodec, describe_correction, integer_code
from src.structural.detectors.gram_peel import peel_in_gram_space
from src.structural.detectors.rome_layer_localizer import hidden_gram
from src.structural.hydra_config import structural_config_from_hydra
from src.structural.planning import build_plan_summary


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def smooth_weights() -> dict[int, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    weight = torch.randn(24, 64, generator=generator)
    weight[-4:] = 0
    return {layer: weight.clone() for layer in range(12)}


@pytest.fixture(scope="module")
def evidence():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    weights = smooth_weights()
    clean = gram_binary.profile_weights(weights)
    generator = torch.Generator().manual_seed(3)
    value = torch.zeros(24)
    value[-1] = 1
    key = torch.randn(64, generator=generator)
    weights[6] = weights[6] + torch.outer(value, key) * 4
    edited = gram_binary.profile_weights(weights)
    torch.set_num_threads(previous)
    return clean, edited, weights


def test_integer_code_accounts_for_actual_words_and_parameter():
    # All zeros: mode bit + gamma(k+1) at k=0 + unary terminator per word.
    assert integer_code(np.zeros(100, dtype=np.int64)).bits == 102
    # Check against an independently enumerated prefix-code length.
    values = np.array([-100, -1, 0, 1, 300], dtype=np.int64)
    gamma = lambda n: 2 * (n + 1).bit_length() - 1
    zigzag = [2 * int(v) if v >= 0 else -2 * int(v) - 1 for v in values]
    width = max(abs(int(v)) for v in values).bit_length() + 1
    costs = [1 + gamma(width - 1) + len(values) * width]
    costs.extend(1 + gamma(k) + sum((v >> k) + 1 + k for v in zigzag) for k in range(20))
    assert integer_code(values).bits == min(costs)
    with pytest.raises(ValueError, match="safe coding range"):
        integer_code(np.array([np.iinfo(np.int64).min, 1], dtype=np.int64))


def test_correction_roundtrip_restores_original_lattice_and_matches_weights():
    generator = torch.Generator().manual_seed(42)
    weight = torch.randn(8, 16, generator=generator, dtype=torch.float64)
    current = hidden_gram(weight)
    reference = hidden_gram(weight + 0.03 * torch.randn(8, 16, generator=generator, dtype=torch.float64))
    direction = torch.randn(8, generator=generator, dtype=torch.float64)
    peel = peel_in_gram_space(current, reference, direction, layer=3)
    codec = GramCodec(8)
    encoded = describe_correction(current, peel.gram, peel.direction, peel.mixer, codec)
    np.testing.assert_array_equal(encoded.reconstruct(codec), codec.quantize(current))
    key = weight.T @ peel.mixer
    peeled_weight = weight - torch.outer(peel.direction, key)
    torch.testing.assert_close(hidden_gram(peeled_weight), peel.gram)
    assert encoded.parameter_bits == 64 * (2 * 8 + 2)
    assert encoded.reconstruction_bits >= codec.entries


def test_fp32_refinement_roundtrip_preserves_words_including_signed_zero():
    codec = GramCodec(8)
    generator = torch.Generator().manual_seed(4)
    weight = torch.randn(8, 16, generator=generator)
    gram = hidden_gram(weight)
    gram[0, 1] = gram[1, 0] = -0.0
    lattice, indices = codec.encode_refinement(gram)
    restored = codec.restore_refinement(lattice, indices)
    original = codec.vector(gram).astype(np.float32)
    np.testing.assert_array_equal(restored.view(np.uint32), original.view(np.uint32))
    assert 0 <= codec.refinement_bits(lattice) <= 32 * codec.entries


def test_zero_edit_model_can_win_and_strong_isolated_edit_can_win(evidence):
    clean, edited, _ = evidence
    negative = gram_binary.detect_from_evidence(clean)
    positive = gram_binary.detect_from_evidence(edited)
    assert negative["edited"] is False and negative["layer"] is None
    assert negative["mdl_delta_bits"] < 0
    assert positive["edited"] is True and positive["layer"] == 6
    assert positive["mdl_delta_bits"] == positive["null_bits"] - positive["edit_bits"] > 0
    assert len(positive["candidates"]) == len(positive["eligible_layers"])


def test_replay_tie_prefers_no_edit_and_equal_candidates_prefer_lower_layer(evidence):
    profile = deepcopy(evidence[0])
    # Make H0 exactly equal to the best already paid H1.
    best = min(item["edit_bits"] for item in profile["candidates"].values())
    profile["null_bits"] = best
    profile["null_background_bits"] = best - profile["common_source_bits"]
    extra = best - sum(profile["null_block_bits"].values()) - profile["common_source_bits"]
    profile["null_block_bits"][str(profile["layers"][0])] += extra
    for candidate in profile["candidates"].values():
        candidate["mdl_delta_bits"] = best - candidate["edit_bits"]
    result = gram_binary.detect_from_evidence(profile)
    assert result["mdl_delta_bits"] == 0 and result["edited"] is False


def test_partial_nonfinite_and_inconsistent_evidence_never_becomes_clean(evidence):
    profile = deepcopy(evidence[0])
    del profile["candidates"][str(profile["eligible_layers"][0])]
    with pytest.raises(ValueError, match="every eligible layer"):
        gram_binary.detect_from_evidence(profile)
    profile = deepcopy(evidence[0])
    next(iter(profile["candidates"].values()))["edit_bits"] += 1
    with pytest.raises(ValueError, match="cost is inconsistent"):
        gram_binary.detect_from_evidence(profile)
    profile = deepcopy(evidence[0])
    next(iter(profile["candidates"].values()))["collapse"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        gram_binary.detect_from_evidence(profile)


def test_capture_replay_ignores_config_ct_and_changed_weight_hints(evidence, tmp_path):
    clean, _, _ = evidence
    context = AnalysisContext(
        run_root=tmp_path,
        model="gpt2-large",
        plan_id="test",
        edit_method="rome",
        target_layer=1,
        execution={"cases": [{"case_id": "one", "status": "complete"}]},
        captures={"gram-binary": [{"case_id": "one", "status": "complete", "data": clean}]},
        config={"model.layer": 1, "causal_tracing_layer": 1},
    )
    first = analyze_gram_binary(context)
    second = analyze_gram_binary(replace(context, target_layer=9, config={"model.layer": 9}))
    assert first["cases"][0]["data"] == second["cases"][0]["data"]
    assert first["summary"]["binary_negative"] == 1
    assert CAPTURES.get("gram-binary").requires_baseline is False
    assert CAPTURES.get("gram-binary").requires_probe is False
    # Invalid evidence becomes an error case without an edited:false payload.
    bad = deepcopy(clean)
    bad["candidates"] = {}
    failure = analyze_gram_binary(
        replace(context, captures={"gram-binary": [{"case_id": "one", "status": "complete", "data": bad}]})
    )
    assert failure["cases"][0]["status"] == "error"
    assert failure["cases"][0]["data"] == {}


def test_presets_keep_original_gram_and_resolve_binary_without_probes():
    root = Path(__file__).resolve().parents[1]
    with hydra.initialize_config_dir(config_dir=str(root / "src/config"), version_base=None):
        cfg = hydra.compose(config_name="latium", overrides=["command=structural/plan", "structural=gram-binary"])
    plan = build_plan_summary(structural_config_from_hydra(cfg, run_analysis=True))
    assert plan["resolved_captures"] == ["gram-localization", "gram-binary"]
    assert plan["resolved_analyses"] == ["gram-localization", "gram-binary"]
    assert plan["resolved_renderers"] == ["gram-report"]


@pytest.mark.parametrize("mutation", ["nan", "shape", "trace", "indefinite"])
def test_invalid_grams_are_rejected(mutation):
    grams = {layer: torch.eye(4) / 4 for layer in range(5)}
    if mutation == "nan":
        grams[2][0, 0] = float("nan")
    elif mutation == "shape":
        grams[2] = torch.eye(5) / 5
    elif mutation == "trace":
        grams[2] *= 2
    else:
        grams[2][0, 1] = grams[2][1, 0] = 1
    with pytest.raises(ValueError):
        gram_binary.profile_grams(grams)


def test_weight_layout_parity_and_no_mutation():
    weights = smooth_weights()
    snapshot = {layer: weight.clone() for layer, weight in weights.items()}
    capture = CaptureContext(weights, None, {}, None, None, {"proj": (5,)},
                             {"model.layer": 5, "gram_binary": {"method": "v1"}})
    first = capture_gram_binary(capture)
    transposed = gram_binary.profile_weights({layer: weight.T.contiguous() for layer, weight in weights.items()})
    one = gram_binary.detect_from_evidence(first)
    two = gram_binary.detect_from_evidence(transposed)
    for field in ("edited", "layer", "mdl_delta_bits", "null_bits", "edit_bits"):
        assert one[field] == two[field]
    assert {layer: item["edit_bits"] for layer, item in first["candidates"].items()} == {
        layer: item["edit_bits"] for layer, item in transposed["candidates"].items()
    }
    assert json.loads(json.dumps(first)) == first
    for layer in weights:
        assert torch.equal(weights[layer], snapshot[layer])


def test_broad_natural_transition_can_remain_negative():
    weights = smooth_weights()
    weights[6] = torch.diag(torch.linspace(0.3, 2.0, 24)) @ weights[6]
    result = gram_binary.detect(weights)
    assert result["edited"] is False
    assert result["layer"] is None


def test_bulk_scale_recovers_trace_ratio_when_reference_is_clean():
    generator = torch.Generator().manual_seed(44)
    clean = torch.randn(12, 32, generator=generator)
    value = torch.randn(12, generator=generator)
    key = torch.randn(32, generator=generator)
    edited = clean + torch.outer(value, key)
    expected = float(clean.square().sum() / edited.square().sum())
    measured = gram_binary.normalization_scale(hidden_gram(edited), hidden_gram(clean))
    assert measured == pytest.approx(expected, rel=1e-5)


def test_standalone_cli_scans_without_model_loading_or_editing(tmp_path):
    source = tmp_path / "weights.pt"
    output = tmp_path / "result.json"
    torch.save(smooth_weights(), source)
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/gram_binary_detect.py"),
            "--weights",
            str(source),
            "--out",
            str(output),
            "--device",
            "cpu",
            "--method",
            "v1",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    result = json.loads(output.read_text())
    printed = json.loads(completed.stdout.strip())
    assert printed["edited"] is result["edited"] is False
    assert printed["mdl_delta_bits"] == result["mdl_delta_bits"]
