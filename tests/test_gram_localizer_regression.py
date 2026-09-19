from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.structural.detectors.rome_layer_localizer import detect_from_profiles


FIXTURE = Path(__file__).parent / "fixtures" / "gram_localizer_profiles.json"
EXPECTED_MODELS = {
    "deepseek-7b-base",
    "deepseek-r1-llama3-8b",
    "falcon-7b",
    "gemma-4-12b",
    "gpt-j-6b",
    "gpt2-xl",
    "granite-4.1-8b",
    "granite4-micro",
    "llama2-7b",
    "ministral-3-8b",
    "mistral-7b-v0.1",
    "mistral-7b-v0.3",
    "olmo-3-1025-7b",
}


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_evidence_aggregate_and_provenance_are_frozen() -> None:
    data = _fixture()
    aggregate = data["aggregate"]

    assert data["evidence_class"] == "exposed-development"
    assert data["scientific_baseline"] is False
    assert data["source"]["evaluation_sha256"] == (
        "f119e82ccba128be21970cf4f730243adc5ddcc9078362a2abd2d5f6f4aefe88"
    )
    assert data["source"]["evaluation_content_sha256"] == (
        "f8565f0e8eb259f830cf7fb7895ef081503dc3c74514947c5f65b327cd8f2b6e"
    )
    assert aggregate["requested_cases"] == 260
    assert aggregate["successful_edits"] == 240
    assert aggregate["exact"] == 196
    assert aggregate["within_one"] == 201
    assert sum(row["successful"] for row in aggregate["per_model"].values()) == 240
    assert sum(row["exact"] for row in aggregate["per_model"].values()) == 196
    assert sum(row["within_one"] for row in aggregate["per_model"].values()) == 201
    assert set(aggregate["per_model"]) == EXPECTED_MODELS


def test_balanced_samples_replay_with_only_score_field() -> None:
    data = _fixture()
    samples = data["samples"]

    assert len(samples) == len(EXPECTED_MODELS)
    assert {sample["model"] for sample in samples} == EXPECTED_MODELS
    assert all(sample["edit_success"] is True for sample in samples)
    assert data["aggregate"]["successful_edits"] < data["aggregate"]["requested_cases"]

    for sample in samples:
        assert all(set(profile) == {"diagonal_relative"} for profile in sample["profiles"].values())
        result = detect_from_profiles(
            sample["profiles"],
            layers=sample["layers"],
            trim_fraction=sample["trim_fraction"],
        )
        assert result["localization"]["selected_layer"] == sample["expected_selected_layer"]

    canonical = json.dumps(samples, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == data["canonical_samples_sha256"]


def test_fixture_contains_no_experimental_or_binary_detector_payloads() -> None:
    text = FIXTURE.read_text(encoding="utf-8")
    forbidden = (
        "m3_control",
        "gram_frobenius",
        "gram_relative",
        "scalar_relative",
        "top2_frobenius",
        "binary",
        "cutoff",
        "prompt",
        "subject",
        "clean_weight",
        "tensor",
    )
    assert all(term not in text for term in forbidden)
