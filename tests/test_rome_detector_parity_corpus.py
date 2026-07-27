from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.structural.detectors.rome_presence import (
    detect_rome_presence_blind,
    detect_rome_presence_delta,
)
from src.structural.detectors.weighted_spectrum import detect_from_profiles


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "tests/fixtures/rome_detector_local_parity.json"
INVENTORY_PATH = ROOT / "tests/fixtures/rome_detector_local_inventory.json"
REQUIRED_TAGS = {
    "correct_localization",
    "localization_miss",
    "unavailable",
    "blind_positive",
    "blind_negative",
    "raw_footprint_disagreement",
    "delta_positive",
    "delta_negative",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score_order(scores: dict[str, Any]) -> list[int]:
    return [
        int(layer)
        for layer, _ in sorted(
            scores.items(),
            key=lambda item: (-float(item[1]), int(item[0])),
        )
    ]


def _assert_optional_close(actual: Any, expected: Any) -> None:
    if expected is None:
        assert actual is None
    else:
        assert float(actual) == pytest.approx(float(expected), rel=1e-5, abs=1e-7)


def test_local_parity_corpus_is_explicitly_non_scientific_and_stratified() -> None:
    fixture = _load(FIXTURE_PATH)
    inventory = _load(INVENTORY_PATH)

    assert fixture["label"] == "local refactor parity corpus"
    assert fixture["scientific_baseline"] is False
    assert "not a replacement for the published 38/40 experiment" in fixture["disclaimer"]
    assert set(fixture["model_families"]) == {
        "deepseek",
        "falcon",
        "gpt-j",
        "gpt2",
        "llama",
        "mistral",
        "opt",
        "qwen",
    }
    assert REQUIRED_TAGS <= {
        tag
        for case in fixture["cases"]
        for tag in case["selection_tags"]
    }
    assert inventory["artifact_count"] == 48
    assert {artifact["producer"] for artifact in inventory["artifacts"]} == {
        "weighted-spectrum",
        "rome-presence-blind-peak",
        "rome-presence-blind-footprint",
        "rome-presence-delta",
    }
    assert all(artifact["sha256"] for artifact in inventory["artifacts"])
    assert all(artifact["result_schema_versions"] == ["unversioned"] for artifact in inventory["artifacts"])


@pytest.mark.parametrize("strategy", ("peak", "footprint"))
def test_committed_fixture_replays_old_blind_implementation(strategy: str) -> None:
    fixture = _load(FIXTURE_PATH)
    expected_key = f"blind_{strategy}"
    for case in fixture["cases"]:
        expected = case["expected"][expected_key]
        if expected is None:
            continue
        result = detect_rome_presence_blind(
            case["profiles"],
            trim_first=5,
            trim_last=5,
            strategy=strategy,
        )
        assert result["is_rome_like"] is expected["is_rome_like"]
        assert result["verdict"] == expected["verdict"]
        assert result["localizer"]["anomalous_layer"] == expected["localized_layer"]
        assert result["anomalous_layer"] == expected["presence_peak_layer"]
        assert _score_order(result["layer_evidence"]) == expected["presence_score_order"]
        _assert_optional_close(result["evidence"]["robust_z"], expected["test_statistic"])
        _assert_optional_close(result["evidence"]["universal_threshold"], expected["threshold"])
        _assert_optional_close(result["evidence"]["peak"], expected["presence_peak"])
        _assert_optional_close(result["evidence"]["evidence_ratio"], expected["evidence_ratio"])


def test_committed_fixture_replays_old_localizer_and_delta() -> None:
    fixture = _load(FIXTURE_PATH)
    for case in fixture["cases"]:
        expected_localizer = case["expected"]["localizer"]
        if expected_localizer is not None:
            result = detect_from_profiles(
                case["profiles"],
                trim_first=5,
                trim_last=5,
            )
            assert result["anomalous_layer"] == expected_localizer["localized_layer"]
            assert result["evaluated_layers"] == expected_localizer["evaluated_layers"]
            assert _score_order(result["layer_scores"]) == expected_localizer["score_order"]
            _assert_optional_close(
                result["detection_score"],
                expected_localizer["localization_score"],
            )
            _assert_optional_close(result["margin"], expected_localizer["margin"])

        expected_delta = case["expected"]["delta"]
        if expected_delta is None:
            continue
        result = detect_rome_presence_delta(case["delta_families"])
        assert result["is_edited"] is expected_delta["changed"]
        assert result["is_rome_like"] is expected_delta["is_rome_like"]
        assert result["verdict"] == expected_delta["verdict"]
        assert result["anomalous_layer"] == expected_delta["changed_layer"]
        _assert_optional_close(
            result.get("evidence", {}).get("rank1_residual"),
            expected_delta["rank1_residual"],
        )
        _assert_optional_close(
            result.get("evidence", {}).get("roundoff_bound"),
            expected_delta["roundoff_bound"],
        )
        _assert_optional_close(result["detection_score"], expected_delta["evidence_ratio"])
