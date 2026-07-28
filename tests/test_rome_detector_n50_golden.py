from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "rome_detector_n50_golden.json"
EXPECTED_PER_MODEL = {
    "gpt2-xl": 47,
    "gpt-j-6b": 49,
    "mistral-7b-v0.1": 46,
    "mistral-7b-v0.3": 39,
    "deepseek-7b-base": 50,
    "falcon-7b": 9,
    "opt-6.7b": 50,
    "llama2-7b": 46,
    "granite4-micro": 50,
}


def _load() -> dict:
    return json.loads(FIXTURE.read_text())


def test_n50_golden_fixture_integrity() -> None:
    fixture = _load()
    cases = fixture["cases"]
    canonical = json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()

    assert fixture["schema_version"] == "rome-detector-n50-localization-v2"
    assert fixture["scientific_baseline"] is False
    assert fixture["source"]["evidence_commit"].startswith("693a949")
    assert fixture["cases_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert len(cases) == 450
    assert len({(case["model"], case["case_id"]) for case in cases}) == 450
    assert all(math.isfinite(case["target_score"]) for case in cases)
    assert all(math.isfinite(case["selected_score"]) for case in cases)
    assert all("is_rome_compatible" not in case for case in cases)


def test_n50_golden_aggregate_regression() -> None:
    fixture = _load()
    cases = fixture["cases"]
    expected = fixture["expected"]
    successful = [case for case in cases if case["edit_success"]]

    assert expected["cases"] == len(cases) == 450
    assert expected["successful_edits"] == len(successful) == 435
    assert expected["localization_correct"] == sum(
        case["selected_layer"] == case["target_layer"] for case in cases
    ) == 386
    assert expected["localization_correct_successful"] == sum(
        case["selected_layer"] == case["target_layer"] for case in successful
    ) == 375


def test_n50_golden_per_model_localization_regression() -> None:
    fixture = _load()
    cases = fixture["cases"]
    per_model = Counter(
        case["model"]
        for case in cases
        if case["selected_layer"] == case["target_layer"]
    )
    assert dict(per_model) == EXPECTED_PER_MODEL
