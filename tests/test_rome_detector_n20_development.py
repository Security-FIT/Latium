from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


FIXTURE = Path(__file__).parent / "fixtures" / "rome_detector_n20_development.json"
EXPECTED_FILE_SHA256 = "e9b3d0a27fc5209ed6e631f261c4ddbfb3375d70b5bb2e546d65c52463349f90"
EXPECTED_EXACT = {
    "gpt2-xl": 17,
    "gpt-j-6b": 19,
    "mistral-7b-v0.1": 19,
    "mistral-7b-v0.3": 17,
    "deepseek-7b-base": 20,
    "falcon-7b": 5,
    "llama2-7b": 18,
    "granite4-micro": 17,
    "deepseek-r1-llama3-8b": 20,
    "olmo-3-1025-7b": 0,
    "granite-4.1-8b": 20,
    "ministral-3-8b": 19,
    "gemma-4-12b": 7,
}


def _load() -> dict:
    return json.loads(FIXTURE.read_text())


def test_n20_development_fixture_integrity() -> None:
    fixture = _load()
    cases = fixture["cases"]
    canonical = json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()

    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == EXPECTED_FILE_SHA256
    assert fixture["schema_version"] == "rome-detector-n20-development-v1"
    assert fixture["scientific_baseline"] is False
    assert fixture["split"] == "exposed_development"
    assert fixture["cases_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert fixture["source"]["artifact_inventory_sha256"] == (
        "65cb0f8bb20ee21810c16a9a79360b0b329e6ce4a2e3e850f94be6784e5b2cf9"
    )
    assert len(cases) == 260
    assert len({(case["model"], case["case_id"]) for case in cases}) == 260
    assert all("is_rome_compatible" not in case for case in cases)


def test_n20_development_localization_regression() -> None:
    fixture = _load()
    cases = fixture["cases"]
    expected = fixture["expected"]
    successful = [case for case in cases if case["edit_success"]]
    available = [case for case in cases if case["detector_available"]]

    assert expected["requested"] == len(cases) == 260
    assert expected["successful_edits"] == len(successful) == 240
    assert expected["detector_available"] == len(available) == 240
    assert expected["exact_successful"] == sum(case["exact"] for case in successful) == 198
    assert expected["within_one_successful"] == sum(
        case["within_one"] for case in successful
    ) == 203
    exact_by_model = Counter(case["model"] for case in successful if case["exact"])
    assert {model: exact_by_model[model] for model in EXPECTED_EXACT} == EXPECTED_EXACT


def test_n20_development_failed_edits_are_not_binary_negatives() -> None:
    fixture = _load()
    failed = [case for case in fixture["cases"] if not case["edit_success"]]

    assert len(failed) == 20
    assert all(not case["detector_available"] for case in failed)
    assert all(case["selected_layer"] is None for case in failed)
