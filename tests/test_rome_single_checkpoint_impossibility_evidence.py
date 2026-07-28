from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = {
    "signed_clean": (
        ROOT / "analysis_out/single-checkpoint-signed-development-v1/clean-only-results.json",
        "27e1529822ef4e4b14f0dd61b06cfa1929301fd370ac9b79372cab870ba01371",
    ),
    "signed_full": (
        ROOT / "analysis_out/single-checkpoint-signed-development-v1/full-negative-results.json",
        "09144356f6e8b793572840fe308ccdf5adcdeae0eb994340dc39aa5228fa79e6",
    ),
    "two_stat": (
        ROOT / "analysis_out/single-checkpoint-two-stat-development-v1/results.json",
        "a8b0bafdc8f19aa7da7f1e199561c94a77f1a1cad6dc1a18cd867e3cd8d79b8e",
    ),
    "linear": (
        ROOT / "analysis_out/single-checkpoint-linear-development-v1/results.json",
        "b54e4b208fbcd91baf72e4716f430a9664355a47a896688f150a44f63fa99281",
    ),
}


def _payload(name: str) -> dict:
    return json.loads(EVIDENCE[name][0].read_text())


def test_single_checkpoint_evidence_files_are_immutable() -> None:
    for path, expected in EVIDENCE.values():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_signed_hard_negative_failure_is_frozen() -> None:
    payload = _payload("signed_full")

    assert payload["counts"] == {
        "records": 299,
        "successful_rome": 94,
        "failed_rome_excluded": 6,
        "standalone_clean": 5,
        "hard_negative": 200,
    }
    assert payload["pooled"]["tp"] == 76
    assert payload["pooled"]["fn"] == 18
    assert payload["pooled"]["tn"] == 74
    assert payload["pooled"]["fp"] == 131
    categories = {row["category"]: row["specificity"] for row in payload["per_negative_category"]}
    assert categories["matched_random_rank_one"] == pytest.approx(0.10)
    assert categories["multiple_layer_low_rank"] == pytest.approx(0.14)
    assert payload["completion_gate_passed"] is False


def test_best_transparent_and_linear_rules_remain_below_gates() -> None:
    transparent = _payload("two_stat")
    transparent_best = next(
        row for row in transparent["candidates"] if row["secondary"] == transparent["selected_secondary"]
    )
    linear = _payload("linear")
    linear_best = next(row for row in linear["candidates"] if row["secondary"] == linear["selected_secondary"])

    assert transparent["selected_secondary"] == "global_prominence"
    assert transparent_best["pooled"]["sensitivity"] == pytest.approx(0.7021276595744681)
    assert transparent_best["pooled"]["specificity"] == pytest.approx(0.6439024390243903)
    assert transparent_best["worst_family_balanced_accuracy"] == 0.5
    assert linear["selected_secondary"] == "robust_z"
    assert linear_best["equal_family_macro_balanced_accuracy"] == pytest.approx(0.5440785907859078)
    assert transparent["completion_gate_passed"] is False
    assert linear["completion_gate_passed"] is False
