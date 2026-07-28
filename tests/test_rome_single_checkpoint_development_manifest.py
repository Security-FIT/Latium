from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "rome_single_checkpoint_development_v1.json"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def test_single_checkpoint_manifest_has_unique_checkpoint_specimens() -> None:
    payload = json.loads(MANIFEST.read_text())
    specimens = payload["specimens"]

    assert payload["schema_version"] == "rome-single-checkpoint-corpus-v1"
    assert payload["scientific_baseline"] is False
    assert payload["threat_model"] == "one suspect checkpoint; no paired clean input"
    assert len(specimens) == 253
    assert len({record["specimen_id"] for record in specimens}) == len(specimens)
    assert sum(record["label"] == "clean" for record in specimens) == 13
    assert sum(record["label"] == "rome" for record in specimens) == 240
    assert len(
        {
            record["base_checkpoint"]
            for record in specimens
            if record["label"] == "clean"
        }
    ) == 13
    assert payload["specimens_sha256"] == _canonical_sha256(specimens)


def test_failed_rome_cases_are_excluded_not_relabeled_negative() -> None:
    payload = json.loads(MANIFEST.read_text())

    assert len(payload["excluded"]) == 20
    assert all(
        record["reason"] == "ROME edit failed; not labeled negative"
        for record in payload["excluded"]
    )
    assert not any(
        record["label"] == "clean" and record["case_id"] != "baseline"
        for record in payload["specimens"]
    )
