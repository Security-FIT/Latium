from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_rome_simple_gram import (
    HARD_NEGATIVE_SCHEMA,
    collect_hard_negative_bundle,
)
from src.structural.experiments.simple_gram import PROFILE_FIELDS


def _capture() -> dict:
    return {
        "spike_statistics": {
            field: {
                "field": field,
                "selected_layer": 4,
                "robust_peak": 3.0,
                "global_prominence": 2.0,
                "local_prominence": 2.5,
            }
            for field in PROFILE_FIELDS
        }
    }


def test_hard_negative_loader_excludes_duplicate_clean_record(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bundle.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": HARD_NEGATIVE_SCHEMA,
                "model_key": "model-a",
                "records": [
                    {
                        "specimen_id": "model-a:clean",
                        "label": "clean",
                        "negative_category": "standalone_clean",
                        "capture": _capture(),
                    },
                    {
                        "specimen_id": "model-a:rank1:0",
                        "label": "hard_negative",
                        "negative_category": "matched_random_rank_one",
                        "capture": _capture(),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    records = collect_hard_negative_bundle(path)

    assert len(records) == len(PROFILE_FIELDS)
    assert {record["field"] for record in records} == set(PROFILE_FIELDS)
    assert {
        record["negative_category"] for record in records
    } == {"matched_random_rank_one"}


def test_hard_negative_loader_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "bundle.json"
    path.write_text(
        json.dumps({"schema_version": "wrong"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unexpected hard-negative schema"):
        collect_hard_negative_bundle(path)
