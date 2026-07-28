from __future__ import annotations

import pytest

from src.structural.experiments.simple_gram import PROFILE_FIELDS
from src.structural.experiments.simple_gram_evaluation import (
    calibrate_global_cutoff,
    leave_one_family_out_presence,
    localization_summary,
)


def _presence_records() -> list[dict]:
    records = []
    for family, shift in (("a", 0.0), ("b", 0.1), ("c", 0.2)):
        records.append(
            {
                "family": family,
                "is_positive": False,
                "statistics": {"robust_peak": 1.0 + shift},
            }
        )
        records.extend(
            {
                "family": family,
                "is_positive": True,
                "statistics": {"robust_peak": value + shift},
            }
            for value in (3.0, 4.0)
        )
    return records


def test_global_cutoff_and_leave_one_family_out_are_family_generic() -> None:
    records = _presence_records()

    cutoff = calibrate_global_cutoff(records, statistic="robust_peak")
    result = leave_one_family_out_presence(
        records,
        statistic="robust_peak",
    )

    assert 1.2 < cutoff < 3.0
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 1.0
    assert result["equal_family_macro_balanced_accuracy"] == 1.0
    assert set(result["per_family"]) == {"a", "b", "c"}
    assert result["counts"] == {"positive": 6, "negative": 3}


def test_presence_reports_hard_negative_specificity() -> None:
    records = _presence_records()
    for record in records:
        record["negative_category"] = (
            None if record["is_positive"] else "standalone_clean"
        )
    for family, shift in (("a", 0.0), ("b", 0.1), ("c", 0.2)):
        records.append(
            {
                "family": family,
                "is_positive": True,
                "negative_category": None,
                "statistics": {"robust_peak": 3.0 + shift},
            }
        )
        records.append(
            {
                "family": family,
                "is_positive": False,
                "negative_category": "matched_random_rank_one",
                "statistics": {"robust_peak": 3.0 + shift},
            }
        )

    result = leave_one_family_out_presence(
        records,
        statistic="robust_peak",
    )

    assert set(result["per_negative_category"]) == {
        "matched_random_rank_one",
        "standalone_clean",
    }
    assert (
        result["per_negative_category"]["standalone_clean"]["specificity"]
        == 1.0
    )
    assert (
        result["per_negative_category"]["matched_random_rank_one"][
            "specificity"
        ]
        == 0.0
    )


def test_calibration_rejects_families_without_both_labels() -> None:
    records = _presence_records()
    records = [
        record
        for record in records
        if not (record["family"] == "a" and not record["is_positive"])
    ]

    with pytest.raises(ValueError, match="Every calibration family"):
        calibrate_global_cutoff(records, statistic="robust_peak")


def test_localization_summary_reports_each_complexity_level() -> None:
    records = [
        {
            "model": "a",
            "edit_success": True,
            "target_layer": 4,
            "selected_layers": {
                field: 4 if field != "gram_frobenius" else 5
                for field in PROFILE_FIELDS
            },
        },
        {
            "model": "b",
            "edit_success": True,
            "target_layer": 7,
            "selected_layers": {field: 7 for field in PROFILE_FIELDS},
        },
        {
            "model": "b",
            "edit_success": False,
            "target_layer": 7,
            "selected_layers": {field: 1 for field in PROFILE_FIELDS},
        },
    ]

    result = localization_summary(records)

    assert result["successful_edits"] == 2
    assert result["candidates"]["gram_frobenius"]["exact"] == 1
    assert result["candidates"]["gram_frobenius"]["within_one"] == 2
    assert result["candidates"]["diagonal_relative"]["exact"] == 2
