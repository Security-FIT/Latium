"""
Structural CounterFact case loading and selection metadata.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Any, Optional

from src.counterfact_selection import (
    build_case_selection_metadata,
    load_cases_by_range,
    load_cases_from_manifest,
)


def load_test_cases(
    n_tests: int,
    start_idx: int = 0,
    *,
    dataset_name: str,
    split: str,
    case_index_file: Optional[str] = None,
) -> tuple[list[dict], dict[str, Any]]:
    if case_index_file:
        manifest, cases = load_cases_from_manifest(case_index_file, n_tests=n_tests)
        case_selection = build_case_selection_metadata(
            manifest=manifest,
            manifest_path=case_index_file,
            selected_cases=cases,
        )
        return cases, case_selection

    cases = load_cases_by_range(n_tests=n_tests, start_idx=start_idx, dataset_name=dataset_name, split=split)
    case_selection = build_case_selection_metadata(
        dataset_name=dataset_name,
        split=split,
        start_idx=start_idx,
        n_cases=len(cases),
    )
    return cases, case_selection


__all__ = ["load_test_cases"]
