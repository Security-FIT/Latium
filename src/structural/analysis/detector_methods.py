"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations
from typing import Any
from src.structural.analysis.common import run_case_analysis
from src.structural.analysis.runtime import AnalysisContext


def analyze_gram_localization(context: AnalysisContext) -> dict[str, Any]:
    from src.structural.detectors.rome_layer_localizer import detect_from_profiles

    def analyze(data: dict[str, Any], _: str) -> dict[str, Any]:
        result = detect_from_profiles(
            data.get("profiles", {}),
            layers=data.get("layers"),
            trim_fraction=float(data.get("trim_fraction", 0.10)),
        )
        localization = result["localization"]
        selected = localization["selected_layer"]
        result["anomalous_layer"] = selected
        result["detection_score"] = float(localization["layer_scores"][str(selected)])
        result["method"] = "diagonal-relative"
        return result

    return run_case_analysis(context, "gram-localization", analyze)
