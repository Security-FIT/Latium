from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from structural_benchmark import normalize_analysis_profile


def test_normalize_analysis_profile_accepts_posthoc_aliases() -> None:
    assert normalize_analysis_profile("paper") == "paper"
    assert normalize_analysis_profile("posthoc") == "paper"
    assert normalize_analysis_profile("posthoc-only") == "paper"
    assert normalize_analysis_profile("detection") == "paper"
    assert normalize_analysis_profile("detection-only") == "paper"


def test_normalize_analysis_profile_posthoc_flag_wins() -> None:
    assert normalize_analysis_profile("full", posthoc_only=True) == "paper"


def test_structural_benchmark_help_lists_paper_alias() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "structural_benchmark.py"), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--paper" in result.stdout
