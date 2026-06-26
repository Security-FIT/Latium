"""
Architecture import hygiene tests.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PATHS = (ROOT / "src", ROOT / "scripts", ROOT / "rome_benchmark.py")


def _python_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(candidate for candidate in path.rglob("*.py") if "__pycache__" not in candidate.parts)


def test_production_code_imports_named_modules_instead_of_utils_facade() -> None:
    offenders = []
    for base in PRODUCTION_PATHS:
        for path in _python_files(base):
            if path == ROOT / "src" / "utils.py":
                continue
            text = path.read_text(encoding="utf-8")
            if "from src.utils import" in text or "import src.utils" in text:
                offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []


def test_production_code_imports_common_model_config_instead_of_facade() -> None:
    offenders = []
    for base in PRODUCTION_PATHS:
        for path in _python_files(base):
            if path == ROOT / "src" / "model_config.py":
                continue
            text = path.read_text(encoding="utf-8")
            if "from src.model_config import" in text or "import src.model_config" in text:
                offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []
