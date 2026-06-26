"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from src.results.naming import safe_slug


def _plan_prefix(model: str, plan_id: str) -> str:
    return f"plan/{safe_slug(model)}/{safe_slug(plan_id)}"


def execution_id(model: str, plan_id: str, edit_method: str | None) -> str:
    prefix = _plan_prefix(model, plan_id)
    if edit_method is None:
        return f"{prefix}/baseline/execution"
    return f"{prefix}/method/{safe_slug(edit_method)}/execution"


def capture_id(
    model: str,
    plan_id: str,
    capture: str,
    edit_method: str | None,
) -> str:
    prefix = _plan_prefix(model, plan_id)
    if edit_method is None:
        return f"{prefix}/baseline/capture/{safe_slug(capture)}"
    return f"{prefix}/method/{safe_slug(edit_method)}/capture/{safe_slug(capture)}"


def analysis_id(
    model: str,
    plan_id: str,
    edit_method: str,
    category: str,
    analysis: str,
    analysis_config_hash: str,
) -> str:
    return (
        f"{_plan_prefix(model, plan_id)}/method/{safe_slug(edit_method)}"
        f"/analysis/{safe_slug(category)}/{safe_slug(analysis)}"
        f"/{safe_slug(analysis_config_hash)}"
    )


def render_id(renderer: str) -> str:
    return f"render/{safe_slug(renderer)}"
