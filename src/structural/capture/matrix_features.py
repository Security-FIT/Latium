"""
Matrix feature-set definitions for structural captures.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from collections.abc import Iterable


PAPER_FEATURES: tuple[str, ...] = (
    "spectral_gap",
    "top1_energy",
    "row_alignment",
    "norm_cv",
    "effective_rank",
)

BLIND_FEATURES: tuple[str, ...] = (
    *PAPER_FEATURES,
    "pcs",
    "spectral_entropy",
)

RANK1_FEATURES: tuple[str, ...] = (
    "top1_energy",
    "top5_energy",
    "gap12",
    "effective_rank",
    "stable_rank",
    "rank1_residual",
)

EDIT_PRESENCE_FEATURES: tuple[str, ...] = (
    *PAPER_FEATURES,
    "frob_norm",
)

STUDY_FEATURES: tuple[str, ...] = (
    "frob_norm",
    "norm_cv",
    "row_norm_mean",
    "row_norm_std",
    "top1_energy",
    "top5_energy",
    "spectral_gap",
    "gap12",
    "effective_rank",
    "stable_rank",
    "rank1_residual",
    "spectral_entropy",
    "global_ipr",
    "row_ipr_mean",
    "row_ipr_std",
    "pcs",
    "row_alignment",
)

MATRIX_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "paper": PAPER_FEATURES,
    "blind": BLIND_FEATURES,
    "rank1": RANK1_FEATURES,
    "edit-presence": EDIT_PRESENCE_FEATURES,
    "studies": STUDY_FEATURES,
    "full": STUDY_FEATURES,
}

SVD_DERIVED_FEATURES: frozenset[str] = frozenset(
    {
        "spectral_gap",
        "top1_energy",
        "top5_energy",
        "gap12",
        "effective_rank",
        "stable_rank",
        "rank1_residual",
        "spectral_entropy",
        "row_alignment",
    }
)

PCS_FEATURES: frozenset[str] = frozenset({"pcs"})


def normalize_feature_list(values: Iterable[object]) -> tuple[str, ...]:
    features: list[str] = []
    seen: set[str] = set()
    for raw in values:
        name = str(raw).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        features.append(name)
    return tuple(features)


def resolve_matrix_features(
    feature_set: str,
    explicit_features: Iterable[object] = (),
) -> tuple[str, ...]:
    explicit = normalize_feature_list(explicit_features)
    if explicit:
        return explicit
    try:
        return MATRIX_FEATURE_SETS[str(feature_set)]
    except KeyError as exc:
        supported = ", ".join(sorted(MATRIX_FEATURE_SETS))
        raise ValueError(f"Unknown matrix feature set {feature_set!r}. Supported: {supported}") from exc


__all__ = [
    "BLIND_FEATURES",
    "EDIT_PRESENCE_FEATURES",
    "MATRIX_FEATURE_SETS",
    "PAPER_FEATURES",
    "PCS_FEATURES",
    "RANK1_FEATURES",
    "STUDY_FEATURES",
    "SVD_DERIVED_FEATURES",
    "normalize_feature_list",
    "resolve_matrix_features",
]
