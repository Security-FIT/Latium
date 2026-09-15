"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

from src.structural.config import AnalysisVariantConfig
from src.registry import NamedRegistry, RegistryEntry, load_object, resolve_preset_selection
from src.registry import supports_model as _supports_model


@dataclass(frozen=True)
class AnalysisSpec(RegistryEntry):
    category: str = ""
    runner: str = ""
    required_captures: tuple[str, ...] = ()
    config_fields: tuple[str, ...] = ()
    variant_fields: tuple[tuple[str, str], ...] = ()
    model_families: tuple[str, ...] = ("all",)
    required_matrix_features: tuple[str, ...] = ()

    def load(self) -> Callable[..., dict[str, Any]]:
        return load_object(self.runner)


SPECTRAL_CONFIG_FIELDS: tuple[str, ...] = (
    "top_k",
    "trim_first",
    "trim_last",
    "neighbor_layers",
    "rolling_window",
    "local_windows",
    "boundary",
)
SPECTRAL_VARIANT_FIELDS: tuple[tuple[str, str], ...] = (
    ("spectral_top_k", "top_k"),
    ("trim_first", "trim_first"),
    ("trim_last", "trim_last"),
    ("spectral_neighbor_layers", "neighbor_layers"),
    ("spectral_rolling_window", "rolling_window"),
    ("local_windows", "local_windows"),
)


def _validated_registry(entries: list[AnalysisSpec]) -> NamedRegistry[AnalysisSpec]:
    variant_sources = set(AnalysisVariantConfig().to_dict())
    for spec in entries:
        config_fields = set(spec.config_fields)
        if spec.required_matrix_features and "matrix-features" not in spec.required_captures:
            raise ValueError(
                f"Analysis {spec.identifier!r} declares matrix features without a matrix-features capture"
            )
        for source, target in spec.variant_fields:
            if source not in variant_sources:
                raise ValueError(f"Analysis {spec.identifier!r} maps unknown variant field {source!r}")
            if target not in config_fields:
                raise ValueError(
                    f"Analysis {spec.identifier!r} maps variant field {source!r} to unknown config field {target!r}"
                )
    return NamedRegistry(entries)


ANALYSES = _validated_registry(
    [
        AnalysisSpec(
            "spectral",
            "Replay spectral anomaly signals from captured decompositions.",
            "detection",
            "src.structural.detectors.spectral:analyze_spectral",
            ("spectral",),
            SPECTRAL_CONFIG_FIELDS,
            SPECTRAL_VARIANT_FIELDS,
        ),
        AnalysisSpec(
            "blind",
            "Detect a structural outlier from captured matrix profiles.",
            "detection",
            "src.structural.analysis.detector_methods:analyze_blind",
            ("matrix-features",),
            required_matrix_features=(
                "spectral_gap",
                "top1_energy",
                "row_alignment",
                "norm_cv",
                "effective_rank",
                "pcs",
                "spectral_entropy",
            ),
        ),
        AnalysisSpec(
            "ccs-composite",
            "Run the self-contained CCS composite layer detector.",
            "detection",
            "src.structural.analysis.detector_methods:analyze_ccs_composite",
            ("matrix-features", "spectral"),
            (
                *SPECTRAL_CONFIG_FIELDS,
                "small_window",
                "large_window",
                "te_window",
                "nc_window",
                "feature_z_min",
                "signal_a_confirm_z_min",
                "signal_ab_boundary_width",
                "signal_ab_cluster_span",
            ),
            SPECTRAL_VARIANT_FIELDS,
            ("non-gpt",),
            (
                "spectral_gap",
                "top1_energy",
                "row_alignment",
                "norm_cv",
                "effective_rank",
            ),
        ),
        AnalysisSpec(
            "gpt-norm-cv",
            "Run the existing GPT norm-CV detector over captured matrix profiles.",
            "detection",
            "src.structural.analysis.detector_methods:analyze_gpt_norm_cv",
            ("matrix-features",),
            ("trim_first", "trim_last"),
            (("trim_first", "trim_first"), ("trim_last", "trim_last")),
            ("gpt",),
            ("norm_cv",),
        ),
        AnalysisSpec(
            "rank1-blind",
            "Rank-one anomaly scoring from captured matrix profiles.",
            "detection",
            "src.structural.analysis.detector_methods:analyze_rank1",
            ("matrix-features",),
            ("trim_first", "trim_last", "local_windows"),
            (
                ("trim_first", "trim_first"),
                ("trim_last", "trim_last"),
                ("local_windows", "local_windows"),
            ),
            required_matrix_features=(
                "top1_energy",
                "top5_energy",
                "gap12",
                "effective_rank",
                "stable_rank",
                "rank1_residual",
            ),
        ),
        AnalysisSpec(
            "edit-presence",
            "Binary edit-presence scoring from captured matrix profiles.",
            "detection",
            "src.structural.analysis.detector_methods:analyze_edit_presence",
            ("matrix-features",),
            ("detection_threshold", "min_peak_robust_z", "min_margin", "local_windows"),
            (("local_windows", "local_windows"),),
            required_matrix_features=(
                "spectral_gap",
                "top1_energy",
                "row_alignment",
                "norm_cv",
                "effective_rank",
                "frob_norm",
            ),
        ),
        AnalysisSpec(
            "bottom-rank-svd",
            "Bottom-rank layer detection from captured token sweeps.",
            "detection",
            "src.structural.analysis.detector_methods:analyze_bottom_rank",
            ("bottom-rank-tokens",),
            ("trim_first", "trim_last"),
            (("trim_first", "trim_first"), ("trim_last", "trim_last")),
        ),
        AnalysisSpec(
            "gram-localization",
            "Localize a ROME-style edit from a single-checkpoint profile.",
            "detection",
            "src.structural.analysis.detector_methods:analyze_gram_localization",
            ("gram-localization",),
        ),
        AnalysisSpec(
            "rome-profile-experiments",
            "Replay ROME scalar background experiments from saved profiles.",
            "artifact-study",
            "src.structural.analysis.detector_methods:analyze_rome_profile_experiments",
            ("gram-localization",),
            ("experiments",),
        ),
        AnalysisSpec(
            "rome-matrix-experiments",
            "Replay ROME matrix-score experiments from a shared capture.",
            "artifact-study",
            "src.structural.analysis.detector_methods:analyze_rome_matrix_experiments",
            ("gram-experiments-v1",),
            ("experiments",),
        ),
        AnalysisSpec(
            "rome-control-experiment",
            "Compare the ROME target profile with an attention-output control.",
            "artifact-study",
            "src.structural.analysis.detector_methods:analyze_rome_control_experiment",
            ("gram-experiments-v1", "gram-control-v1"),
            ("experiments",),
        ),
        AnalysisSpec(
            "ipr",
            "IPR layer profiles and anomaly summaries.",
            "artifact-study",
            "src.structural.analysis.studies:analyze_ipr",
            ("matrix-features",),
            ("trim_first", "trim_last"),
            (("trim_first", "trim_first"), ("trim_last", "trim_last")),
            required_matrix_features=("global_ipr", "row_ipr_mean", "row_ipr_std"),
        ),
        AnalysisSpec(
            "symmetry",
            "Mirror-layer symmetry study from captured profiles.",
            "artifact-study",
            "src.structural.analysis.studies:analyze_symmetry",
            ("matrix-features",),
            ("local_windows",),
            (("local_windows", "local_windows"),),
            required_matrix_features=("top1_energy", "effective_rank", "stable_rank"),
        ),
        AnalysisSpec(
            "interlayer",
            "Cross-layer transitions from captured matrix profiles.",
            "artifact-study",
            "src.structural.analysis.studies:analyze_interlayer",
            ("matrix-features",),
            required_matrix_features=("top1_energy", "spectral_gap", "effective_rank", "norm_cv"),
        ),
        AnalysisSpec(
            "attention",
            "Attention-to-MLP structural contrast.",
            "artifact-study",
            "src.structural.analysis.studies:analyze_attention",
            ("matrix-features", "attention-features"),
            ("local_windows",),
            (("local_windows", "local_windows"),),
            required_matrix_features=(
                "spectral_gap",
                "top1_energy",
                "row_alignment",
                "norm_cv",
                "effective_rank",
            ),
        ),
        AnalysisSpec(
            "matrix-anomaly",
            "Experimental matrix anomaly metric study.",
            "artifact-study",
            "src.structural.analysis.studies:analyze_matrix_anomaly",
            ("matrix-anomaly-features",),
        ),
    ]
)

ANALYSIS_PRESETS: dict[str, tuple[str, ...]] = {
    "none": (),
    "paper": ("ccs-composite", "gpt-norm-cv", "spectral"),
    **{identifier: (identifier,) for identifier in ANALYSES.identifiers()},
    "full": ANALYSES.identifiers(),
}


def supports_model(spec: AnalysisSpec, model: str) -> bool:
    return _supports_model(spec, model)


def resolve_analyses(
    preset: str,
    *,
    enabled: Sequence[str] = (),
    disabled: Sequence[str] = (),
) -> tuple[str, ...]:
    return resolve_preset_selection(
        ANALYSIS_PRESETS,
        ANALYSES,
        preset,
        enabled=enabled,
        disabled=disabled,
        preset_label="analysis preset",
    )
