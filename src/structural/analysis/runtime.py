"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from src.results import ArtifactWriter, RunArtifactReader, build_artifact, config_hash
from src.results.ids import analysis_id, capture_id, execution_id
from src.results.layout import RunLayout
from src.common.config import plain as _plain
from src.structural.analysis.registry import ANALYSES, AnalysisSpec, resolve_analyses, supports_model
from src.structural.capture.registry import CAPTURES
from src.structural.analysis.trim import resolve_trim


@dataclass(frozen=True)
class AnalysisContext:
    run_root: Path
    model: str
    plan_id: str
    edit_method: str
    target_layer: Optional[int]
    execution: dict[str, Any]
    captures: dict[str, list[dict[str, Any]]]
    config: dict[str, Any]


class AnalysisUnavailableError(RuntimeError):
    """Raised when saved captures cannot satisfy an analysis configuration."""


class AnalysisExecutionError(RuntimeError):
    """Raised after analysis failures have been persisted as artifacts."""


STRUCTURAL_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "structural" / "default.yaml"


def _default_method_configs() -> dict[str, dict[str, Any]]:
    from omegaconf import OmegaConf

    structural = _plain(OmegaConf.load(STRUCTURAL_DEFAULT_CONFIG))
    analysis = structural.get("analysis", {}) if isinstance(structural, Mapping) else {}
    methods = analysis.get("methods", {}) if isinstance(analysis, Mapping) else {}
    if not isinstance(methods, Mapping):
        return {}
    return {str(key): dict(value) for key, value in methods.items() if isinstance(value, Mapping)}


def _case_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(case.get("case_id")): case
        for case in payload.get("cases", [])
        if isinstance(case, dict) and case.get("case_id") is not None
    }


def _merge_dict(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _overlay_spectral(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = _merge_dict(
        base, {key: value for key, value in patch.items() if key not in {"pcs_pairwise_rows", "mode", "changed_layers"}}
    )
    rows = patch.get("pcs_pairwise_rows", {})
    mapping = (
        ("dot_weight_cumsum", "pcs_pairwise_dot_weight_cumsum"),
        ("weight_cumsum", "pcs_pairwise_weight_cumsum"),
    )
    for patch_key, target_key in mapping:
        source_rows = rows.get(patch_key, {})
        if not source_rows or target_key not in merged:
            continue
        matrix = np.asarray(merged[target_key], dtype=np.float64)
        for raw_index, raw_row in source_rows.items():
            index = int(raw_index)
            row = np.asarray(raw_row, dtype=np.float64)
            matrix[:, index, :] = row
            matrix[:, :, index] = row
        merged[target_key] = matrix.tolist()
    merged["mode"] = "materialized"
    return merged


def materialize_capture(
    baseline: dict[str, Any],
    method: dict[str, Any],
) -> list[dict[str, Any]]:
    baseline_cases = list(baseline.get("cases", []))
    baseline_data = baseline_cases[0].get("data", {}) if baseline_cases and isinstance(baseline_cases[0], dict) else {}
    baseline_case = baseline_cases[0] if baseline_cases and isinstance(baseline_cases[0], dict) else None
    output: list[dict[str, Any]] = []
    for case in method.get("cases", []):
        if not isinstance(case, dict):
            continue
        case_id = case.get("case_id")
        if baseline_case is None or baseline_case.get("status") != "complete":
            output.append(
                {
                    "case_id": case_id,
                    "status": ("error" if baseline_case and baseline_case.get("status") == "error" else "unavailable"),
                    "data": {},
                    "error": (baseline_case.get("error") if baseline_case else "baseline capture is missing"),
                }
            )
            continue
        if case.get("status") != "complete":
            output.append(deepcopy(case))
            continue
        patch = case.get("data", {})
        if method.get("producer") == "spectral":
            data = _overlay_spectral(baseline_data, patch)
        else:
            data = _merge_dict(
                baseline_data, {key: value for key, value in patch.items() if key not in {"mode", "changed_layers"}}
            )
            data["mode"] = "materialized"
        output.append(
            {
                **{key: deepcopy(value) for key, value in case.items() if key != "data"},
                "data": data,
            }
        )
    return output


def _unavailable_cases(execution: dict[str, Any], error: str) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case.get("case_id"),
            "status": "unavailable",
            "data": {},
            "error": error,
        }
        for case in execution.get("cases", [])
        if isinstance(case, dict)
    ]


def _resolve_analysis_config(
    spec: AnalysisSpec,
    variant: Mapping[str, Any],
    *,
    num_layers: int,
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    trim_first, trim_last = resolve_trim(
        num_layers,
        variant.get("trim_first"),
        variant.get("trim_last"),
    )
    normalized_variant = {
        **dict(variant),
        "trim_first": trim_first,
        "trim_last": trim_last,
    }
    projected = {
        target: normalized_variant[source] for source, target in spec.variant_fields if source in normalized_variant
    }
    resolved = {
        **dict(defaults),
        **projected,
        **dict(overrides),
    }
    unknown = sorted(set(resolved) - set(spec.config_fields))
    if unknown:
        raise ValueError(f"Analysis {spec.identifier!r} has unknown config keys: {', '.join(unknown)}")
    return resolved


def _analysis_variants(
    execution: Mapping[str, Any],
    record: Mapping[str, Any],
) -> Any:
    record_metadata = record.get("metadata")
    if isinstance(record_metadata, Mapping) and isinstance(record_metadata.get("analysis_variants"), list):
        return record_metadata["analysis_variants"]
    metadata = execution.get("metadata")
    if isinstance(metadata, Mapping) and isinstance(metadata.get("analysis_variants"), list):
        return metadata["analysis_variants"]
    return execution.get("config", {}).get("analysis_variants")


def _status_summary(
    cases: list[dict[str, Any]],
    summary: Mapping[str, Any],
) -> tuple[str, dict[str, Any], Optional[str]]:
    complete = sum(case.get("status") == "complete" for case in cases)
    unavailable = sum(case.get("status") == "unavailable" for case in cases)
    errors = sum(case.get("status") == "error" for case in cases)
    resolved_summary = {
        **dict(summary),
        "cases_total": len(cases),
        "cases_complete": complete,
        "cases_unavailable": unavailable,
        "cases_error": errors,
    }
    if errors:
        return "error", resolved_summary, "one or more analysis cases failed"
    if complete:
        return "complete", resolved_summary, None
    return "unavailable", resolved_summary, "no analysis cases were available"


def run_analyses(
    run_root: str | Path,
    *,
    selected: Sequence[str] = (),
    preset: str = "paper",
    disabled: Sequence[str] = (),
    method_configs: Optional[Mapping[str, Mapping[str, Any]]] = None,
    config_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    force: bool = False,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    root = Path(run_root)
    reader = RunArtifactReader(root)
    writer = ArtifactWriter(root)
    layout = RunLayout(root)
    analysis_ids = resolve_analyses(preset, enabled=selected, disabled=disabled)
    defaults = {
        str(key): dict(value)
        for key, value in (method_configs if method_configs is not None else _default_method_configs()).items()
    }
    overrides = dict(config_overrides or {})
    written: list[str] = []
    skipped: list[str] = []
    failures: list[str] = []

    executions = list(reader.records(kind="execution"))
    for execution_record in executions:
        edit_method = execution_record.get("edit_method")
        if not edit_method:
            continue
        model = str(execution_record["model"])
        plan_id = str(execution_record["plan_id"])
        execution_artifact_id = str(execution_record["artifact_id"])
        execution = reader.load(execution_artifact_id)
        target_layer = execution.get("summary", {}).get("target_layer")
        num_layers = int(execution.get("summary", {}).get("num_layers", 0))
        variants = _analysis_variants(execution, execution_record)
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"Execution artifact has no analysis variants: {execution_artifact_id}")

        for identifier in analysis_ids:
            spec = ANALYSES.get(identifier)
            resolved_configs: dict[str, dict[str, Any]] = {}
            for variant in variants:
                if not isinstance(variant, Mapping):
                    continue
                resolved = _resolve_analysis_config(
                    spec,
                    variant,
                    num_layers=num_layers,
                    defaults=defaults.get(identifier, {}),
                    overrides=overrides.get(identifier, {}),
                )
                resolved_configs.setdefault(config_hash(resolved), resolved)

            for digest, analysis_config in resolved_configs.items():
                artifact_id = analysis_id(
                    model,
                    plan_id,
                    str(edit_method),
                    spec.category,
                    identifier,
                    digest,
                )
                input_refs = [reader.ref(execution_artifact_id)]
                capture_ids: dict[str, tuple[str | None, str]] = {}
                missing: list[str] = []
                supported = supports_model(spec, model)
                if supported:
                    for capture_name in spec.required_captures:
                        capture_spec = CAPTURES.get(capture_name)
                        baseline_id = (
                            capture_id(model, plan_id, capture_name, None) if capture_spec.requires_baseline else None
                        )
                        method_id = capture_id(model, plan_id, capture_name, str(edit_method))
                        try:
                            method_ref = reader.ref(method_id)
                            baseline_ref = reader.ref(baseline_id) if baseline_id else None
                        except KeyError:
                            missing.append(capture_name)
                            continue
                        if baseline_ref is not None:
                            input_refs.append(baseline_ref)
                        input_refs.append(method_ref)
                        capture_ids[capture_name] = (baseline_id, method_id)

                unavailable_reason = None
                if not supported:
                    unavailable_reason = f"analysis {identifier} does not support model family for {model}"
                elif missing:
                    unavailable_reason = f"missing captures: {', '.join(sorted(missing))}"

                current = writer.current(
                    artifact_id,
                    expected_config_hash=digest,
                    inputs=input_refs,
                )
                if not force and current is not None:
                    skipped.append(artifact_id)
                    continue

                if unavailable_reason is not None:
                    cases = _unavailable_cases(execution, unavailable_reason)
                    status, summary, _ = _status_summary(cases, {})
                    error = unavailable_reason
                else:
                    capture_payloads = {
                        capture_name: (
                            materialize_capture(reader.load(baseline_id), reader.load(method_id))
                            if baseline_id is not None
                            else list(reader.load(method_id).get("cases", []))
                        )
                        for capture_name, (baseline_id, method_id) in capture_ids.items()
                    }
                    context = AnalysisContext(
                        run_root=root,
                        model=model,
                        plan_id=plan_id,
                        edit_method=str(edit_method),
                        target_layer=int(target_layer) if target_layer is not None else None,
                        execution=execution,
                        captures=capture_payloads,
                        config=analysis_config,
                    )
                    try:
                        result = spec.load()(context)
                        cases = list(result.get("cases", []))
                        status, summary, error = _status_summary(
                            cases,
                            dict(result.get("summary", {})),
                        )
                    except Exception as exc:
                        status = "error"
                        cases = []
                        summary = {
                            "cases_total": 0,
                            "cases_complete": 0,
                            "cases_unavailable": 0,
                            "cases_error": 1,
                        }
                        error = str(exc)

                payload = build_artifact(
                    artifact_id=artifact_id,
                    kind="analysis",
                    category=spec.category,
                    producer=identifier,
                    run_id=str(reader.manifest["run_id"]),
                    model=model,
                    plan_id=plan_id,
                    edit_method=str(edit_method),
                    status=status,
                    config=analysis_config,
                    config_hash=digest,
                    inputs=input_refs,
                    created_at=datetime.now().isoformat(),
                    cases=cases,
                    summary=summary,
                    error=error,
                )
                path = layout.analysis_path(
                    model,
                    plan_id,
                    str(edit_method),
                    spec.category,
                    identifier,
                    digest,
                )
                writer.write(path, payload, force=force)
                written.append(artifact_id)
                if status == "error":
                    failures.append(f"{artifact_id}: {error or 'analysis failed'}")

    if failures and not continue_on_error:
        raise AnalysisExecutionError("analysis failures: " + "; ".join(failures))

    return {
        "run_id": reader.manifest["run_id"],
        "written": written,
        "skipped": skipped,
        "errors": failures,
    }
