from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from src.graphs.runtime import render_run
from src.results import ArtifactWriter, build_artifact, config_hash
from src.structural.detectors.weighted_spectrum import PROFILE_FIELDS


def _write_artifact(
    writer: ArtifactWriter,
    path: Path,
    *,
    artifact_id: str,
    kind: str,
    producer: str,
    cases: list[dict],
    category: str | None = None,
    config: dict | None = None,
) -> None:
    resolved_config = dict(config or {})
    writer.write(
        path,
        build_artifact(
            artifact_id=artifact_id,
            kind=kind,
            category=category,
            producer=producer,
            run_id="run",
            model="architecture-neutral-model",
            plan_id="cases0-0_r01",
            edit_method="rome",
            status="complete",
            config=resolved_config,
            config_hash=config_hash(resolved_config),
            inputs=[],
            created_at=datetime.now().isoformat(),
            cases=cases,
            summary={
                "target_layer": 4,
                "num_layers": 8,
                "cases_total": len(cases),
                "cases_complete": len(cases),
            },
        ),
    )


def _profiles() -> dict[str, dict[str, float]]:
    profiles: dict[str, dict[str, float]] = {}
    for layer in range(8):
        profile = {field: 0.1 + 0.01 * layer for field in PROFILE_FIELDS}
        profile.update(
            {
                "rank1_energy": 0.7,
                "rank2_energy": 0.9,
                "relative_subspace_rank1_energy": 0.8,
                "bilateral_coherence": 0.3,
                "bilateral_alignment": 0.2,
                "bilateral_balance": 0.6,
            }
        )
        profiles[str(layer)] = profile
    profiles["4"].update(
        {
            "relative_subspace_frobenius": 4.0,
            "relative_subspace_operator_norm": 3.5,
            "relative_operator_norm": 3.0,
            "bilateral_coherence": 1.0,
            "bilateral_alignment": 1.0,
            "bilateral_balance": 1.0,
        }
    )
    return profiles


def _analysis_case(data: dict) -> dict:
    return {
        "case_id": "case",
        "status": "complete",
        "data": data,
        "error": None,
        "accuracy": {
            "target_layer": 4,
            "detected_layer": data.get("anomalous_layer", 4),
            "correct": data.get("anomalous_layer", 4) == 4,
        },
    }


def _write_detector_inputs(tmp_path: Path, *, omit: str | None = None) -> None:
    writer = ArtifactWriter(tmp_path, run_id="run")
    _write_artifact(
        writer,
        tmp_path / "execution.json",
        artifact_id="execution/rome",
        kind="execution",
        producer="rome",
        cases=[{"case_id": "case", "status": "complete", "edit": {"success": True}, "error": None}],
    )
    profiles = _profiles()
    layer_scores = {
        str(layer): profiles[str(layer)]["relative_subspace_frobenius"]
        for layer in range(8)
    }
    localizer = {
        "anomalous_layer": 4,
        "detection_score": 4.0,
        "margin": 3.5,
        "score_field": "relative_subspace_frobenius",
        "layer_scores": layer_scores,
        "profiles": profiles,
        "config": {"trim_first": 1, "trim_last": 1},
        "evaluated_layers": list(range(1, 7)),
        "excluded_layers": [0, 7],
    }
    evidence = {
        "median": 0.1,
        "mad_scale": 0.02,
        "effective_mad_scale": 0.02,
        "robust_z": 8.0,
        "universal_threshold": 2.0,
        "evidence_ratio": 4.0,
    }
    analyses = {
        "weighted-spectrum": ("detection", localizer),
        "rome-presence-blind-peak": (
            "attribution",
            {
                "anomalous_layer": 4,
                "detection_score": 4.0,
                "verdict": "rome_like",
                "is_rome_like": True,
                "layer_evidence": {str(layer): 0.1 + (2.0 if layer == 4 else 0.0) for layer in range(1, 7)},
                "evidence": evidence,
            },
        ),
        "rome-presence-blind-footprint": (
            "attribution",
            {
                "anomalous_layer": 4,
                "detection_score": 3.0,
                "verdict": "rome_like",
                "is_rome_like": True,
                "layer_evidence": {str(layer): 0.05 + (1.5 if layer == 4 else 0.0) for layer in range(1, 7)},
                "evidence": evidence,
            },
        ),
        "rome-presence-delta": (
            "attribution",
            {
                "anomalous_layer": 4,
                "detection_score": 20.0,
                "verdict": "rome_like",
                "is_rome_like": True,
                "evidence": {
                    "rank1_residual": 1e-7,
                    "roundoff_bound": 2e-6,
                },
            },
        ),
    }
    for producer, (category, data) in analyses.items():
        if producer == omit:
            continue
        config = {} if producer == "rome-presence-delta" else {"trim_first": 1, "trim_last": 1}
        _write_artifact(
            writer,
            tmp_path / f"{producer}.json",
            artifact_id=f"analysis/{producer}",
            kind="analysis",
            category=category,
            producer=producer,
            config=config,
            cases=[_analysis_case(data)],
        )


def test_rome_detector_renderer_exports_every_layer_stat_and_decision(tmp_path: Path) -> None:
    _write_detector_inputs(tmp_path)

    result = render_run(
        tmp_path,
        preset="none",
        enabled=("rome-detector-explainer",),
        renderer_options={
            "rome-detector-explainer": {
                "formats": ["png", "json", "csv"],
            }
        },
    )

    assert result["written"] == ["render/rome-detector-explainer"]
    root = tmp_path / "graphs" / "rome-detector-explainer"
    artifact = json.loads((root / "artifact.json").read_text(encoding="utf-8"))
    assert artifact["status"] == "complete"
    assert len(artifact["inputs"]) == 5

    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    assert index["profile_fields"] == list(PROFILE_FIELDS)
    assert index["cases"][0]["verdicts"] == {
        "blind_peak": "rome_like",
        "blind_footprint": "rome_like",
        "delta": "rome_like",
    }
    case_root = root / index["cases"][0]["outputs"][0]
    assert case_root.is_file()
    assert any(output.endswith("detector-overview.png") for output in index["cases"][0]["outputs"])
    assert any(output.endswith("all-layer-statistics.png") for output in index["cases"][0]["outputs"])
    assert any(output.endswith("detector-data.json") for output in index["cases"][0]["outputs"])
    assert (root / "detector-outcomes.png").is_file()

    with (root / "detector-layer-statistics.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert set(PROFILE_FIELDS).issubset(rows[0])
    assert rows[4]["target"] == "True"
    assert rows[4]["detected"] == "True"


def test_rome_detector_renderer_declares_missing_presence_analysis(tmp_path: Path) -> None:
    _write_detector_inputs(tmp_path, omit="rome-presence-delta")

    render_run(tmp_path, preset="none", enabled=("rome-detector-explainer",))

    artifact = json.loads(
        (tmp_path / "graphs" / "rome-detector-explainer" / "artifact.json").read_text(encoding="utf-8")
    )
    assert artifact["status"] == "unavailable"
    assert "analysis:rome-presence-delta" in artifact["error"]
