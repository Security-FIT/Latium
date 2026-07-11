"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

from src.editing.base import EditOutcome
from src.results import ArtifactWriter, RunArtifactReader, RunLayout, build_artifact, config_hash
from src.results.ids import capture_id, execution_id
from src.structural.config import AnalysisVariantConfig, ModelRunPlan, StructuralBenchmarkConfig
from src.structural.capture.artifacts import capture_options, execution_config
from src.structural.detectors.weighted_spectrum import (
    FOOTPRINT_PROFILE_FIELDS,
    LOCALIZER_PROFILE_FIELDS,
)
from src.structural.execution.edit_execution import run_edit_method


class _FailingMethod:
    def apply(self, handler, case):
        raise RuntimeError("boom")

    def evaluate(self, handler, case, outcome):
        return {}


class _Tokenizer:
    def decode(self, token_ids) -> str:
        return str(token_ids)


class _Model:
    def __init__(self) -> None:
        self.head = torch.nn.Linear(2, 2, bias=False)

    def get_output_embeddings(self):
        return self.head


class _Handler:
    _layer = 0
    num_of_layers = 1

    def __init__(self) -> None:
        self.model = _Model()
        self.tokenizer = _Tokenizer()

    def remove_hooks(self) -> None:
        pass


class _CountingMethod:
    def __init__(self) -> None:
        self.count = 0

    def apply(self, handler, case):
        self.count += 1
        return EditOutcome(metrics={"apply_count": self.count})

    def evaluate(self, handler, case, outcome):
        return {}


def _plan() -> ModelRunPlan:
    return ModelRunPlan(
        model_key="gpt2-large",
        run_id="run",
        plan_id="cases0-0_r01",
        run_index=1,
        start_idx=0,
        end_idx=0,
    )


def test_execution_config_uses_model_inputs_not_analysis_windows() -> None:
    plan = _plan()
    case_selection = {
        "mode": "explicit_indices",
        "manifest_hash": "abc",
        "selected_case_ids": [10],
        "selected_dataset_indices": [3],
    }
    first = StructuralBenchmarkConfig(
        models=("gpt2-large",),
        seed=11,
        analysis_variants=(AnalysisVariantConfig(spectral_top_k=50, trim_first=1, trim_last=1),),
    )
    trim_changed = StructuralBenchmarkConfig(
        models=("gpt2-large",),
        seed=11,
        analysis_variants=(AnalysisVariantConfig(spectral_top_k=50, trim_first=2, trim_last=1),),
    )
    seed_changed = StructuralBenchmarkConfig(
        models=("gpt2-large",),
        seed=12,
        analysis_variants=first.analysis_variants,
    )

    base_config = execution_config(
        first,
        plan,
        "rome",
        model_context={"model_key": "gpt2-large", "target_layer": 17},
        case_selection=case_selection,
        options=capture_options(first),
    )
    trim_config = execution_config(
        trim_changed,
        plan,
        "rome",
        model_context={"model_key": "gpt2-large", "target_layer": 17},
        case_selection=case_selection,
        options=capture_options(trim_changed),
    )
    seed_config = execution_config(
        seed_changed,
        plan,
        "rome",
        model_context={"model_key": "gpt2-large", "target_layer": 17},
        case_selection=case_selection,
        options=capture_options(seed_changed),
    )
    manifest_changed = execution_config(
        first,
        plan,
        "rome",
        model_context={"model_key": "gpt2-large", "target_layer": 17},
        case_selection={**case_selection, "manifest_hash": "def"},
        options=capture_options(first),
    )

    assert "analysis_variants" not in base_config
    assert config_hash(base_config) == config_hash(trim_config)
    assert config_hash(base_config) != config_hash(seed_config)
    assert config_hash(base_config) != config_hash(manifest_changed)


def test_capture_options_request_only_fields_consumed_by_selected_analyses() -> None:
    detection = capture_options(
        StructuralBenchmarkConfig(
            capture_profile="detection",
            analysis_preset="detection",
        )
    )
    presence = capture_options(
        StructuralBenchmarkConfig(
            capture_profile="rome-presence",
            analysis_preset="rome-presence",
        )
    )
    capture_only_presence = capture_options(
        StructuralBenchmarkConfig(
            capture_profile="rome-presence",
            analysis_preset="none",
            run_analysis=False,
        )
    )

    assert detection["weighted_spectrum_fields"] == LOCALIZER_PROFILE_FIELDS
    assert presence["weighted_spectrum_fields"] == FOOTPRINT_PROFILE_FIELDS
    assert capture_only_presence["weighted_spectrum_fields"] == FOOTPRINT_PROFILE_FIELDS


def _write_baseline_capture(
    writer: ArtifactWriter,
    layout: RunLayout,
    plan: ModelRunPlan,
) -> dict:
    model = "gpt2-large"
    config = {"capture": "matrix-features"}
    return writer.write(
        layout.capture_path(model, plan.plan_id, "matrix-features", edit_method=None),
        build_artifact(
            artifact_id=capture_id(model, plan.plan_id, "matrix-features", None),
            kind="capture",
            producer="matrix-features",
            run_id=plan.run_id,
            model=model,
            plan_id=plan.plan_id,
            edit_method=None,
            status="complete",
            config=config,
            config_hash=config_hash(config),
            inputs=[],
            created_at="now",
            cases=[{"case_id": "baseline", "status": "complete", "data": {}}],
            summary={},
        ),
    )


def test_missing_method_capture_preserves_complete_execution(
    tmp_path: Path,
) -> None:
    method = _CountingMethod()

    writer = ArtifactWriter(tmp_path, run_id="run")
    layout = RunLayout(tmp_path)
    plan = _plan()
    config = StructuralBenchmarkConfig(models=("gpt2-large",))
    baseline = _write_baseline_capture(writer, layout, plan)

    common_kwargs = dict(
        writer=writer,
        layout=layout,
        config=config,
        plan=plan,
        model="gpt2-large",
        handler=_Handler(),
        test_cases=[{"case_id": "case"}],
        edit_method_name="rome",
        capture_names=("matrix-features",),
        options=capture_options(config),
        baseline_records={"matrix-features": baseline},
        baseline_proj={0: torch.eye(2)},
        baseline_fc=None,
        baseline_attention={},
        proj_template="layer.{}",
        fc_template=None,
        case_selection={"mode": "contiguous_slice", "selected_case_ids": ["case"]},
        model_context={"model_key": "gpt2-large", "target_layer": 0},
        method_loader=lambda _: method,
        capture_case_fn=lambda _name, _context, *, case_id: {
            "case_id": case_id,
            "status": "complete",
            "data": {"captured": method.count},
            "error": None,
        },
    )

    run_edit_method(**common_kwargs)
    reader = RunArtifactReader(tmp_path)
    execution_artifact_id = execution_id("gpt2-large", plan.plan_id, "rome")
    capture_artifact_id = capture_id("gpt2-large", plan.plan_id, "matrix-features", "rome")
    first_execution_hash = reader.ref(execution_artifact_id)["content_hash"]
    capture_record = reader.manifest["artifacts"][capture_artifact_id]
    (tmp_path / capture_record["path"]).unlink()

    run_edit_method(**common_kwargs)
    reader = RunArtifactReader(tmp_path)
    execution_ref = reader.ref(execution_artifact_id)
    capture_ref = reader.ref(capture_artifact_id)
    capture_record = reader.manifest["artifacts"][capture_artifact_id]

    assert execution_ref["content_hash"] == first_execution_hash
    assert capture_record["inputs"][0] == execution_ref
    assert capture_ref["content_hash"]


@pytest.mark.parametrize(
    ("skip_traceback", "expect_traceback"),
    ((True, False), (False, True)),
)
def test_log_skip_traceback_matches_name(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    skip_traceback: bool,
    expect_traceback: bool,
) -> None:
    caplog.set_level(logging.WARNING, logger="src.structural.execution.edit_execution")

    run_edit_method(
        writer=ArtifactWriter(tmp_path, run_id="run"),
        layout=RunLayout(tmp_path),
        config=StructuralBenchmarkConfig(
            models=("gpt2-large",),
            log_skip_traceback=skip_traceback,
        ),
        plan=_plan(),
        model="gpt2-large",
        handler=_Handler(),
        test_cases=[{"case_id": "case"}],
        edit_method_name="rome",
        capture_names=(),
        options={},
        baseline_records={},
        baseline_proj={0: torch.eye(2)},
        baseline_fc=None,
        baseline_attention={},
        proj_template="layer.{}",
        fc_template=None,
        method_loader=lambda _: _FailingMethod(),
        traceback_formatter=lambda: "TRACEBACK",
    )

    assert ("TRACEBACK" in caplog.text) is expect_traceback
