#!/usr/bin/env python3
"""Compare ROME layers on facts disjoint from tracing, then test one winner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import causal_rome_pipeline as pipeline


def candidate_layers(selected: int, configured: int, num_layers: int) -> list[int]:
    """Always compare the trace choice, adjacent layers, and the model default."""
    return sorted({layer for layer in (selected - 1, selected, selected + 1, configured) if 0 <= layer < num_layers})


def _ensure_covariance(model: str, layer: int, target_samples: int) -> list[Path]:
    existing = pipeline.find_second_moment_files(model, layer)
    if not existing:
        pipeline._run_command(
            pipeline.build_second_moment_command(model=model, layer=layer, target_samples=target_samples)
        )
        existing = pipeline.find_second_moment_files(model, layer)
    if not existing:
        raise FileNotFoundError(f"No second moment for {model} layer {layer}")
    return existing


def _run_rome(model: str, layer: int, n_cases: int, start_idx: int, output_dir: Path) -> tuple[Path, dict[str, Any]]:
    pipeline._run_command(
        pipeline.build_rome_command(
            model=model, layer=layer, n_tests=n_cases, start_idx=start_idx, output_dir=output_dir
        )
    )
    report_path, summary = pipeline._rome_state(output_dir, model=model, selected_layer=layer)
    if int(summary["n_evaluated"]) != n_cases:
        raise RuntimeError(
            f"Layer {layer} evaluated {summary['n_evaluated']} of {n_cases} cases; "
            "comparison requires identical case coverage"
        )
    return report_path, summary


def validate_layers(
    *,
    model: str,
    trace_summary_path: Path,
    output_root: Path,
    validation_cases: int = 30,
    test_cases: int = 30,
    covariance_samples: int = 100_000,
) -> Path:
    """Select by ROME validation, report final metrics only on separate test rows."""
    if min(validation_cases, test_cases, covariance_samples) <= 0:
        raise ValueError("Case and covariance counts must be positive")
    trace = json.loads(trace_summary_path.read_text(encoding="utf-8"))
    model_config = pipeline._model_config(model)
    if str(trace.get("model")) != str(model_config.name):
        raise ValueError("Trace summary belongs to a different model")
    selected = trace.get("selected_trace_center")
    if (
        selected is None
        or not trace.get("confirmation_passed")
        or not trace.get("selected_layer_directly_tested")
        or int(trace.get("window_size", 0)) != 1
    ):
        raise ValueError("Layer validation requires a confirmed single-layer trace")
    lower_bound = trace.get("confirmation_ci_lower")
    if lower_bound is None or float(lower_bound) <= 0:
        raise ValueError("Layer validation requires a positive held-out interval")
    num_layers = len(trace.get("trace_mlp_output_modules", []))
    if num_layers <= 0 or not 0 <= int(selected) < num_layers:
        raise ValueError("Trace summary has no valid layer map")
    trace_end = int(trace.get("num_dataset_examples_scanned", 0))
    if trace_end <= 0:
        raise ValueError("Trace summary has no dataset scan boundary")
    candidates = candidate_layers(int(selected), int(model_config.layer), num_layers)
    output_root.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for layer in candidates:
        covariance = _ensure_covariance(model, layer, covariance_samples)
        report, summary = _run_rome(
            model, layer, validation_cases, trace_end, output_root / "validation" / f"layer-{layer}"
        )
        comparisons.append(
            {
                "layer": layer,
                "covariance_files": [str(path) for path in covariance],
                "rome_report": str(report),
                "metrics": summary,
            }
        )
    winner = max(
        comparisons,
        key=lambda item: (
            float(item["metrics"]["mean_overall_score"]),
            float(item["metrics"].get("mean_neighborhood_score", 0)),
            -int(item["layer"]),
        ),
    )
    test_start = trace_end + validation_cases
    test_report, test_summary = _run_rome(model, int(winner["layer"]), test_cases, test_start, output_root / "test")
    report = {
        "schema": "latium.rome_layer_validation.v1",
        "model": model,
        "trace_summary": str(trace_summary_path),
        "trace_selected_layer": int(selected),
        "configured_layer": int(model_config.layer),
        "validation_case_range": [trace_end, test_start],
        "test_case_range": [test_start, test_start + test_cases],
        "candidate_layers": candidates,
        "validation_results": comparisons,
        "best_rome_layer": int(winner["layer"]),
        "test_report": str(test_report),
        "test_metrics": test_summary,
    }
    path = output_root / "layer-validation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-cases", type=int, default=30)
    parser.add_argument("--test-cases", type=int, default=30)
    parser.add_argument("--covariance-samples", type=int, default=100_000)
    args = parser.parse_args(argv)
    print(
        validate_layers(
            model=args.model,
            trace_summary_path=args.trace_summary,
            output_root=args.output_root,
            validation_cases=args.validation_cases,
            test_cases=args.test_cases,
            covariance_samples=args.covariance_samples,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
