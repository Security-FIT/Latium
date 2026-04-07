#!/usr/bin/env python3
"""
metric_benchmark.py — Evaluate new matrix metrics for ROME detection.

Runs ROME edits, computes per-layer metrics (baseline vs edited),
and reports which metrics best identify the edited layer.

Usage:
    python metric_benchmark.py --models gpt2-large --n-tests 3
    python metric_benchmark.py --models gpt2-large gpt2-xl qwen3-4b --n-tests 3
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import datasets
import numpy as np
import torch
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from src.handlers.rome import ModelHandler
from src.rome.common import gather_k, optimize_v, insert_kv
from src.structural.matrix_metrics import (
    compute_all_layers,
    curvature_profile,
    differential_curvature,
    mad_outlier_score,
    find_anomalous_layer,
)

CONFIG_DIR = Path(__file__).parent / "src" / "config"
MODEL_CONFIG_DIR = CONFIG_DIR / "model"

# Maps proj (output) layer key -> fc (input) layer key across architectures
FC_TEMPLATE_MAP = {
    "c_proj": "c_fc",
    "fc_out": "fc_in",
    "down_proj": "up_proj",
}

# All metric names to evaluate
METRIC_NAMES = [
    # SVD-free
    "row_norm_cv", "row_norm_skew", "row_norm_kurtosis", "row_norm_max_over_mean",
    "col_norm_cv", "col_norm_skew", "col_norm_kurtosis",
    "schatten4_concentration",
    "entry_kurtosis", "entry_skew",
    "gram_cos_mean", "gram_cos_std", "gram_cos_max",
    # SVD-based
    "stable_rank", "relative_gap", "log_gap_1_2",
    "top5_concentration", "sv_decay_rate", "numerical_rank_01",
    "sv_entropy_top20", "log_condition",
]


def to_serializable(obj):
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def get_fc_template(layer_name_template: str) -> str:
    for proj_key, fc_key in FC_TEMPLATE_MAP.items():
        if proj_key in layer_name_template:
            return layer_name_template.replace(proj_key, fc_key)
    raise ValueError(f"Cannot derive fc layer template from: {layer_name_template}")


def load_model_config(model_name: str) -> OmegaConf:
    yaml_path = MODEL_CONFIG_DIR / f"{model_name}.yaml"
    if yaml_path.exists():
        return OmegaConf.load(yaml_path)
    for path in sorted(MODEL_CONFIG_DIR.glob("*.yaml")):
        if path.name == "boilerplate.yaml":
            continue
        cfg = OmegaConf.load(path)
        if getattr(cfg, "name", "") == model_name:
            return cfg
    available = ", ".join(
        p.stem for p in sorted(MODEL_CONFIG_DIR.glob("*.yaml")) if p.name != "boilerplate.yaml"
    )
    raise FileNotFoundError(f"No config for '{model_name}'. Available: {available}")


def build_cfg(model_name: str) -> OmegaConf:
    model_cfg = load_model_config(model_name)
    return OmegaConf.create({
        "model": model_cfg,
        "generation": OmegaConf.load(CONFIG_DIR / "generation" / "generation.yaml"),
        "dataset_sm": OmegaConf.load(CONFIG_DIR / "dataset_sm" / "wikitext.yaml"),
    })


def extract_weights(handler: ModelHandler, template: str) -> Dict[int, torch.Tensor]:
    return {
        idx: handler._get_module(template.format(idx)).weight.detach().clone().cpu()
        for idx in range(handler.num_of_layers)
    }


def load_test_cases(n_tests: int, start_idx: int = 0) -> List[dict]:
    ds = datasets.load_dataset("azhx/counterfact", split="train")
    cases = []
    for i, item in enumerate(ds):
        if i < start_idx:
            continue
        if len(cases) >= n_tests:
            break
        rw = item["requested_rewrite"]
        cases.append({
            "case_id": item.get("case_id", i),
            "subject": rw["subject"],
            "fact_tuple": (
                rw["prompt"], rw["subject"],
                " " + rw["target_new"]["str"],
                " " + rw["target_true"]["str"],
            ),
        })
    return cases


def evaluate_metrics(
    proj_metrics: Dict[int, Dict[str, float]],
    fc_metrics: Dict[int, Dict[str, float]],
    target_layer: int,
    trim_first: int = 2,
    trim_last: int = 2,
) -> Dict[str, dict]:
    """Evaluate each metric's ability to find the target layer."""
    results = {}
    has_fc = fc_metrics is not None and len(fc_metrics) > 0

    for mname in METRIC_NAMES:
        # Skip metrics that might not be computed
        sample = next(iter(proj_metrics.values()))
        if mname not in sample:
            continue

        # Raw curvature detection
        layers, vals, curv = curvature_profile(proj_metrics, mname, trim_first, trim_last)
        if len(layers) == 0:
            continue

        detected, score = find_anomalous_layer(layers, curv, method="mad", boundary=1)
        correct = detected == target_layer

        entry = {
            "detected_layer": detected,
            "score": round(score, 4),
            "correct": correct,
            "method": "curvature_mad",
        }

        # Differential curvature (proj - fc)
        if has_fc:
            diff_layers, diff_curv = differential_curvature(
                proj_metrics, fc_metrics, mname, trim_first, trim_last,
            )
            if len(diff_layers) > 0:
                diff_det, diff_score = find_anomalous_layer(
                    diff_layers, diff_curv, method="mad", boundary=1,
                )
                entry["diff_detected_layer"] = diff_det
                entry["diff_score"] = round(diff_score, 4)
                entry["diff_correct"] = diff_det == target_layer

        # Z-score detection (for comparison)
        detected_z, score_z = find_anomalous_layer(layers, curv, method="z", boundary=1)
        entry["z_detected"] = detected_z
        entry["z_score"] = round(score_z, 4)
        entry["z_correct"] = detected_z == target_layer

        results[mname] = entry

    return results


def run_single_model(
    model_name: str,
    test_cases: List[dict],
    n_prompts: int = 10,
    trim_first: int = 2,
    trim_last: int = 2,
) -> dict:
    """Run metric evaluation for one model."""
    cfg = build_cfg(model_name)
    LOG.info("Loading %s ...", cfg.model.name)
    handler = ModelHandler(cfg)
    LOG.info("Loaded. layer=%d, layers=%d", handler._layer, handler.num_of_layers)

    proj_template = handler._layer_name_template
    try:
        fc_template = get_fc_template(proj_template)
    except ValueError:
        fc_template = None

    # Extract baseline weights
    original_proj = extract_weights(handler, proj_template)
    original_fc = None
    if fc_template:
        try:
            original_fc = extract_weights(handler, fc_template)
        except (KeyError, ValueError):
            LOG.warning("Could not extract fc weights")

    # Compute baseline metrics
    LOG.info("Computing baseline metrics...")
    baseline_proj_metrics = compute_all_layers(original_proj, device="cuda")
    baseline_fc_metrics = None
    if original_fc:
        baseline_fc_metrics = compute_all_layers(original_fc, device="cuda")

    results = {
        "metadata": {
            "model": cfg.model.name,
            "target_layer": handler._layer,
            "n_layers": handler.num_of_layers,
            "n_tests": len(test_cases),
            "timestamp": datetime.now().isoformat(),
        },
        "baseline_metrics": to_serializable(baseline_proj_metrics),
        "tests": [],
    }

    # Per-metric accuracy counters
    metric_correct = {}
    metric_diff_correct = {}

    for i, case in enumerate(test_cases):
        LOG.info("[%d/%d] %s", i + 1, len(test_cases), case["subject"])
        layer_name = proj_template.format(handler._layer)
        old_W = handler._get_module(layer_name).weight.detach().clone()
        entry = {"case_id": case["case_id"], "subject": case["subject"], "error": None}

        try:
            fact = case["fact_tuple"]

            # ROME edit
            k = gather_k(handler, fact_tuple=fact, N=n_prompts)
            delta = optimize_v(
                handler, fact_tuple=fact,
                N_prompts=n_prompts, N_optim_steps=handler.epochs,
            )
            new_W, _, _ = insert_kv(handler, k, delta)

            # Check success
            prompt = fact[0].format(fact[1])
            tokens = handler.tokenize_prompt(prompt)
            with torch.no_grad():
                out = handler.model(**tokens)
            predicted = handler.tokenizer.decode(out.logits[0, -1, :].argmax())
            rome_ok = predicted.strip().lower() == fact[2].strip().lower()
            entry["rome_success"] = rome_ok

            # Build modified weight dict
            modified_proj = {idx: w.clone() for idx, w in original_proj.items()}
            modified_proj[handler._layer] = new_W.detach().cpu()

            # Compute metrics on modified weights
            LOG.info("  Computing modified metrics...")
            mod_proj_metrics = compute_all_layers(modified_proj, device="cuda")

            # Evaluate each metric
            eval_results = evaluate_metrics(
                mod_proj_metrics, baseline_fc_metrics,
                handler._layer, trim_first, trim_last,
            )

            entry["metric_results"] = eval_results

            # Track accuracy
            for mname, mres in eval_results.items():
                metric_correct.setdefault(mname, 0)
                if mres.get("correct"):
                    metric_correct[mname] += 1
                if mres.get("diff_correct"):
                    metric_diff_correct.setdefault(mname, 0)
                    metric_diff_correct[mname] += 1

            # Log key results
            top_metrics = sorted(
                eval_results.items(),
                key=lambda x: x[1].get("score", 0),
                reverse=True,
            )[:5]
            for mname, mres in top_metrics:
                LOG.info(
                    "  %s: L%s (%.2f) %s | diff: L%s",
                    mname, mres["detected_layer"], mres["score"],
                    "✓" if mres.get("correct") else "✗",
                    mres.get("diff_detected_layer", "N/A"),
                )

        except Exception as e:
            entry["error"] = str(e)
            LOG.warning("  SKIPPED: %s", e)
            import traceback
            traceback.print_exc()
        finally:
            handler.remove_hooks()
            handler._get_module(layer_name).weight = torch.nn.Parameter(old_W)
            results["tests"].append(entry)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    n_ok = sum(1 for t in results["tests"] if not t.get("error"))
    LOG.info("\n" + "=" * 60)
    LOG.info("METRIC ACCURACY SUMMARY (%s, %d tests)", cfg.model.name, n_ok)
    LOG.info("=" * 60)

    summary = {}
    for mname in sorted(set(list(metric_correct.keys()) + list(metric_diff_correct.keys()))):
        c = metric_correct.get(mname, 0)
        dc = metric_diff_correct.get(mname, 0)
        summary[mname] = {
            "curvature_correct": c,
            "curvature_rate": c / n_ok if n_ok else 0,
            "diff_correct": dc,
            "diff_rate": dc / n_ok if n_ok else 0,
        }
        LOG.info(
            "  %-28s curv=%d/%d (%.0f%%) diff=%d/%d (%.0f%%)",
            mname, c, n_ok, 100 * c / n_ok if n_ok else 0,
            dc, n_ok, 100 * dc / n_ok if n_ok else 0,
        )

    results["summary"] = summary

    del handler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Metric Benchmark for ROME Detection")
    parser.add_argument("--models", "--model", nargs="+", default=["gpt2-large"])
    parser.add_argument("--n-tests", type=int, default=3)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--output-dir", default="./analysis_out")
    parser.add_argument("--trim-first", type=int, default=2)
    parser.add_argument("--trim-last", type=int, default=2)
    args = parser.parse_args()

    test_cases = load_test_cases(args.n_tests, args.start_idx)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        LOG.info("=" * 60)
        LOG.info("Metric Benchmark: %s", model_name)
        LOG.info("=" * 60)

        model_results = run_single_model(
            model_name, test_cases, args.n_prompts,
            trim_first=args.trim_first, trim_last=args.trim_last,
        )

        safe_name = model_name.replace("/", "_").replace("\\", "_")
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_file = output_path / f"metric_bench_{safe_name}_{ts}.json"
        with open(out_file, "w") as f:
            json.dump(to_serializable(model_results), f, indent=2)
        LOG.info("Saved: %s", out_file)
