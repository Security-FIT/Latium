#!/usr/bin/env python3
"""
structural_benchmark.py - ROME + Structural Analysis Benchmark

Runs ROME edits on one or more models and evaluates structural detectors
on the modified weights.  All model parameters (layer, template, hyperparams)
are read from per-model YAML configs in src/config/model/.

Usage examples:
    # Single model (uses src/config/model/gpt2-large.yaml):
    python structural_benchmark.py --models gpt2-large

    # Multiple models in one run:
    python structural_benchmark.py --models gpt2-large qwen3-4b mistral-7b-v0.1
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
LOGGER = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from src.handlers.rome import ModelHandler
from src.rome.common import gather_k, optimize_v, insert_kv
from src.structural.detector import WeightMSDDetector
from src.structural.blind_detector import BlindMSDDetector
from src.structural.spectral_detector import SpectralDetector
from src.structural.interlayer import collect_all_interlayer_data
from src.structural.ipr import (
    layer_ipr_summary,
    layer_fc_proj_ipr_discrepancy,
    IPRDetector,
)

CONFIG_DIR = Path(__file__).parent / "src" / "config"
MODEL_CONFIG_DIR = CONFIG_DIR / "model"

SPECTRAL_SIGNAL_KEYS = [
    "sv_z_scores", "sv_ratio_scores",
    "sv_z_rolling_z_scores", "sv_ratio_rolling_z_scores",
    "pcs_composite_rank_scores", "sv_pcs_contradiction_scores",
    "rome_hybrid_scores",
    "pcs_neighbor_shift_scores", "pcs_neighbor_var_scores",
    "pcs_neighbor_min_shift_scores", "pcs_neighbor_flip_fraction_scores",
    "pcs_next_shift_scores", "pcs_next_jump_scores", "pcs_next_curvature_scores",
]

# Maps proj (output) layer key -> fc (input) layer key across architectures
FC_TEMPLATE_MAP = {
    "c_proj": "c_fc",        # GPT-2
    "fc_out": "fc_in",       # GPT-J
    "down_proj": "up_proj",  # Llama / Mistral / Qwen
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_serializable(obj):
    """Convert numpy/torch types to JSON-serializable Python types."""
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
    """Derive the fc (input) layer template from the proj (output) layer template."""
    for proj_key, fc_key in FC_TEMPLATE_MAP.items():
        if proj_key in layer_name_template:
            return layer_name_template.replace(proj_key, fc_key)
    raise ValueError(f"Cannot derive fc layer template from: {layer_name_template}")


def load_model_config(model_name: str) -> OmegaConf:
    """Load a model config YAML by stem name or by the 'name' field inside the YAML."""
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
    """Compose a full runtime config from per-model YAML + shared dataset/generation defaults."""
    model_cfg = load_model_config(model_name)
    return OmegaConf.create({
        "model": model_cfg,
        "generation": OmegaConf.load(CONFIG_DIR / "generation" / "generation.yaml"),
        "dataset_sm": OmegaConf.load(CONFIG_DIR / "dataset_sm" / "wikitext.yaml"),
    })


def extract_weights(handler: ModelHandler, template: str) -> Dict[int, torch.Tensor]:
    """Extract weights from all layers using the given name template, moved to CPU."""
    return {
        idx: handler._get_module(template.format(idx)).weight.detach().clone().cpu()
        for idx in range(handler.num_of_layers)
    }


def add_ipr_z_scores(summary: Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    """Add within-distribution z-scores for key IPR metrics."""
    if not summary:
        return summary
    layers = sorted(summary.keys())
    for metric in ("global_ipr", "row_ipr_mean", "row_ipr_std"):
        values = np.array([summary[idx][metric] for idx in layers])
        mean, std = values.mean(), values.std() + 1e-12
        for idx in layers:
            summary[idx][f"{metric}_z"] = float((summary[idx][metric] - mean) / std)
    return summary


def per_layer_delta(baseline, modified, fields):
    """Compute per-layer delta for the given field names."""
    common = sorted(set(baseline) & set(modified))
    return {
        idx: {f"{f}_delta": modified[idx][f] - baseline[idx][f] for f in fields}
        for idx in common
    }


def spectral_signal_delta(baseline_block, modified_block):
    """Per-signal, per-layer delta between two spectral detector outputs."""
    deltas = {}
    for key in SPECTRAL_SIGNAL_KEYS:
        base = {int(k): float(v) for k, v in (baseline_block.get(key) or {}).items()}
        mod = {int(k): float(v) for k, v in (modified_block.get(key) or {}).items()}
        common = sorted(set(base) & set(mod))
        if common:
            deltas[key] = {l: mod[l] - base[l] for l in common}
    return deltas


def load_test_cases(n_tests: int, start_idx: int = 0) -> List[dict]:
    """Load test cases from the CounterFact dataset."""
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


# ---------------------------------------------------------------------------
# Single-model benchmark
# ---------------------------------------------------------------------------

def run_single_model(
    model_name: str,
    test_cases: List[dict],
    n_prompts: int,
    spectral_top_k: int = 50,
    trim_first: int = 2,
    trim_last: int = 2,
    spectral_neighbor_layers: int = 1,
) -> dict:
    """Run the full ROME + structural-detection benchmark for one model."""
    cfg = build_cfg(model_name)
    LOGGER.info("Loading %s ...", cfg.model.name)
    handler = ModelHandler(cfg)
    LOGGER.info(
        "Loaded. layer=%d, emb=%d, hidden=%d, prompts=%d",
        handler._layer, handler.emb_shape, handler.hidden_dim, n_prompts,
    )

    proj_template = handler._layer_name_template
    fc_template = get_fc_template(proj_template)

    # Baseline weights (CPU to free GPU for ROME edits)
    original_proj = extract_weights(handler, proj_template)
    try:
        original_fc = extract_weights(handler, fc_template)
        has_fc = True
    except (KeyError, ValueError):
        LOGGER.warning("Could not extract fc weights (%s) - IPR fc-vs-proj disabled", fc_template)
        original_fc = None
        has_fc = False

    # Detectors
    blind_detector = BlindMSDDetector()
    spectral_detector = SpectralDetector(
        top_k=spectral_top_k, boundary=2,
        trim_first_layers=trim_first, trim_last_layers=trim_last,
        neighbor_layers=spectral_neighbor_layers,
    )
    ipr_detector = IPRDetector(trim_first=trim_first, trim_last=trim_last) if has_fc else None

    # Baselines
    baseline_spectral = spectral_detector.detect(original_proj, fc_weights=original_fc)
    baseline_ipr_proj = add_ipr_z_scores(layer_ipr_summary(original_proj))
    baseline_ipr_fc = add_ipr_z_scores(layer_ipr_summary(original_fc)) if has_fc else {}
    baseline_ipr_disc = layer_fc_proj_ipr_discrepancy(original_proj, original_fc) if has_fc else {}

    results = {
        "metadata": {
            "model": cfg.model.name,
            "target_layer": handler._layer,
            "n_tests": len(test_cases),
            "n_prompts": n_prompts,
            "timestamp": datetime.now().isoformat(),
            "spectral_config": {
                "top_k": spectral_top_k, "boundary": 2,
                "trim_first": trim_first, "trim_last": trim_last,
                "neighbor_layers": spectral_neighbor_layers,
                "signal_keys": SPECTRAL_SIGNAL_KEYS,
            },
        },
        "baseline_blind": to_serializable(blind_detector.detect(original_proj)),
        "baseline_spectral": to_serializable(baseline_spectral),
        "baseline_interlayer": to_serializable(collect_all_interlayer_data(original_proj)),
        "baseline_ipr": {
            "proj": to_serializable(baseline_ipr_proj),
            "fc": to_serializable(baseline_ipr_fc),
            "fc_vs_proj": to_serializable(baseline_ipr_disc),
        },
        "tests": [],
    }

    counts = {k: 0 for k in ("rome", "normal", "blind", "spectral", "ipr")}

    for i, case in enumerate(test_cases):
        LOGGER.info("[%d/%d] %s", i + 1, len(test_cases), case["subject"])
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

            # Check ROME success
            prompt = fact[0].format(fact[1])
            tokens = handler.tokenize_prompt(prompt)
            with torch.no_grad():
                out = handler.model(**tokens)
            predicted = handler.tokenizer.decode(out.logits[0, -1, :].argmax())
            rome_ok = predicted.strip().lower() == fact[2].strip().lower()

            entry["rome"] = {
                "success": rome_ok, "predicted": predicted,
                "k_norm": k.norm().item(), "delta_norm": delta.norm().item(),
            }
            if rome_ok:
                counts["rome"] += 1

            # Build modified weight dict (only proj layer changed by ROME)
            modified_proj = {idx: w.clone() for idx, w in original_proj.items()}
            modified_proj[handler._layer] = new_W.detach().cpu()

            # Run detectors
            normal_res = WeightMSDDetector(original_proj).detect(modified_proj)
            blind_res = blind_detector.detect(modified_proj)
            spectral_res = spectral_detector.detect(modified_proj, fc_weights=original_fc)
            ipr_res = ipr_detector.detect(modified_proj, original_fc) if ipr_detector else {}

            # IPR analysis
            mod_ipr_proj = add_ipr_z_scores(layer_ipr_summary(modified_proj))
            mod_ipr_disc = layer_fc_proj_ipr_discrepancy(modified_proj, original_fc) if has_fc else {}

            correct = {
                "normal": normal_res.get("anomalous_layer") == handler._layer,
                "blind": blind_res.get("anomalous_layer") == handler._layer,
                "spectral": spectral_res.get("anomalous_layer") == handler._layer,
                "ipr": ipr_res.get("anomalous_layer") == handler._layer if ipr_res else False,
            }
            for name, ok in correct.items():
                if ok:
                    counts[name] += 1

            entry.update({
                "normal_detection": to_serializable(normal_res),
                "blind_detection": to_serializable(blind_res),
                "spectral_detection": to_serializable(spectral_res),
                "spectral_delta": to_serializable(spectral_signal_delta(baseline_spectral, spectral_res)),
                "interlayer": to_serializable(collect_all_interlayer_data(modified_proj)),
                "ipr": {
                    "proj": to_serializable(mod_ipr_proj),
                    "proj_delta": to_serializable(per_layer_delta(
                        baseline_ipr_proj, mod_ipr_proj,
                        ["global_ipr", "row_ipr_mean", "row_ipr_std"],
                    )),
                    "fc_vs_proj": to_serializable(mod_ipr_disc),
                    "fc_vs_proj_delta": to_serializable(per_layer_delta(
                        baseline_ipr_disc, mod_ipr_disc,
                        ["global_ipr_gap", "global_ipr_ratio_proj_over_fc",
                         "row_ipr_mean_gap", "row_ipr_std_gap", "row_ipr_median_gap"],
                    )) if has_fc else {},
                    "detection": to_serializable(ipr_res),
                },
                "accuracy": {
                    "rome_success": rome_ok,
                    **{f"{name}_correct": v for name, v in correct.items()},
                },
            })

            LOGGER.info(
                "  ROME=%s Normal=L%s Blind=L%s Spectral=L%s IPR=L%s(%.3f)",
                "OK" if rome_ok else "FAIL",
                normal_res.get("anomalous_layer"),
                blind_res.get("anomalous_layer"),
                spectral_res.get("anomalous_layer"),
                ipr_res.get("anomalous_layer", "N/A"),
                ipr_res.get("anomaly_score", 0),
            )

        except Exception as e:
            entry["error"] = str(e)
            entry["skipped"] = True
            LOGGER.warning("  SKIPPED: %s", e)
        finally:
            handler.remove_hooks()
            handler._get_module(layer_name).weight = torch.nn.Parameter(old_W)
            results["tests"].append(entry)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    ok_tests = [t for t in results["tests"] if not t.get("skipped")]
    n = len(ok_tests)
    results["summary"] = {
        "total": len(test_cases), "successful": n, "skipped": len(test_cases) - n,
        **{f"{k}_rate": counts[k] / n if n else 0 for k in counts},
    }
    LOGGER.info(
        "[%s] ROME=%d/%d Normal=%d/%d Blind=%d/%d Spectral=%d/%d IPR=%d/%d skip=%d",
        cfg.model.name, counts["rome"], n, counts["normal"], n, counts["blind"], n,
        counts["spectral"], n, counts["ipr"], n, len(test_cases) - n,
    )

    # Free GPU
    del handler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# Multi-model entry point
# ---------------------------------------------------------------------------

def run_benchmark(
    models: List[str],
    n_tests: int = 30,
    start_idx: int = 0,
    output_dir: str = "./analysis_out",
    n_prompts: int = 10,
    spectral_top_k: int = 50,
    trim_first: int = 2,
    trim_last: int = 2,
    spectral_neighbor_layers: int = 1,
):
    """Run the benchmark across one or more models, saving results per model."""
    test_cases = load_test_cases(n_tests, start_idx)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_name in models:
        LOGGER.info("=" * 60)
        LOGGER.info("Benchmark: %s", model_name)
        LOGGER.info("=" * 60)

        model_results = run_single_model(
            model_name, test_cases, n_prompts,
            spectral_top_k=spectral_top_k,
            trim_first=trim_first, trim_last=trim_last,
            spectral_neighbor_layers=spectral_neighbor_layers,
        )

        safe_name = model_name.replace("/", "_").replace("\\", "_")
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_file = output_path / f"rome_structural_{safe_name}_{ts}.json"
        with open(out_file, "w") as f:
            json.dump(to_serializable(model_results), f, indent=2)
        LOGGER.info("Saved: %s", out_file)

        all_results[model_name] = model_results

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ROME + Structural Analysis Benchmark")
    parser.add_argument("--models", "--model", nargs="+", default=["gpt2-large"],
                        help="Model config names (YAML stems or HF model names)")
    parser.add_argument("--n-tests", type=int, default=30)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--output-dir", default="./analysis_out")
    parser.add_argument("--n-prompts", type=int, default=10)
    parser.add_argument("--spectral-top-k", type=int, default=50)
    parser.add_argument("--trim-first", type=int, default=2)
    parser.add_argument("--trim-last", type=int, default=2)
    parser.add_argument("--spectral-neighbor-layers", type=int, default=1)
    args = parser.parse_args()

    run_benchmark(
        models=args.models,
        n_tests=args.n_tests,
        start_idx=args.start_idx,
        output_dir=args.output_dir,
        n_prompts=args.n_prompts,
        spectral_top_k=args.spectral_top_k,
        trim_first=args.trim_first,
        trim_last=args.trim_last,
        spectral_neighbor_layers=args.spectral_neighbor_layers,
    )
