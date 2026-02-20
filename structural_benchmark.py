#!/usr/bin/env python3
"""
nice_benchmark.py - Simplified ROME + Structural Analysis Benchmark

Runs ROME edits and uses WeightMSDDetector/BlindMSDDetector to analyze weight changes.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

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


def to_serializable(obj):
    """Convert numpy/torch types to JSON-serializable Python types."""
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist() if hasattr(obj, 'tolist') else list(obj)
    elif isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def extract_all_weights(handler) -> Dict[int, torch.Tensor]:
    """Extract MLP projection weights from all layers."""
    return {
        idx: handler._get_module(handler._layer_name_template.format(idx)).weight.detach().clone()
        for idx in range(handler.num_of_layers)
    }


def extract_fc_weights(handler) -> Dict[int, torch.Tensor]:
    """Extract MLP feed-forward c_fc weights from all layers."""
    fc_template = handler._layer_name_template.replace("c_proj", "c_fc")
    return {
        idx: handler._get_module(fc_template.format(idx)).weight.detach().clone()
        for idx in range(handler.num_of_layers)
    }


def run_benchmark(
    model_name: str = "gpt2-large",
    n_tests: int = 10,
    start_idx: int = 0,
    output_dir: str = "./outputs",
    spectral_top_k: int = 50,
    trim_first_layers: int = 2,
    trim_last_layers: int = 2,
):
    """Run the complete benchmark."""
    
    # Config
    layer = 8 if "medium" in model_name else 12 if "large" in model_name else 5
    cfg = OmegaConf.create({
        "model": {
            "handler": "gpt2", "name": model_name, "models_dir": "./models",
            "second_moment_dir": "./second_moment_stats",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "save_to_local": True, "layer_name_template": "transformer.h.{}.mlp.c_proj",
            "layer": layer, "epochs": 25, "lr": 0.5, "kl_factor": 0.0625, "weight_decay": 0.5,
        }
    })
    
    LOGGER.info(f"Loading {model_name}...")
    handler = ModelHandler(cfg)
    
    # Get baseline weights
    original_weights = extract_all_weights(handler)
    fc_weights = extract_fc_weights(handler)

    spectral_top_k = max(1, int(spectral_top_k))
    trim_first_layers = max(0, int(trim_first_layers))
    trim_last_layers = max(0, int(trim_last_layers))

    spectral_detector = SpectralDetector(
        top_k=spectral_top_k,
        boundary=0,
        trim_first_layers=trim_first_layers,
        trim_last_layers=trim_last_layers,
    )
    LOGGER.info(
        "Spectral config: top_k=%s, boundary=%s, trim_first_layers=%s, trim_last_layers=%s",
        spectral_top_k,
        0,
        trim_first_layers,
        trim_last_layers,
    )
    
    # Load test cases
    ds = datasets.load_dataset("azhx/counterfact", split="train")
    test_cases = []
    for i, item in enumerate(ds):
        if i < start_idx: continue
        if len(test_cases) >= n_tests: break
        rw = item["requested_rewrite"]
        test_cases.append({
            "case_id": item.get("case_id", i), "subject": rw["subject"],
            "fact_tuple": (rw["prompt"], rw["subject"], " " + rw["target_new"]["str"], " " + rw["target_true"]["str"]),
        })
    
    blind_detector = BlindMSDDetector()
    baseline_spectral = spectral_detector.detect(original_weights, fc_weights=fc_weights)

    results = {
        "metadata": {
            "model": model_name,
            "target_layer": handler._layer,
            "n_tests": len(test_cases),
            "timestamp": datetime.now().isoformat(),
            "spectral_config": {
                "top_k": spectral_top_k,
                "boundary": 0,
                "trim_first_layers": trim_first_layers,
                "trim_last_layers": trim_last_layers,
            },
        },
        "baseline_blind": to_serializable(blind_detector.detect(original_weights)),
        "baseline_spectral": to_serializable(baseline_spectral),
        "baseline_interlayer": to_serializable(collect_all_interlayer_data(original_weights)),
        "tests": [],
    }

    successes = {"rome": 0, "normal_detection": 0, "blind_detection": 0, "spectral_detection": 0}
    
    for i, case in enumerate(test_cases):
        LOGGER.info(f"[{i+1}/{len(test_cases)}] {case['subject']}")
        
        layer_name = handler._layer_name_template.format(handler._layer)
        old_W = handler._get_module(layer_name).weight.detach().clone()
        
        test_entry = {"case_id": case["case_id"], "subject": case["subject"], "error": None}
        
        try:
            # ROME edit
            fact = case["fact_tuple"]
            k = gather_k(handler, fact_tuple=fact, N=50)
            delta = optimize_v(handler, fact_tuple=fact, N_prompts=50, N_optim_steps=handler.epochs)
            new_W, old_W_backup = insert_kv(handler, k, delta)
            prompt = fact[0].format(fact[1])
            tokens = handler.tokenize_prompt(prompt)
            with torch.no_grad():
                out = handler.model(**tokens)
            predicted = handler.tokenizer.decode(out.logits[0, -1, :].argmax())
            success = predicted.strip().lower() == fact[2].strip().lower()
            
            test_entry["rome"] = {"success": success, "predicted": predicted, "k_norm": k.norm().item(), "delta_norm": delta.norm().item()}
            if success: successes["rome"] += 1
            
            # Structural detection
            modified_weights = {idx: w.clone() for idx, w in original_weights.items()}
            modified_weights[handler._layer] = new_W.detach()
            
            normal_result = WeightMSDDetector(original_weights).detect(modified_weights)
            blind_result = blind_detector.detect(modified_weights)
            spectral_result = spectral_detector.detect(modified_weights, fc_weights=fc_weights)
            
            normal_correct = normal_result.get("anomalous_layer") == handler._layer
            blind_correct = blind_result.get("anomalous_layer") == handler._layer
            spectral_correct = spectral_result.get("anomalous_layer") == handler._layer
            if normal_correct: successes["normal_detection"] += 1
            if blind_correct: successes["blind_detection"] += 1
            if spectral_correct: successes["spectral_detection"] += 1
            
            test_entry.update({
                "normal_detection": to_serializable(normal_result),
                "blind_detection": to_serializable(blind_result),
                "spectral_detection": to_serializable(spectral_result),
                "interlayer": to_serializable(collect_all_interlayer_data(modified_weights)),
                "accuracy": {
                    "rome_success": success,
                    "normal_correct": normal_correct,
                    "blind_correct": blind_correct,
                    "spectral_correct": spectral_correct,
                },
            })
            LOGGER.info(
                f"  ROME: {'OK' if success else 'FAIL'}, "
                f"Normal: layer {normal_result.get('anomalous_layer')}, "
                f"Blind: layer {blind_result.get('anomalous_layer')}, "
                f"Spectral: layer {spectral_result.get('anomalous_layer')}"
            )
            
        except Exception as e:
            test_entry["error"] = str(e)
            test_entry["skipped"] = True
            LOGGER.warning(f"  SKIPPED - Edit failed: {e}")
        finally:
            handler._get_module(layer_name).weight = torch.nn.Parameter(old_W)
            results["tests"].append(test_entry)
            # Clean up memory after each test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Only count successful (non-skipped) tests
    successful_tests = [t for t in results["tests"] if not t.get("skipped", False)]
    n = len(successful_tests)
    results["summary"] = {
        "total_tests": len(test_cases),
        "successful_tests": n,
        "skipped_tests": len(test_cases) - n,
        "rome_success_rate": successes["rome"] / n if n else 0,
        "normal_detection_accuracy": successes["normal_detection"] / n if n else 0,
        "blind_detection_accuracy": successes["blind_detection"] / n if n else 0,
        "spectral_detection_accuracy": successes["spectral_detection"] / n if n else 0,
    }
    
    LOGGER.info(
        f"Summary: ROME {successes['rome']}/{n}, "
        f"Normal {successes['normal_detection']}/{n}, "
        f"Blind {successes['blind_detection']}/{n}, "
        f"Spectral {successes['spectral_detection']}/{n} "
        f"(skipped {len(test_cases) - n})"
    )

    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"rome_structural_{model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
    
    with open(output_file, "w") as f:
        json.dump(to_serializable(results), f, indent=2)
    
    LOGGER.info(f"Saved to: {output_file}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-large")
    parser.add_argument("--n-tests", type=int, default=3)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--output-dir", default="./analysis_out")
    parser.add_argument("--spectral-top-k", type=int, default=50)
    parser.add_argument("--trim-first-layers", type=int, default=2)
    parser.add_argument("--trim-last-layers", type=int, default=2)
    args = parser.parse_args()

    run_benchmark(
        args.model,
        args.n_tests,
        args.start_idx,
        args.output_dir,
        args.spectral_top_k,
        args.trim_first_layers,
        args.trim_last_layers,
    )
