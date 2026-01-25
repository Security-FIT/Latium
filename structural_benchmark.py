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
from typing import Dict, List

import datasets
import numpy as np
import torch
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from reimagined.handlers.common import get_handler
from reimagined.rome.weight_intervention.common import compute_k, compute_v, insert_kv
from reimagined.structural.detector import WeightMSDDetector
from reimagined.structural.blind_detector import BlindMSDDetector


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


def run_benchmark(model_name: str = "gpt2-medium", n_tests: int = 5, start_idx: int = 0, output_dir: str = "./outputs"):
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
    handler = get_handler(cfg)
    
    # Get baseline weights
    original_weights = extract_all_weights(handler)
    
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
    
    results = {
        "metadata": {"model": model_name, "target_layer": handler._layer, "n_tests": len(test_cases), "timestamp": datetime.now().isoformat()},
        "baseline_blind": to_serializable(BlindMSDDetector().detect(original_weights)),
        "tests": [],
    }
    
    successes = {"rome": 0, "normal_detection": 0, "blind_detection": 0}
    
    for i, case in enumerate(test_cases):
        LOGGER.info(f"[{i+1}/{len(test_cases)}] {case['subject']}")
        
        layer_name = handler._layer_name_template.format(handler._layer)
        old_W = handler._get_module(layer_name).weight.detach().clone()
        
        test_entry = {"case_id": case["case_id"], "subject": case["subject"], "error": None}
        
        try:
            # ROME edit
            fact = case["fact_tuple"]
            k = compute_k(handler, fact_tuple=fact, N=50)
            k_init = compute_k(handler, fact_tuple=fact, N=50)
            v, delta, v_init = compute_v(handler, k, fact, N_prompts=50, N_optim_steps=handler.epochs, epsilon=0.005)
            new_W = insert_kv(handler, k, v, delta, k_init, v_init)
            
            # Test prediction
            handler._get_module(layer_name).weight = torch.nn.Parameter(new_W)
            prompt = fact[0].format(fact[1])
            tokens = handler.tokenize_prompt(prompt)
            with torch.no_grad():
                out = handler.model(**tokens)
            predicted = handler.tokenizer.decode(out.logits[0, -1, :].argmax())
            success = predicted.strip().lower() == fact[2].strip().lower()
            
            test_entry["rome"] = {"success": success, "predicted": predicted, "k_norm": k.norm().item(), "v_norm": v.norm().item(), "delta_norm": delta.norm().item()}
            if success: successes["rome"] += 1
            
            # Structural detection
            modified_weights = {idx: w.clone() for idx, w in original_weights.items()}
            modified_weights[handler._layer] = new_W
            
            normal_result = WeightMSDDetector(original_weights).detect(modified_weights)
            blind_result = BlindMSDDetector().detect(modified_weights)
            
            normal_correct = normal_result.get("anomalous_layer") == handler._layer
            blind_correct = blind_result.get("anomalous_layer") == handler._layer
            if normal_correct: successes["normal_detection"] += 1
            if blind_correct: successes["blind_detection"] += 1
            
            test_entry.update({
                "normal_detection": to_serializable(normal_result),
                "blind_detection": to_serializable(blind_result),
                "accuracy": {"rome_success": success, "normal_correct": normal_correct, "blind_correct": blind_correct},
            })
            LOGGER.info(f"  ROME: {'OK' if success else 'FAIL'}, Normal: layer {normal_result.get('anomalous_layer')}, Blind: layer {blind_result.get('anomalous_layer')}")
            
        except Exception as e:
            test_entry["error"] = str(e)
            LOGGER.error(f"  Error: {e}")
        finally:
            handler._get_module(layer_name).weight = torch.nn.Parameter(old_W)
            results["tests"].append(test_entry)
    
    n = len(test_cases)
    results["summary"] = {
        "rome_success_rate": successes["rome"] / n if n else 0,
        "normal_detection_accuracy": successes["normal_detection"] / n if n else 0,
        "blind_detection_accuracy": successes["blind_detection"] / n if n else 0,
    }
    
    LOGGER.info(f"Summary: ROME {successes['rome']}/{n}, Normal {successes['normal_detection']}/{n}, Blind {successes['blind_detection']}/{n}")
    
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
    parser.add_argument("--model", default="gpt2-medium")
    parser.add_argument("--n-tests", type=int, default=5)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--output-dir", default="./outputs")
    args = parser.parse_args()
    
    run_benchmark(args.model, args.n_tests, args.start_idx, args.output_dir)
