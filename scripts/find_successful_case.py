#!/usr/bin/env python3
"""Hunt for a CounterFact case where ROME succeeds on the given model.

Writes a 1-case manifest JSON compatible with ``structural_benchmark.py
--case-index-file``.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import traceback
from pathlib import Path

import torch
from omegaconf import OmegaConf

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.counterfact_selection import build_case_manifest, load_cases_by_range
from src.handlers.rome import ModelHandler
from src.model_config import load_model_config
from src.rome.common import gather_k, insert_kv, optimize_v
from structural_benchmark import compute_rome_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def _load_model_cfg(args: argparse.Namespace):
    if args.model:
        return load_model_config(args.model)
    return OmegaConf.load(args.config)


def main() -> None:
    parser = argparse.ArgumentParser()
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", help="Model key, fleet key, or HuggingFace model id")
    model_group.add_argument("--config", help="Backward-compatible path to a YAML model config")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--n-prompts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.create({
        "model": _load_model_cfg(args),
        "generation": OmegaConf.load(_REPO / "src/config/generation/generation.yaml"),
        "dataset_sm": OmegaConf.load(_REPO / "src/config/dataset_sm/wikitext.yaml"),
    })

    LOGGER.info("Loading %s ...", cfg.model.name)
    handler = ModelHandler(cfg)
    LOGGER.info("Loaded. layer=%d", handler._layer)

    cases = load_cases_by_range(n_tests=args.max_cases, start_idx=args.start_idx)
    rng = random.Random(args.seed)
    rng.shuffle(cases)

    layer_name = handler._layer_name_template.format(handler._layer)
    found = False

    for attempt, case in enumerate(cases, start=1):
        fact = case["fact_tuple"]
        LOGGER.info("Attempt %d/%d: %s", attempt, len(cases), case["subject"])
        old_weight = None
        try:
            old_weight = handler._get_module(layer_name).weight.detach().clone()

            k = gather_k(handler, fact_tuple=fact, N=args.n_prompts)
            delta = optimize_v(
                handler,
                fact_tuple=fact,
                N_prompts=args.n_prompts,
                N_optim_steps=handler.epochs,
            )
            insert_kv(handler, k, delta)

            metrics = compute_rome_metrics(
                handler,
                fact[0].format(fact[1]),
                case.get("target_new_str", fact[2].strip()),
                case.get("target_true_str", fact[3].strip()),
                paraphrase_prompts=case.get("paraphrase_prompts", []),
                neighborhood_prompts=case.get("neighborhood_prompts", []),
            )

            handler._get_module(layer_name).weight = torch.nn.Parameter(old_weight)
            handler.remove_hooks()

            efficacy = metrics.get("efficacy_score", 0.0)
            LOGGER.info("  Result: ES=%.3f S=%.3f", efficacy, metrics.get("overall_score", 0.0))
            if efficacy >= 1.0:
                LOGGER.info("SUCCESS on case %s (dataset_index=%d)", case["subject"], case["dataset_index"])
                manifest = build_case_manifest([case["dataset_index"]])
                Path(args.output_manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                print(args.output_manifest)
                found = True
                break
        except Exception as exc:
            LOGGER.warning("  ERROR on attempt %d: %s", attempt, exc)
            if "LATIUM_LOG_SKIP_TRACEBACK" not in __import__("os").environ:
                LOGGER.debug(traceback.format_exc())
            try:
                handler.remove_hooks()
                if old_weight is not None:
                    handler._get_module(layer_name).weight = torch.nn.Parameter(old_weight)
            except Exception:
                pass

    del handler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not found:
        LOGGER.error("No successful case found after %d attempts", len(cases))
        sys.exit(1)


if __name__ == "__main__":
    main()
