#!/usr/bin/env python3
"""
Hunt for a CounterFact case where ROME succeeds on the given model.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>

Writes a one-case manifest JSON for ``python -m src structural run
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
from src.evaluation.rome import compute_rome_metrics
from src.common.model_config import load_model_config
from src.rome.common import gather_k, insert_kv, optimize_v, resolve_rome_sample_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)


def main() -> None:
    script_cfg = OmegaConf.load(_REPO / "src/config/find_successful_case/default.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model key, fleet key, or HuggingFace model id")
    parser.add_argument("--start-idx", type=int, default=int(script_cfg.start_idx))
    parser.add_argument("--max-cases", type=int, default=int(script_cfg.max_cases))
    parser.add_argument("--seed", type=int, default=int(script_cfg.seed))
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()

    cfg = OmegaConf.create(
        {
            "model": load_model_config(args.model),
            "generation": OmegaConf.load(_REPO / "src/config/generation/generation.yaml"),
            "dataset_facts": OmegaConf.load(_REPO / "src/config/dataset_facts/counterfact.yaml"),
            "dataset_sm": OmegaConf.load(_REPO / "src/config/dataset_sm/wikitext.yaml"),
            "runtime": OmegaConf.load(_REPO / "src/config/runtime/default.yaml"),
        }
    )

    LOGGER.info("Loading %s ...", cfg.model.name)
    handler = ModelHandler(cfg)
    LOGGER.info("Loaded. layer=%d", handler._layer)

    cases = load_cases_by_range(
        n_tests=args.max_cases,
        dataset_name=str(cfg.dataset_facts.name),
        split=str(cfg.dataset_facts.split),
        start_idx=args.start_idx,
    )
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

            k = gather_k(
                handler,
                fact_tuple=fact,
                N=resolve_rome_sample_count(handler, "k_N"),
            )
            delta = optimize_v(
                handler,
                fact_tuple=fact,
                N_prompts=resolve_rome_sample_count(handler, "v_N"),
                N_optim_steps=handler.epochs,
            )
            if delta is None:
                raise RuntimeError("optimize_v returned None")
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
                manifest = build_case_manifest(
                    [case["dataset_index"]],
                    dataset_name=str(cfg.dataset_facts.name),
                    split=str(cfg.dataset_facts.split),
                )
                Path(args.output_manifest).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                print(args.output_manifest)
                found = True
                break
        except Exception as exc:
            LOGGER.warning("  ERROR on attempt %d: %s", attempt, exc)
            if not bool(cfg.runtime.log_skip_traceback):
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
