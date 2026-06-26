"""
Manual edit command handler.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import json
from typing import Any

from omegaconf import DictConfig

from src.common.config import string_list as _string_list
from src.common.io import to_serializable


def _selected_edit_method(cfg: DictConfig) -> str:
    selected = cfg.edit_method
    if isinstance(selected, str):
        return selected
    return str(selected.identifier)


def _space_prefixed(value: Any) -> str:
    text = str(value or "")
    return text if not text or text.startswith(" ") else " " + text


def run_edit(cfg: DictConfig) -> int:
    import torch

    from src.editing.registry import get_edit_method
    from src.handlers.rome import ModelHandler

    fact_cfg = cfg.edit.fact
    fact_tuple = (
        str(fact_cfg.prompt),
        str(fact_cfg.subject),
        _space_prefixed(fact_cfg.target_new),
        _space_prefixed(fact_cfg.target_true),
    )
    case = {
        "case_id": "manual",
        "subject": fact_tuple[1],
        "target_new_str": fact_tuple[2].strip(),
        "target_true_str": fact_tuple[3].strip(),
        "fact_tuple": fact_tuple,
        "paraphrase_prompts": [],
        "neighborhood_prompts": [],
    }
    handler = ModelHandler(cfg)
    method_name = _selected_edit_method(cfg)
    method = get_edit_method(method_name)
    outcome = None
    try:
        outcome = method.apply(handler, case)
        metrics = method.evaluate(handler, case, outcome)
        outcome.metrics.update(metrics)
        print(
            json.dumps(
                to_serializable(
                    {
                        "edit_method": method_name,
                        "success": outcome.success,
                        "metrics": outcome.metrics,
                        "metadata": outcome.metadata,
                    }
                ),
                indent=2,
            )
        )

        prompt = handler.tokenize_prompt(fact_tuple[0].format(fact_tuple[1]))
        target_token_count = int(handler.tokenize_prompt(fact_tuple[2]).input_ids.shape[1])
        outputs = handler.model.generate(
            **prompt,
            max_length=prompt.input_ids.shape[1] + target_token_count,
        )
        print(handler.tokenizer.batch_decode(outputs))

        generation_cfg = cfg.edit.generation
        for probe_template in _string_list(cfg.edit.probe_prompts):
            probe = handler.tokenize_prompt(probe_template.format(fact_tuple[1]))
            outputs = handler.model.generate(
                **probe,
                max_new_tokens=int(generation_cfg.max_new_tokens),
                do_sample=bool(generation_cfg.do_sample),
                temperature=float(generation_cfg.temperature),
                top_k=int(generation_cfg.top_k),
                min_p=float(generation_cfg.min_p),
            )
            print(handler.tokenizer.batch_decode(outputs))
    finally:
        handler.remove_hooks()
        if outcome is not None:
            for module_name, old_weight in outcome.restorations.items():
                handler._get_module(module_name).weight = torch.nn.Parameter(old_weight)
    return 0


__all__ = ["run_edit"]
