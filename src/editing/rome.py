"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Any

import torch

from src.editing.base import EditOutcome
from src.evaluation.rome import compute_rome_metrics


class RomeEditMethod:
    identifier = "rome"
    description = "Apply the repository's existing ROME implementation."

    def apply(self, handler: Any, case: dict[str, Any]) -> EditOutcome:
        from src.rome.optimization import gather_k, insert_kv, optimize_v
        from src.rome.prefixes import resolve_rome_sample_count

        fact = case["fact_tuple"]
        layer_name = handler._layer_name_template.format(handler._layer)
        old_weight = handler._get_module(layer_name).weight.detach().clone()
        try:
            probe_vector = gather_k(
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
            insert_kv(handler, probe_vector, delta)
            return EditOutcome(
                success=True,
                probe_vector=probe_vector,
                metadata={
                    "probe_norm": probe_vector.norm().item(),
                    "delta_norm": delta.norm().item(),
                },
                modified_weights={"proj": (int(handler._layer),)},
                restorations={layer_name: old_weight},
            )
        except Exception:
            handler._get_module(layer_name).weight = torch.nn.Parameter(old_weight)
            raise

    def evaluate(
        self,
        handler: Any,
        case: dict[str, Any],
        outcome: EditOutcome,
    ) -> dict[str, Any]:
        del outcome
        fact = case["fact_tuple"]
        prompt_text = fact[0].format(fact[1])
        target_new = case.get("target_new_str", fact[2].strip())
        target_true = case.get("target_true_str", fact[3].strip())
        return compute_rome_metrics(
            handler,
            prompt_text,
            target_new,
            target_true,
            paraphrase_prompts=case.get("paraphrase_prompts", []),
            neighborhood_prompts=case.get("neighborhood_prompts", []),
        )


__all__ = ["RomeEditMethod"]
