"""Registry adapter with mode-aware scoring and shared ROME metric aggregation."""

from __future__ import annotations

from typing import Any

import torch

from src.editing.base import EditOutcome
from src.editing.rome import RomeEditMethod


class RomeT4EditMethod(RomeEditMethod):
    identifier = "rome_t4"
    description = "Memory-aware ROME with exact micro-batched k/v steps."

    def evaluate(self, handler: Any, case: dict[str, Any], outcome: EditOutcome) -> dict[str, Any]:
        from src.editing.rome_t4.probing import score_target
        from src.editing.rome_t4.prompts import is_chat_mode
        from src.evaluation.rome import compute_rome_metrics

        if not is_chat_mode(handler):
            return super().evaluate(handler, case, outcome)

        def mean_nll(prompt: str, target: str) -> float:
            count = len(handler.tokenizer.encode(target.strip(), add_special_tokens=False))
            return -score_target(handler, prompt, target) / count if count else float("inf")

        fact = case["fact_tuple"]
        return compute_rome_metrics(
            handler, fact[0].format(fact[1]),
            case.get("target_new_str", fact[2].strip()),
            case.get("target_true_str", fact[3].strip()),
            paraphrase_prompts=case.get("paraphrase_prompts", []),
            neighborhood_prompts=case.get("neighborhood_prompts", []),
            target_nll=mean_nll,
        )

    def apply(self, handler: Any, case: dict[str, Any]) -> EditOutcome:
        from src.editing.rome_t4.common import gather_k, insert_kv, optimize_v
        from src.rome.common import resolve_rome_sample_count

        fact = case["fact_tuple"]
        layer_name = handler._layer_name_template.format(handler._layer)
        # One owned snapshot covers both insertion modes, including in-place.
        old_weight = handler._get_module(layer_name).weight.detach().clone()
        try:
            handler.remove_hooks()
            probe_vector = gather_k(
                handler, fact_tuple=fact, N=resolve_rome_sample_count(handler, "k_N"),
            )
            delta = optimize_v(
                handler, fact_tuple=fact,
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
                    "delta_norm": delta.detach().norm().item(),
                },
                modified_weights={"proj": (int(handler._layer),)},
                restorations={layer_name: old_weight},
            )
        except Exception:
            handler._get_module(layer_name).weight = torch.nn.Parameter(old_weight, requires_grad=False)
            raise
        finally:
            handler.remove_hooks()
