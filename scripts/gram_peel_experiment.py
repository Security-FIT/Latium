"""Blind recovery of ROME edits from weights with GRAM peeling.

For every edited checkpoint the script:

1. runs GRAM on the down projections and peels the anomalous direction(s)
   until the profile is normal again (``src.structural.detectors.gram_peel``);
2. compares each peeled rank-one piece ``v k_hat^T`` with the true update
   (the unedited weights are known in this experiment);
3. decodes the peel without using the edit request:
   * key side: ranks every CounterFact subject and every vocabulary token by
     how strongly the peeled update responds to its MLP key (``key . k_hat``);
     the strongest response also fixes the sign of ``(v, k_hat)``;
   * value side: projects ``v`` through the final norm and the unembedding
     and reports the rank of the new target token.

Usage (inside the repository, with the Latium environment):

    python scripts/gram_peel_experiment.py --model gpt2-xl --facts 20 \
        --case-index-file manifests/counterfact_seed42_n1000.json \
        --out analysis_out/gram-peel/gpt2-xl.json

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json  # noqa: E402
from src.common.loading import load_dataset  # noqa: E402
from src.editing.registry import get_edit_method  # noqa: E402
from src.evaluation.rome import _get_target_token_ids  # noqa: E402
from src.handlers.rome import ModelHandler  # noqa: E402
from src.structural.detectors import gram_peel  # noqa: E402
from src.structural.execution.case_selection import load_test_cases  # noqa: E402
from src.structural.execution.edit_execution import restore  # noqa: E402
from src.structural.execution.weights import build_cfg  # noqa: E402


LOGGER = logging.getLogger("gram_peel_experiment")
FINAL_NORMS = ("transformer.ln_f", "model.norm", "model.decoder.final_layer_norm", "model.language_model.norm",
               "transformer.norm_f", "model.final_layernorm")
KEY_PREFIX = "Here is a fact about"
PROBE_TEMPLATES = ("{}", "{} is", "{} was", "{} is a", "{} is located in", "The language of {} is", "{} works as a",
                   "{} plays", "{} is a citizen of", "{} was born in", "{} was created by", "The official language of {} is")


class _Stop(Exception):
    pass


# --------------------------------------------------------------------------- model helpers


def final_norm_weight(handler: ModelHandler) -> Optional[torch.Tensor]:
    for name in FINAL_NORMS:
        try:
            module = handler._get_module(name)
        except KeyError:
            continue
        weight = getattr(module, "weight", None)
        if weight is not None:
            return weight.detach().float()
    return None


def unembedding(handler: ModelHandler) -> torch.Tensor:
    head = handler.model.get_output_embeddings()
    return head.weight.detach()


def collect_keys(
    handler: ModelHandler,
    layers: Sequence[int],
    sequences: list[list[int]],
    *,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    """Input of each layer's down projection at the last token of every sequence (fp16, GPU)."""
    template = str(handler._layer_name_template)
    modules = {layer: handler._get_module(template.format(layer)) for layer in layers}
    deepest = max(layers)
    device = next(handler.model.parameters()).device
    pad = handler.tokenizer.pad_token_id if handler.tokenizer.pad_token_id is not None else 0
    outputs: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        width = max(len(sequence) for sequence in chunk)
        ids = torch.full((len(chunk), width), pad, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, sequence in enumerate(chunk):  # right padding: last real token is len - 1
            ids[row, : len(sequence)] = torch.tensor(sequence)
            mask[row, : len(sequence)] = 1
        last = torch.tensor([len(sequence) - 1 for sequence in chunk], device=device)
        captured: dict[int, torch.Tensor] = {}

        def hook_for(layer: int):
            def hook(_module, args):
                hidden = args[0] if isinstance(args, tuple) else args
                if hidden.dim() == 2:
                    hidden = hidden.view(len(chunk), width, -1)
                captured[layer] = hidden[torch.arange(len(chunk), device=hidden.device), last.to(hidden.device)].half()
                if layer == deepest:
                    raise _Stop
                return None

            return hook

        handles = [module.register_forward_pre_hook(hook_for(layer)) for layer, module in modules.items()]
        try:
            with torch.no_grad():
                handler.model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
        except _Stop:
            pass
        finally:
            for handle in handles:
                handle.remove()
        for layer in layers:
            outputs[layer].append(captured[layer])
    return {layer: torch.cat(parts, dim=0) for layer, parts in outputs.items()}


def subject_sequences(handler: ModelHandler, subjects: Sequence[str]) -> tuple[list[list[int]], list[list[int]]]:
    tokenizer = handler.tokenizer
    bare = [tokenizer(subject)["input_ids"] for subject in subjects]
    prefixed = [tokenizer(f"{KEY_PREFIX} {subject}")["input_ids"] for subject in subjects]
    return bare, prefixed


def vocabulary_sequences(handler: ModelHandler) -> list[list[int]]:
    prefix = handler.tokenizer(KEY_PREFIX)["input_ids"]
    return [list(prefix) + [token] for token in range(int(unembedding(handler).shape[0]))]


# --------------------------------------------------------------------------- decoding


def rank_of(scores: torch.Tensor, indices: Sequence[int]) -> list[int]:
    order = torch.argsort(scores, descending=True)
    position = torch.empty_like(order)
    position[order] = torch.arange(order.numel(), device=order.device)
    return [int(position[index].item()) + 1 for index in indices]


def decode_key(key: torch.Tensor, subject_keys: torch.Tensor, vocab_keys: Optional[torch.Tensor]) -> tuple[float, torch.Tensor, Optional[torch.Tensor]]:
    """Return the blind sign and the signed response of every subject and vocabulary token."""
    k = key.to(device=subject_keys.device, dtype=torch.float32)
    subject_scores = subject_keys.float() @ k
    sign = 1.0 if float(subject_scores.max()) >= float(-subject_scores.min()) else -1.0
    vocab_scores = None if vocab_keys is None else sign * (vocab_keys.float() @ k)
    return sign, sign * subject_scores, vocab_scores


def decode_value(direction: torch.Tensor, sign: float, unembed: torch.Tensor, norm: Optional[torch.Tensor]) -> torch.Tensor:
    v = sign * direction.to(device=unembed.device, dtype=torch.float32)
    if norm is not None:
        v = v * norm.to(v.device)
    return unembed.float() @ v


def probe_logprobs(handler: ModelHandler, subject: str) -> torch.Tensor:
    """Mean next-token log-probabilities over generic subject prompts."""
    tokenizer = handler.tokenizer
    prompts = [template.format(subject) for template in PROBE_TEMPLATES]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    device = next(handler.model.parameters()).device
    mask = encoded["attention_mask"]
    last = torch.tensor([int(torch.nonzero(row).max()) for row in mask])
    with torch.no_grad():
        logits = handler.model(input_ids=encoded["input_ids"].to(device), attention_mask=mask.to(device),
                               use_cache=False).logits
    rows = logits[torch.arange(len(prompts)), last.to(logits.device)].float()
    return torch.log_softmax(rows, dim=-1).mean(dim=0)


def behavioural_value(
    handler: ModelHandler,
    template: str,
    edited: dict[int, torch.Tensor],
    peeled: dict[int, torch.Tensor],
    subject: str,
) -> torch.Tensor:
    """Log-probability gain of every token caused by the peeled component (edited minus peeled model)."""
    padding_side = handler.tokenizer.padding_side
    handler.tokenizer.padding_side = "left"
    modules = {layer: handler._get_module(template.format(layer)) for layer in edited}
    originals = {layer: module.weight for layer, module in modules.items()}
    try:
        for layer, module in modules.items():
            module.weight = torch.nn.Parameter(edited[layer].to(originals[layer]), requires_grad=False)
        with_edit = probe_logprobs(handler, subject)
        for layer, module in modules.items():
            module.weight = torch.nn.Parameter(peeled[layer].to(originals[layer]), requires_grad=False)
        without_edit = probe_logprobs(handler, subject)
    finally:
        handler.tokenizer.padding_side = padding_side
        for layer, module in modules.items():
            module.weight = originals[layer]
    return with_edit - without_edit


def target_token(tokenizer: Any, text: str) -> int:
    ids = _get_target_token_ids(tokenizer, text)
    return int(ids[0])


# --------------------------------------------------------------------------- ground truth


def true_update(baseline: torch.Tensor, edited: torch.Tensor, device: torch.device) -> torch.Tensor:
    before, _ = gram_peel.residual_matrix(baseline, device=device)
    after, _ = gram_peel.residual_matrix(edited, device=device)
    return after - before


def truth_svd(delta: torch.Tensor, rank: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(delta.float(), full_matrices=False)
    return u[:, :rank].double(), s[:rank].double(), vh[:rank].double()


def compare_with_truth(step: gram_peel.PeelStep, delta: torch.Tensor, svd) -> dict[str, float]:
    v = step.direction.to(delta.device)
    k = step.key.to(delta.device)
    estimate = torch.outer(v, k)
    norm = float(delta.norm())
    if norm == 0:
        return {}
    u, s, vh = svd
    sign = 1.0 if float((estimate * delta).sum()) >= 0 else -1.0
    return {
        "cos_value_top1": abs(float(v @ u[:, 0])),
        "cos_key_top1": abs(float(torch.nn.functional.cosine_similarity(k, vh[0], dim=0))),
        "value_in_update_span": float((u[:, : max(1, int((s > s[0] * 1e-3).sum()))].T @ v).norm()),
        "relative_error": float((sign * estimate - delta).norm() / norm),
        "true_rank1_share": float(s[0] ** 2 / norm**2),
        "estimate_norm_ratio": float(estimate.norm() / norm),
    }


def explained_energy(steps: Sequence[gram_peel.PeelStep], delta: torch.Tensor) -> float:
    if not steps or float(delta.norm()) == 0:
        return 0.0
    basis, _ = torch.linalg.qr(torch.stack([step.direction.to(delta.device) for step in steps], dim=1))
    return float((basis.T @ delta).square().sum() / delta.square().sum())


# --------------------------------------------------------------------------- experiment


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = build_cfg(args.model, runtime={"second_moment_allow_autocompute": False})
    handler = ModelHandler(cfg)
    device = gram_peel.compute_device()
    template = str(handler._layer_name_template)
    peel_options = {"grid": args.grid, "refine": args.refine}
    LOGGER.info("Model %s, %d layers, ROME layer %d", args.model, handler.num_of_layers, handler._layer)

    baseline = {layer: handler._get_module(template.format(layer)).weight.detach().clone() for layer in range(handler.num_of_layers)}
    started = time.time()
    base_state = gram_peel.GramState(baseline, device=device)
    base_grams = dict(base_state.grams)
    base_peel = gram_peel.peelability(base_state, **peel_options)
    base_peak = base_state.peak()
    LOGGER.info("Baseline GRAM peak %s, peelability in %.1fs", base_peak, time.time() - started)

    methods = {name: get_edit_method(name) for name in args.methods}
    key_layers = sorted({int(handler._layer)} | {layer for method in methods.values()
                                                 if hasattr(method, "target_layers")
                                                 for layer in method.target_layers(handler)})
    dataset = load_dataset(cfg)
    cases, _ = load_test_cases(args.facts, args.start, dataset_name=cfg.dataset_facts.name,
                               split=cfg.dataset_facts.split, case_index_file=args.case_index_file)
    subjects = sorted({str(row["requested_rewrite"]["subject"]) for row in dataset})
    if args.max_subjects:
        required = {str(case["fact_tuple"][1]) for case in cases}
        subjects = sorted(required | set(subjects[: args.max_subjects]))
    started = time.time()
    bare, prefixed = subject_sequences(handler, subjects)
    keys_bare = collect_keys(handler, key_layers, bare, batch_size=args.key_batch)
    keys_prefixed = collect_keys(handler, key_layers, prefixed, batch_size=args.key_batch)
    subject_keys = {layer: ((keys_bare[layer].float() + keys_prefixed[layer].float()) * 0.5).half() for layer in key_layers}
    del keys_bare, keys_prefixed
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    vocab_keys = (
        collect_keys(handler, key_layers, vocabulary_sequences(handler), batch_size=args.vocab_batch)
        if args.vocab_scan else {}
    )
    LOGGER.info("Keys for %d subjects%s at layers %s in %.1fs", len(subjects),
                " and the vocabulary" if args.vocab_scan else "", key_layers, time.time() - started)
    unembed = unembedding(handler)
    norm = final_norm_weight(handler)

    records: list[dict[str, Any]] = []
    payload = {
        "model": args.model,
        "rome_layer": int(handler._layer),
        "baseline": {
            "scores": {str(layer): value for layer, value in base_state.scores.items()},
            "peak": {"layer": base_peak[0], "z": base_peak[1]},
            "peelability": {str(layer): value for layer, value in base_peel.items()},
        },
        "subjects": len(subjects),
        "vocab_scan": bool(args.vocab_scan),
        "options": vars(args),
        "cases": records,
    }
    for name, method in methods.items():
        groups = method.group_cases(cases, handler) if hasattr(method, "group_cases") else cases
        for group in groups:
            members = list(group.get("requests") or [group])
            started = time.time()
            outcome = None
            try:
                outcome = method.apply(handler, group)
                metrics = method.evaluate(handler, group, outcome)
                changed = [int(layer) for layer in outcome.modified_weights.get("proj", ())]
                edited = {layer: handler._get_module(template.format(layer)).weight.detach().clone() for layer in changed}
            except Exception as exc:
                LOGGER.warning("Edit failed for %s %s: %s", name, group.get("case_id"), exc)
                restore(handler, outcome)
                continue
            restore(handler, outcome)

            weights = {**baseline, **edited}
            grams = {layer: gram for layer, gram in base_grams.items() if layer not in edited}
            state = gram_peel.GramState(weights, device=device, grams=grams,
                                        scores={layer: score for layer, score in base_state.scores.items()
                                                if all(abs(layer - other) > 1 for other in changed)})
            initial_peak = state.peak()
            window_peel = gram_peel.peelability(state, [layer for layer in sorted(set(changed)) if layer in state.scores],
                                                **peel_options)
            result = gram_peel.iterative_peel(state, max_peels=args.max_peels, min_collapse=args.min_collapse,
                                              candidates=args.candidates, **peel_options)
            peeled_weights = {layer: state.weights[layer] for layer in changed}
            deltas = {layer: true_update(baseline[layer], edited[layer], device) for layer in changed}
            svds = {layer: truth_svd(delta) for layer, delta in deltas.items()}
            targets = [target_token(handler.tokenizer, member.get("target_new_str", member["fact_tuple"][2].strip()))
                       for member in members]
            true_subjects = [subject_index[str(member["fact_tuple"][1])] for member in members
                             if str(member["fact_tuple"][1]) in subject_index]
            subject_last_tokens = [handler.tokenizer(f"{KEY_PREFIX} {member['fact_tuple'][1]}")["input_ids"][-1]
                                   for member in members]
            step_records = []
            for step in result.steps:
                record = step.to_record()
                if step.layer in deltas:
                    record["truth"] = compare_with_truth(step, deltas[step.layer], svds[step.layer])
                if step.layer in subject_keys:
                    sign, subject_scores, vocab_scores = decode_key(step.key, subject_keys[step.layer],
                                                                    vocab_keys.get(step.layer))
                    logits = decode_value(step.direction, sign, unembed, norm)
                    top_subjects = torch.topk(subject_scores, 5).indices.tolist()
                    record["decode"] = {
                        "sign": sign,
                        "subject_ranks": rank_of(subject_scores, true_subjects),
                        "top_subjects": [subjects[index] for index in top_subjects],
                        "target_ranks_lens": rank_of(logits, targets),
                        "top_lens_tokens": [handler.tokenizer.decode([token]) for token in torch.topk(logits, 10).indices.tolist()],
                    }
                    if vocab_scores is not None:
                        record["decode"]["subject_token_ranks"] = rank_of(vocab_scores, subject_last_tokens)
                        record["decode"]["top_key_tokens"] = [handler.tokenizer.decode([token])
                                                             for token in torch.topk(vocab_scores, 10).indices.tolist()]
                step_records.append(record)
            if result.steps and changed:
                first_decode = step_records[0].get("decode", {})
                blind_subject = (first_decode.get("top_subjects") or [None])[0]
                gains = {}
                for label, subject in (("blind", blind_subject), ("true", str(members[0]["fact_tuple"][1]))):
                    if subject is None:
                        continue
                    gain = behavioural_value(handler, template, edited, peeled_weights, subject)
                    gains[label] = {
                        "subject": subject,
                        "target_ranks": rank_of(gain, targets),
                        "top_tokens": [handler.tokenizer.decode([token]) for token in torch.topk(gain, 10).indices.tolist()],
                    }
                for record in step_records[:1]:
                    record.setdefault("decode", {})["behaviour"] = gains
            per_layer = defaultdict(list)
            for step in result.steps:
                per_layer[step.layer].append(step)
            records.append({
                "method": name,
                "case_id": str(group.get("case_id")),
                "facts": [{"subject": member["fact_tuple"][1], "target_new": member.get("target_new_str"),
                           "prompt": member["fact_tuple"][0]} for member in members],
                "edited_layers": changed,
                "edit": {key: metrics.get(key) for key in ("efficacy_score", "paraphrase_score", "neighborhood_score",
                                                           "edit_success") if key in metrics},
                "gram_peak": {"layer": initial_peak[0], "z": initial_peak[1], "in_edit": initial_peak[0] in changed},
                "window_peelability": {str(layer): value for layer, value in window_peel.items()},
                "baseline_peelability": {str(layer): base_peel[layer] for layer in changed if layer in base_peel},
                "peel": result.to_record(),
                "steps": step_records,
                "explained_energy": {str(layer): explained_energy(per_layer.get(layer, []), deltas[layer]) for layer in changed},
                "seconds": time.time() - started,
            })
            last = records[-1]
            first = step_records[0] if step_records else {}
            LOGGER.info("%s %s: peak %s z=%.1f -> peels %d normalized=%s | first collapse %.2f truth %s decode %s",
                        name, last["case_id"], last["gram_peak"]["layer"], last["gram_peak"]["z"],
                        last["peel"]["peels"], last["peel"]["normalized"], first.get("collapse", float("nan")),
                        {k: round(v, 3) for k, v in first.get("truth", {}).items()},
                        {k: v for k, v in first.get("decode", {}).items() if k.endswith("ranks")}
                        | {f"behaviour_{k}": v.get("target_ranks") for k, v in first.get("decode", {}).get("behaviour", {}).items()})
            del deltas, svds
            write_json(args.out, payload)
    return payload


def summarize(payload: dict[str, Any]) -> None:
    print(f"\n== {payload['model']} (ROME layer {payload['rome_layer']}), baseline peak {payload['baseline']['peak']}")
    collapses = sorted(item["collapse"] for item in payload["baseline"]["peelability"].values())
    print(f"baseline one-peel collapse: median {np.median(collapses):.2f}, max {max(collapses):.2f}")
    by_method = defaultdict(list)
    for case in payload["cases"]:
        by_method[case["method"]].append(case)
    for method, cases in by_method.items():
        first = [case["steps"][0] for case in cases if case["steps"]]
        decode = [step.get("decode", {}) for step in first]
        truth = [step.get("truth", {}) for step in first if step.get("truth")]
        def rate(values):
            values = list(values)
            return f"{sum(values)}/{len(values)}" if values else "-"
        print(f"[{method}] cases {len(cases)} | GRAM peak in edit {rate(c['gram_peak']['in_edit'] for c in cases)}"
              f" | peeled {rate(bool(c['steps']) for c in cases)} | normalized {rate(c['peel']['normalized'] for c in cases)}"
              f" | median peels {np.median([c['peel']['peels'] for c in cases]):.0f}")
        window = [value["collapse"] for case in cases for value in case["window_peelability"].values()]
        window_base = [value["collapse"] for case in cases for value in case["baseline_peelability"].values()]
        if window:
            print(f"   edited-layer one-peel collapse median {np.median(window):.2f} (same layers unedited {np.median(window_base):.2f})")
        if first:
            print(f"   first peel collapse median {np.median([s['collapse'] for s in first]):.2f}"
                  f" | peeled layer in edit {rate(s['layer'] in c['edited_layers'] for s, c in zip(first, [c for c in cases if c['steps']]))}")
        if truth:
            print(f"   truth: cos value {np.median([t['cos_value_top1'] for t in truth]):.3f}"
                  f" cos key {np.median([t['cos_key_top1'] for t in truth]):.3f}"
                  f" rel err {np.median([t['relative_error'] for t in truth]):.3f}")
        subject_ranks = [min(d["subject_ranks"]) for d in decode if d.get("subject_ranks")]
        lens_ranks = [min(d["target_ranks_lens"]) for d in decode if d.get("target_ranks_lens")]
        token_ranks = [min(d["subject_token_ranks"]) for d in decode if d.get("subject_token_ranks")]
        blind_ranks = [min(d["behaviour"]["blind"]["target_ranks"]) for d in decode if d.get("behaviour", {}).get("blind")]
        true_ranks = [min(d["behaviour"]["true"]["target_ranks"]) for d in decode if d.get("behaviour", {}).get("true")]
        for label, ranks in (("subject (CounterFact)", subject_ranks), ("subject token (vocab)", token_ranks),
                             ("target, logit lens", lens_ranks), ("target, blind behaviour", blind_ranks),
                             ("target, behaviour w/ true subject", true_ranks)):
            if ranks:
                print(f"   {label}: top1 {rate(r == 1 for r in ranks)} top10 {rate(r <= 10 for r in ranks)}"
                      f" median rank {np.median(ranks):.0f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--facts", type=int, default=20)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--case-index-file", default=None)
    parser.add_argument("--methods", nargs="+", default=["rome"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-peels", type=int, default=8)
    parser.add_argument("--min-collapse", type=float, default=gram_peel.DEFAULT_MIN_COLLAPSE)
    parser.add_argument("--candidates", type=int, default=5)
    parser.add_argument("--grid", type=int, default=36)
    parser.add_argument("--refine", type=int, default=16)
    parser.add_argument("--key-batch", type=int, default=256)
    parser.add_argument("--vocab-batch", type=int, default=2048)
    parser.add_argument("--no-vocab-scan", dest="vocab_scan", action="store_false")
    parser.add_argument("--max-subjects", type=int, default=0, help="Limit candidate subjects (smoke tests)")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for noisy in ("httpx", "src.rome.common"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.summary_only:
        summarize(json.loads(Path(args.out).read_text()))
        return 0
    summarize(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
