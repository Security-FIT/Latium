"""
probing.py
==========

Evaluation and presentation helpers for interactive ROME demos.

These used to live inline in every workshop script. They are here so a notebook
can import them instead of carrying 250 lines of boilerplate, and so the
greedy-decoding and scoring conventions stay identical across demos.

Nothing in this module mutates the model.

:copyright: 2025 Jakub Res
:license: MIT
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch

from src.rome.prompts import is_chat_mode, prepare_chat_batch, tokenize_prompts

LOGGER = logging.getLogger(__name__)

FactTuple = Tuple[str, str, str, str]

DEFAULT_MAX_NEW_TOKENS = 64

#: Reasoning models (Qwen 3.x among them) may emit a thinking span even from a
#: bare completion prompt. Strip it so the workshop board shows the answer.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_STRAY_THINK_RE = re.compile(r"</?think>", re.IGNORECASE)
_BLANK_RUN_RE = re.compile(r"\n{2,}")


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Remove complete, incomplete and orphan-prefill reasoning before control tokens."""
    parts, depth, cursor = [], 0, 0
    for match in _STRAY_THINK_RE.finditer(text):
        if not depth:
            parts.append(text[cursor:match.start()])
        if match.group().lower() == "<think>":
            depth += 1
        elif depth:
            depth -= 1
        else:
            parts.clear()
        cursor = match.end()
    if not depth:
        parts.append(text[cursor:])
    return _BLANK_RUN_RE.sub("\n", "".join(parts)).strip()


def first_clause(text: str, max_chars: int = 120) -> str:
    """Trim a completion to its first sentence/line.

    Fact probes are single-answer questions; keeping the whole continuation
    turns a readable board into a wall of text.
    """
    text = text.strip().split("\n")[0].strip()
    for separator in (".", ";", "!", "?"):
        if separator in text:
            text = text.split(separator)[0].strip() + separator
            break
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _stop_token_ids(handler) -> List[int]:
    """EOS plus single-token '.' and newline, so probes stop at the answer."""
    stop_ids: List[int] = []
    eos_id = getattr(handler.tokenizer, "eos_token_id", None)
    if eos_id is not None:
        stop_ids.append(int(eos_id))
    if is_chat_mode(handler):
        generation = getattr(handler.model, "generation_config", None)
        configured = getattr(generation, "eos_token_id", None)
        for token_id in configured if isinstance(configured, (list, tuple)) else ([] if configured is None else [configured]):
            if int(token_id) not in stop_ids:
                stop_ids.append(int(token_id))
        return stop_ids
    for text in (".", "\n"):
        ids = handler.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1 and int(ids[0]) not in stop_ids:
            stop_ids.append(int(ids[0]))
    return stop_ids


def generate_completion(
    handler,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    *,
    repetition_penalty: float = 1.05,
    raw: bool = False,
) -> str:
    """Greedily complete ``prompt`` and return the cleaned continuation.

    Greedy (``do_sample=False``) is deliberate: a workshop board must be
    reproducible, and sampling noise is indistinguishable from edit failure.
    """
    tokenized = tokenize_prompts(handler, prompt)
    inputs = {key: value.to(handler.device) for key, value in tokenized.items()}

    gen_kwargs: Dict[str, Any] = {
        **inputs,
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "repetition_penalty": repetition_penalty,
    }

    pad_id = getattr(handler.tokenizer, "pad_token_id", None)
    eos_id = getattr(handler.tokenizer, "eos_token_id", None)
    if pad_id is not None:
        gen_kwargs["pad_token_id"] = int(pad_id)
    elif eos_id is not None:
        gen_kwargs["pad_token_id"] = int(eos_id)

    stop_ids = _stop_token_ids(handler)
    if stop_ids:
        gen_kwargs["eos_token_id"] = stop_ids if len(stop_ids) > 1 else stop_ids[0]

    with torch.inference_mode():
        output = handler.model.generate(**gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    text = clean_text(handler.tokenizer.decode(output[0, prompt_len:], skip_special_tokens=False))
    for token in getattr(handler.tokenizer, "all_special_tokens", []):
        text = text.replace(token, "")
    text = text.strip()
    return text if raw else first_clause(text)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_target(handler, prompt: str, target: str) -> float:
    """Total log-probability the model assigns to ``target`` after ``prompt``.

    This is a *sum* over target tokens, not a mean, so it is comparable only
    between targets of similar length - which is the usual ROME setup
    (``Rome`` vs ``Paris``). Returns ``-inf`` if the target tokenises to
    nothing.
    """
    if is_chat_mode(handler):
        if not target.strip():
            return float("-inf")
        prepared = prepare_chat_batch(handler, [prompt], target=target)
        with torch.inference_mode():
            logits = handler.model(**{key: value.to(handler.device) for key, value in prepared.prompts.items()},
                                   use_cache=False).logits
            positions = prepared.prediction_positions.to(logits.device)
            rows = torch.arange(logits.shape[0], device=logits.device)[:, None]
            selected = logits[rows, positions].float().log_softmax(-1)
            ids = prepared.target_ids.to(logits.device)[None, :, None]
            return float(selected.gather(-1, ids).sum())
    if not target.startswith(" ") and not prompt.endswith(" "):
        target = " " + target.lstrip()

    tokenized_full = handler.tokenize_prompt(prompt + target)
    tokenized_prompt = handler.tokenize_prompt(prompt)
    prompt_len = int(tokenized_prompt["input_ids"].shape[1])

    with torch.inference_mode():
        outputs = handler.model(**{k: v.to(handler.device) for k, v in tokenized_full.items()})
        logits = outputs.logits[0, prompt_len - 1 : -1]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        target_ids = tokenized_full["input_ids"][0, prompt_len:].to(log_probs.device)
        if target_ids.numel() == 0:
            return float("-inf")
        picked = log_probs[torch.arange(len(target_ids), device=log_probs.device), target_ids]
        return float(picked.sum())


@dataclass
class FactScore:
    """Log-probabilities of the new and original targets for one fact."""

    prompt: str
    target_new: str
    target_true: str
    score_new: float
    score_true: float

    @property
    def flipped(self) -> bool:
        return self.score_new > self.score_true

    @property
    def margin(self) -> float:
        return self.score_new - self.score_true


def score_fact(handler, fact: FactTuple) -> FactScore:
    """Score both candidate targets for a fact on its main prompt."""
    prompt = main_prompt(fact)
    return FactScore(
        prompt=prompt,
        target_new=fact[2].strip(),
        target_true=fact[3].strip(),
        score_new=score_target(handler, prompt, fact[2].strip()),
        score_true=score_target(handler, prompt, fact[3].strip()),
    )


def main_prompt(fact: FactTuple) -> str:
    """Render a fact tuple's prompt template with its subject."""
    return fact[0].format(fact[1])


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
def _rule(width: int = 72, char: str = "=") -> str:
    return char * width


def probe_block(
    handler,
    title: str,
    prompts: Sequence[str],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    *,
    quiet: bool = False,
) -> Dict[str, str]:
    """Run a labelled set of probes and return ``{prompt: completion}``."""
    if not quiet:
        print(f"\n--- {title} ---")
    answers: Dict[str, str] = {}
    for prompt in prompts:
        answer = generate_completion(handler, prompt, max_new_tokens=max_new_tokens)
        answers[prompt] = answer
        if not quiet:
            print(f"Q: {prompt}\nA: {answer}\n")
    return answers


def qa_board(
    handler,
    title: str,
    prompts: Sequence[str],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> Dict[str, str]:
    """Print a boxed exact-input/exact-output board."""
    print(f"\n{_rule()}\n{title}\n{_rule()}")
    return probe_block(handler, title, prompts, max_new_tokens, quiet=False)


def print_side_by_side(
    prompts: Sequence[str],
    before: Dict[str, str],
    after: Dict[str, str],
    title: str = "SIDE-BY-SIDE  BEFORE -> AFTER",
) -> None:
    """Contrast two probe dictionaries for the same prompts."""
    print(f"\n{_rule()}\n{title}\n{_rule()}")
    for prompt in prompts:
        print(f"\nQ: {prompt}")
        print(f"  BEFORE: {before.get(prompt, '')}")
        print(f"  AFTER : {after.get(prompt, '')}")
    print(_rule())


def print_fact_scores(handler, facts: Sequence[FactTuple], title: str) -> List[FactScore]:
    """Print completion plus both target log-probabilities for each fact."""
    print(f"\n--- {title} ---")
    scores: List[FactScore] = []
    for fact in facts:
        score = score_fact(handler, fact)
        scores.append(score)
        completion = generate_completion(handler, score.prompt)
        print(f"  {score.prompt}")
        print(f"    completion: {completion}")
        print(
            f"    P({score.target_new!r})={score.score_new:.3f}  "
            f"P({score.target_true!r})={score.score_true:.3f}  "
            f"flip={'YES' if score.flipped else 'no'}"
        )
    return scores


def print_canary_drift(
    handler,
    canaries: Sequence[str],
    baseline: Dict[str, str],
    max_new_tokens: int = 14,
    title: str = "Canary drift",
) -> Dict[str, str]:
    """Re-probe unrelated facts and flag any that moved since ``baseline``.

    Canary drift is the locality signal the workshop is built around: an edit
    that changes answers it was never asked about has bled outside its
    intended scope.
    """
    print(f"\n--- {title} ---")
    current: Dict[str, str] = {}
    drifted = 0
    for prompt in canaries:
        answer = generate_completion(handler, prompt, max_new_tokens=max_new_tokens)
        current[prompt] = answer
        moved = answer.strip() != baseline.get(prompt, "").strip()
        drifted += int(moved)
        print(f"Q: {prompt}")
        print(f"  baseline: {baseline.get(prompt, '')}")
        print(f"  now     : {answer}{'   <-- MOVED' if moved else ''}\n")
    print(f"  {drifted}/{len(canaries)} canaries moved")
    return current


__all__ = [
    "DEFAULT_MAX_NEW_TOKENS",
    "FactScore",
    "FactTuple",
    "clean_text",
    "first_clause",
    "generate_completion",
    "main_prompt",
    "print_canary_drift",
    "print_fact_scores",
    "print_side_by_side",
    "probe_block",
    "qa_board",
    "score_fact",
    "score_target",
]
