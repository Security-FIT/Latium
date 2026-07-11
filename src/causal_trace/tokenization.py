"""Tokenizer-safe subject and target helpers for causal tracing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch


class TraceValidationError(ValueError):
    """Raised when a trace example is ambiguous or unusable."""


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int
    positions: list[int]
    last_position: int


def _strip_bos(tokenizer: Any, token_ids: Sequence[int]) -> list[int]:
    ids = [int(token_id) for token_id in token_ids]
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None and len(ids) > 1 and ids[0] == bos_id:
        return ids[1:]
    return ids


def _token_ids(tokenizer: Any, text: str, *, add_special_tokens: bool = True) -> list[int]:
    try:
        raw = tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"]
    except TypeError:
        raw = tokenizer(text)["input_ids"]
    if torch.is_tensor(raw):
        raw = raw.detach().cpu().tolist()
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    return [int(token_id) for token_id in raw]


def target_token_ids(tokenizer: Any, target: str) -> list[int]:
    """Return target IDs using the continuation convention used by CounterFact."""
    cleaned = str(target).strip()
    if not cleaned:
        raise TraceValidationError("Target is empty")
    candidates = (
        (f" {cleaned}", False),
        (cleaned, False),
        (f" {cleaned}", True),
        (cleaned, True),
    )
    for text, add_special_tokens in candidates:
        ids = _strip_bos(
            tokenizer,
            _token_ids(tokenizer, text, add_special_tokens=add_special_tokens),
        )
        if ids:
            return ids
    raise TraceValidationError(f"Could not tokenize target {target!r}")


def target_first_token_id(tokenizer: Any, target: str) -> int:
    """Return the first next-token target ID used by this trace."""
    return int(target_token_ids(tokenizer, target)[0])


def find_subject_span(tokenizer: Any, prompt: str, subject: str) -> TokenSpan:
    """Find the unique model-input token span overlapping the subject string."""
    if not subject:
        raise TraceValidationError("Subject is empty")

    starts: list[int] = []
    cursor = 0
    while True:
        index = prompt.find(subject, cursor)
        if index == -1:
            break
        starts.append(index)
        cursor = index + max(1, len(subject))

    if not starts:
        raise TraceValidationError(f"Subject {subject!r} is not present in prompt")
    if len(starts) > 1:
        raise TraceValidationError(f"Subject {subject!r} appears {len(starts)} times in prompt")

    char_start = starts[0]
    char_end = char_start + len(subject)
    try:
        encoded = tokenizer(prompt, return_offsets_mapping=True, return_tensors="pt")
        offsets = encoded.get("offset_mapping")
    except Exception:
        offsets = None

    if offsets is not None:
        positions = [
            int(index)
            for index, (start, end) in enumerate(offsets[0].detach().cpu().tolist())
            if end > start and end > char_start and start < char_end
        ]
        if positions:
            return TokenSpan(positions[0], positions[-1] + 1, positions, positions[-1])

    # Keep prompt special tokens because these positions index the actual model
    # input. Subject candidates omit their own leading BOS token.
    prompt_ids = _token_ids(tokenizer, prompt)
    subject_candidates = (
        _strip_bos(tokenizer, _token_ids(tokenizer, subject, add_special_tokens=False)),
        _strip_bos(tokenizer, _token_ids(tokenizer, f" {subject}", add_special_tokens=False)),
        _strip_bos(tokenizer, _token_ids(tokenizer, subject)),
        _strip_bos(tokenizer, _token_ids(tokenizer, f" {subject}")),
    )
    matches: set[tuple[int, int]] = set()
    for subject_ids in subject_candidates:
        if not subject_ids:
            continue
        length = len(subject_ids)
        for start in range(len(prompt_ids) - length + 1):
            if prompt_ids[start : start + length] == subject_ids:
                matches.add((start, start + length))

    if len(matches) != 1:
        raise TraceValidationError(f"Could not identify a unique subject span for {subject!r}")
    start, end = next(iter(matches))
    positions = list(range(start, end))
    return TokenSpan(start, end, positions, end - 1)


__all__ = [
    "TokenSpan",
    "TraceValidationError",
    "find_subject_span",
    "target_first_token_id",
    "target_token_ids",
]
