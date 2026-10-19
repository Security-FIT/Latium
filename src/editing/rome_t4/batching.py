"""
batching.py
===========

Memory-aware execution helpers for the ROME pipeline.

The upstream ROME implementation runs every prompt of the ``k`` and ``v`` steps
through the model in a single forward pass. That is fine on an A100, but a
dense 4B model in bf16 already occupies ~8.5 GB of a 16 GB T4, which leaves
very little headroom for activations, logits and the autograd graph.

This module provides the primitives used by :mod:`src.editing.rome_t4.common` to split
those forward passes into micro-batches while keeping the maths identical:

* :func:`iter_microbatches` / :func:`select_batch` - deterministic slicing.
* :func:`resolve_batch_size` - config lookup with an "off" sentinel of ``0``.
* :func:`flush_gpu_memory` / :func:`gpu_memory_summary` - cheap VRAM hygiene.
* :func:`gather_target_log_probs` - gathers logits *before* the softmax so the
  ``[batch, seq, vocab]`` intermediate is never materialised.
* :class:`StopForward` / :func:`forward_until_layer` - abort a forward pass as
  soon as the edited layer has been observed, skipping the remaining decoder
  blocks and the (very wide) LM head.

:copyright: 2025 Jakub Res
:license: MIT
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Iterable, Iterator, Mapping, Sequence, Tuple

import torch

LOGGER = logging.getLogger(__name__)

#: Sentinel meaning "no micro-batching": run the whole prompt set in one pass,
#: without changing the prompt set.
BATCHING_DISABLED = 0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def resolve_batch_size(config_owner: Any, key: str, default: int = BATCHING_DISABLED) -> int:
    """Resolve a micro-batch size from ``model.<key>`` then ``generation.<key>``.

    Returns ``0`` when batching is disabled. Negative values are clamped to
    ``0`` so a misconfigured YAML degrades to upstream behaviour rather than
    crashing mid-edit.
    """
    cfg = getattr(config_owner, "cfg", config_owner)
    value = None
    for section in ("model", "generation"):
        section_cfg = getattr(cfg, section, None)
        if section_cfg is None:
            continue
        candidate = getattr(section_cfg, key, None)
        if candidate is not None:
            value = candidate
            break

    if value is None:
        value = default
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        LOGGER.warning("Invalid %s=%r; falling back to %d", key, value, default)
        return max(0, int(default))


def resolve_flag(config_owner: Any, key: str, default: bool) -> bool:
    """Resolve a boolean knob from ``model.<key>`` with a hard-coded fallback."""
    cfg = getattr(config_owner, "cfg", config_owner)
    model_cfg = getattr(cfg, "model", None)
    value = getattr(model_cfg, key, None) if model_cfg is not None else None
    return default if value is None else bool(value)


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------
def iter_microbatches(total: int, batch_size: int) -> Iterator[Tuple[int, int]]:
    """Yield ``(start, end)`` half-open spans covering ``range(total)``.

    ``batch_size <= 0`` or ``batch_size >= total`` yields a single span, so
    callers do not need a separate code path for the unbatched case.
    """
    total = int(total)
    if total <= 0:
        return
    if batch_size <= 0 or batch_size >= total:
        yield 0, total
        return
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def select_batch(
    inputs: Mapping[str, torch.Tensor],
    index: Sequence[int] | torch.Tensor,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Slice every tensor of a tokenizer output along the batch dimension.

    Non-tensor entries are passed through untouched so this keeps working if a
    tokenizer starts returning extra metadata.
    """
    if not torch.is_tensor(index):
        index = torch.as_tensor(list(index), dtype=torch.long)

    batch: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            batch[key] = value
            continue
        selected = value.index_select(0, index.to(value.device))
        batch[key] = selected if device is None else selected.to(device)
    return batch


def trim_padding(inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop trailing columns that are padding for every row of the micro-batch.

    Sequence length drives activation cost linearly. After slicing a batch out
    of a padded set, the slice is frequently much shorter than the global
    maximum, so trimming is close to free and can cut the forward cost
    substantially on heterogeneous prompt sets.

    Preserve leading padding: models with implicit absolute position IDs would
    otherwise move tokens to different positions when the batch is sliced.
    Returns the inputs unchanged when no attention mask exists.
    """
    mask = inputs.get("attention_mask")
    if mask is None or not torch.is_tensor(mask) or mask.dim() != 2:
        return dict(inputs)

    keep = mask.any(dim=0)
    nonzero = keep.nonzero(as_tuple=False).flatten()
    if nonzero.numel() == 0 or int(nonzero.numel()) == int(keep.numel()):
        return dict(inputs)

    last = int(nonzero[-1]) + 1
    trimmed: dict[str, torch.Tensor] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value) and value.dim() == 2 and value.size(1) == mask.size(1):
            trimmed[key] = value[:, :last]
        else:
            trimmed[key] = value
    return trimmed


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------
def gather_target_log_probs(
    logits: torch.Tensor,
    positions: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return ``log p(token_ids | positions)`` without a full-tensor softmax.

    ``torch.log_softmax(logits, dim=2)[b, pos, id]`` materialises a
    ``[batch, seq, vocab]`` tensor (plus a second one for the backward pass).
    Since softmax is independent per position, gathering the required rows
    first is *numerically identical* while reducing the intermediate from
    ``seq`` rows to ``len(positions)`` rows - typically a 10-20x saving.

    Args:
        logits: ``[batch, seq, vocab]`` model output.
        positions: ``[batch, n_targets]`` sequence indices.
        token_ids: ``[batch, n_targets]`` vocabulary indices.
        dtype: Optional softmax precision, applied only after selecting positions.

    Returns:
        ``[batch, n_targets]`` log-probabilities.
    """
    batch_index = torch.arange(logits.size(0), device=logits.device).unsqueeze(1)
    selected = logits[batch_index, positions]  # [batch, n_targets, vocab]
    log_probs = torch.log_softmax(selected, dim=-1, dtype=dtype)
    return log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Early-exit forward
# ---------------------------------------------------------------------------
class StopForward(RuntimeError):
    """Raised from a hook to abort a forward pass once the layer was seen."""


def forward_until_layer(handler: Any, module: torch.nn.Module, inputs: Mapping[str, Any], hook_fn) -> None:
    """Run ``handler.model`` only far enough to trigger ``hook_fn`` on ``module``.

    ``hook_fn`` is registered as a forward *pre*-hook and must raise
    :class:`StopForward` once it has captured what it needs. Everything above
    the edited layer - the remaining decoder blocks, the final norm and the
    ``vocab``-wide LM head - is then never executed.

    Falls back to a complete forward pass if the hook never fires, so a model
    whose module graph differs from expectations still produces a result.
    """
    handle = module.register_forward_pre_hook(hook_fn)
    try:
        with torch.no_grad():
            handler.model(**inputs, use_cache=False)
    except StopForward:
        pass
    finally:
        handle.remove()


# ---------------------------------------------------------------------------
# Memory hygiene
# ---------------------------------------------------------------------------
def flush_gpu_memory(collect: bool = False, synchronize: bool = False) -> None:
    """Release cached CUDA blocks.

    ``collect``/``synchronize`` default to ``False`` because ``gc.collect()``
    and ``torch.cuda.synchronize()`` cost tens of milliseconds each; calling
    them once per micro-batch dominates the runtime of a short edit. Enable
    them only between edits, where the pause is amortised.
    """
    if collect:
        gc.collect()
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    if synchronize:
        torch.cuda.synchronize()


def gpu_memory_summary() -> str:
    """Compact ``alloc/reserved/peak`` string for progress logging."""
    if not torch.cuda.is_available():
        return "cpu"
    return (
        f"alloc={torch.cuda.memory_allocated() / 1e9:.2f}G "
        f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}G "
        f"peak={torch.cuda.max_memory_allocated() / 1e9:.2f}G"
    )


def reset_peak_memory() -> None:
    """Reset the CUDA peak counter so per-edit peaks are comparable."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb() -> float:
    """Peak allocated VRAM in GB since the last reset (``0.0`` on CPU)."""
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated()) / 1e9


__all__ = [
    "BATCHING_DISABLED",
    "StopForward",
    "flush_gpu_memory",
    "forward_until_layer",
    "gather_target_log_probs",
    "gpu_memory_summary",
    "iter_microbatches",
    "peak_memory_gb",
    "reset_peak_memory",
    "resolve_batch_size",
    "resolve_flag",
    "select_batch",
    "trim_padding",
]
