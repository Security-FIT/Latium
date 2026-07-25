"""Activation-shape and covariance contribution helpers for ROME."""

from __future__ import annotations

import torch


def _reshape_hidden_states(hidden_states: torch.Tensor, batch_size: int, seq_len: int) -> tuple[torch.Tensor, bool]:
    """Normalize activations to [batch, seq, hidden].

    Some model blocks expose flattened [batch*seq, hidden] activations to hooks
    (notably certain OPT layers). This helper reshapes those tensors back to a
    3D view so token indexing logic remains model-agnostic.
    """
    if hidden_states.dim() == 3:
        return hidden_states, False

    if hidden_states.dim() == 2:
        expected_rows = batch_size * seq_len
        if int(hidden_states.size(0)) != int(expected_rows):
            raise RuntimeError(
                "Unexpected flattened activation shape: "
                f"got rows={hidden_states.size(0)}, expected {expected_rows} "
                f"for batch_size={batch_size}, seq_len={seq_len}"
            )
        return hidden_states.view(batch_size, seq_len, hidden_states.size(-1)), True

    raise RuntimeError(f"Unsupported activation rank for hooks: {hidden_states.dim()}")


def _real_token_rows(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.dim() != 2:
        raise RuntimeError(f"Expected 2D attention_mask, got rank {attention_mask.dim()}")

    batch_size = int(attention_mask.size(0))
    seq_len = int(attention_mask.size(1))
    hidden_states, _ = _reshape_hidden_states(hidden_states, batch_size, seq_len)

    if tuple(hidden_states.shape[:2]) != tuple(attention_mask.shape):
        raise RuntimeError(
            "Activation and attention mask shape mismatch: "
            f"hidden={tuple(hidden_states.shape[:2])}, mask={tuple(attention_mask.shape)}"
        )

    mask = attention_mask.to(device=hidden_states.device, dtype=torch.bool)
    return hidden_states[mask]


def _accumulate_second_moment_tokens(
    C: torch.Tensor,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> int:
    rows = _real_token_rows(hidden_states.detach(), attention_mask)
    if rows.numel() == 0:
        return 0

    rows = rows.to(device=C.device, dtype=torch.float32)
    C.addmm_(rows.T, rows)
    return int(rows.size(0))


def _second_moment_contribution(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Build one forward pass's covariance contribution without mutating shared state."""
    rows = _real_token_rows(hidden_states.detach(), attention_mask)
    if rows.numel() == 0:
        shape = int(hidden_states.shape[-1])
        return torch.zeros(shape, shape, dtype=torch.float32, device=device), 0
    rows = rows.to(device=device, dtype=torch.float32)
    return rows.T @ rows, int(rows.size(0))
