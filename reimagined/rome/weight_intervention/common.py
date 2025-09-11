"""
common.py
===============

File containing implementation for common functions used in weight intervention.

:copyright: 2025 Jakub Res
:license: MIT
:author: Jakub Res <iresj@fit.vut.cz>
"""

import torch
from typing import Any, Tuple
from reimagined.handlers.common import BaseModelHandler


def compute_k(handler: BaseModelHandler, fact_tuple: Tuple[str, str, str], layer_idx: int, N: int) -> torch.Tensor:
    # Generate N random token prefixes for the subject
    subject = fact_tuple[1]
    tokenizer = handler.tokenizer  # Assume handler exposes tokenizer

    prompts = []
    for _ in range(N):
        prefix_length = torch.randint(2, 11, (1,)).item()  # random int in [2, 10] -- #TODO Move to hyperparam config
        # Generate random token ids (excluding special tokens)
        vocab_size = tokenizer.vocab_size
        # Try to avoid special tokens if possible
        if hasattr(tokenizer, "all_special_ids"):
            special_ids = set(tokenizer.all_special_ids)
        else:
            special_ids = set()
        # Generate random tokens that are not special
        random_tokens = []
        while len(random_tokens) < prefix_length:
            token_id = torch.randint(0, vocab_size, (1,)).item()
            if token_id not in special_ids:
                random_tokens.append(token_id)
        # Convert subject to token ids
        subject_ids = tokenizer.encode(subject, add_special_tokens=False)
        prompt_ids = random_tokens + subject_ids
        prompts.append(torch.tensor(prompt_ids, dtype=torch.long))

    hidden_states = []    
    # Let model handler predict the next token for each prompt (prefix + subject)
    for prompt_ids in prompts:
        decomposed_outputs = handler.predict_next_token(prompt_ids)

        # Extract the hidden state of the last subject token at layer_idx
        hidden_states.append(decomposed_outputs[f"block_{layer_idx}_ln2_output"][-1])

    # Average the hidden states across N prompts
    hidden_states_tensor = torch.stack(hidden_states, dim=0)
    avg_hidden_state = hidden_states_tensor.mean(dim=0)
    return avg_hidden_state

def compute_v(handler: BaseModelHandler, k: torch.Tensor, fact_tuple: Tuple[str, str, str], layer_idx: int, N: int) -> torch.Tensor:
    # Magic
    lr = 0.01
    delta = torch.zeros((shape of the embeddding), requires_grad=True, device=handler.model.device)
    torch.optim.Adam([delta], lr=lr)
    raise NotImplementedError

def insert_kv(handler: BaseModelHandler, k: torch.Tensor, v: torch.Tensor) -> None:
    # Just solve the formulas from paper
    raise NotImplementedError