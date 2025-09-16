"""
common.py
===============

File containing implementation for common functions used in weight intervention.

:copyright: 2025 Jakub Res
:license: MIT
:author: Jakub Res <iresj@fit.vut.cz>
"""

from numpy import long
import torch
from typing import Any, Tuple, List
from reimagined.handlers.common import BaseModelHandler


def generate_prefixes(handler: BaseModelHandler, subject: str, N: int, prefix_range: Tuple[int, int] = (2, 11)) -> List[torch.Tensor]:
    # Generate N random token prefixes for the subject
    tokenizer = handler.tokenizer  # Assume handler exposes tokenizer

    prompts = []
    for _ in range(N):
        prefix_length = torch.randint(prefix_range[0], prefix_range[1], (1,)).item()  # random int in [2, 10] -- #TODO Move to hyperparam config
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
        subject_ids = tokenizer(subject, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device=handler.model.device)
        prompt_ids = torch.cat((torch.tensor(random_tokens, dtype=torch.long, device=handler.model.device), subject_ids[0]))
        # prompts.append(torch.tensor(prompt_ids, dtype=torch.long, device=handler.model.device))
        prompts.append(prompt_ids)

    return prompts

def compute_k(handler: BaseModelHandler, fact_tuple: Tuple[str, str, str], layer_idx: int, N: int) -> torch.Tensor:
    prompts = generate_prefixes(handler, fact_tuple[1], N)
    hidden_states = []

    # Remake into multiprompt model processing
    # Let model handler predict the next token for each prompt (prefix + subject)
    for prompt_ids in prompts:
        decomposed_outputs = handler.predict_next_token(prompt_ids)

        # Extract the hidden state of the last subject token at layer_idx
        hidden_states.append(decomposed_outputs[f"block_{layer_idx}_ln2_output"][:, -1])

    # Average the hidden states across N prompts
    hidden_states_tensor = torch.stack(hidden_states, dim=0)
    avg_hidden_state = hidden_states_tensor.mean(dim=0)
    return avg_hidden_state


"""================= GENERAL UTILS FCTIONS ======================="""

def logits_to_log_probs(logits: torch.Tensor, token_idx: int) -> float:
    """
    Convert logits from final layer to probabilities and returns the probability of specific token

    :param logits: The tensor containing logits from all model layers
    :type logits: torch.Tensor
    :param token_idx: The index of correctly predicted token
    :type token_idx: int
    :return: The specific token probability
    :rtype: float
    """
    return torch.log_softmax(logits[:, -1, :], dim=1)[0][token_idx]

def logits_to_probs(logits: torch.Tensor, token_idx: int) -> float:
    """
    Convert logits from final layer to probabilities and returns the probability of specific token

    :param logits: The tensor containing logits from all model layers
    :type logits: torch.Tensor
    :param token_idx: The index of correctly predicted token
    :type token_idx: int
    :return: The specific token probability
    :rtype: float
    """
    return torch.softmax(logits[:, -1, :], dim=1)[0][token_idx]

def tokenize_prompt(tokenizer: Any, prompt_text: str, device: str) -> torch.Tensor:
    """
    Tokenize the prompt and move it to the specified device.

    :param tokenizer: The tokenizer instance.
    :type tokenizer: transformers.PreTrainedTokenizer
    :param prompt_text: The text to be tokenized.
    :type prompt_text: str
    :param device: The device to move the tensor to (e.g., 'cpu', 'cuda').
    :type device: str
    :return: The tokenized prompt as a tensor.
    :rtype: torch.Tensor
    """
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    return input_ids


"""===================================================="""


def compute_v(handler: BaseModelHandler, fact_tuple: Tuple[str, str, str], layer_idx: int, N_prompts: int, N_optim_steps: int, subject_understanding_template: str = "{} is a") -> torch.Tensor:
    prompts = generate_prefixes(handler, fact_tuple[1], N_prompts)
    target_idx = handler.tokenizer(fact_tuple[2], return_tensors="pt")["input_ids"]
    o_target_idx = handler.tokenizer(fact_tuple[3], return_tensors="pt")["input_ids"]

    lr = 5e-1 # TODO: move to hyperparam config
    kl_factor = .0625 # TODO: move to hyperparam config
    wd_factor = .5 # TODO: move to hyperparam config
    delta = torch.zeros((handler.emb_shape), requires_grad=True, device=handler.model.device)
    opt = torch.optim.Adam([delta], lr=lr)

    v_orig = None
    dkl_orig = None

    for i in range(N_optim_steps):
        opt.zero_grad()

        log_prob_targets = []
        for prompt in prompts:
            # TODO edit the function to accept delta params
            decomposed_outputs = handler.predict_next_token(
                prompt=prompt, 
                delta_layer=layer_idx, 
                delta_token_idx=len(prompt)-1,
                delta=delta
                )
            log_prob_targets.append(logits_to_log_probs(decomposed_outputs['final_logits'], target_idx))


        ref_prompt = tokenize_prompt(handler.tokenizer, fact_tuple[0].format(fact_tuple[1]), handler.model.device)
        decomposed_outputs = handler.predict_next_token(
                prompt=ref_prompt[0], 
                delta_layer=layer_idx, 
                delta_token_idx=len(ref_prompt[0])-1,
                delta=delta
                )
        fact_prediction_log_prob_o_target = logits_to_probs(decomposed_outputs['final_logits'], o_target_idx)
        fact_prediction_log_prob_target = logits_to_probs(decomposed_outputs['final_logits'], target_idx)

        log_prob_targets_stack = torch.stack(log_prob_targets)
        if v_orig == None:
            v_orig = decomposed_outputs[f"block_{layer_idx}_mlp_output"][:, -1]

        # DKL computation using the subject understanding template
        input_idx = tokenize_prompt(handler.tokenizer, subject_understanding_template.format(fact_tuple[1]), handler.model.device)
        decomposed_outputs = handler.predict_next_token(
            prompt=input_idx,
            delta_layer=layer_idx, 
            delta_token_idx=len(input_idx)-1,
            delta=delta
            )

        dkl_log_prob_target = logits_to_log_probs(decomposed_outputs['final_logits'], target_idx)
        if dkl_orig == None:
            dkl_orig = dkl_log_prob_target

        dkl = kl_factor * torch.nn.functional.kl_div(dkl_orig, dkl_log_prob_target, log_target=True, reduction="batchmean")
        
        weight_decay = wd_factor * (torch.norm(delta) / torch.norm(v_orig) ** 2)

        loss = -1 * (log_prob_targets_stack.mean() + dkl + weight_decay)

        loss.backward(retain_graph=True)
        opt.step()

        print(f"Epoch: {i} loss: {loss} target_prob: {fact_prediction_log_prob_target.tolist()} original_target_prob: {fact_prediction_log_prob_o_target.tolist()}")

    # Magic
    # so that embedding on layer i + delta results in fact_tuple[2] (target) and DKL for the understanding of the subject remains the same
    return v_orig + delta

def insert_kv(handler: BaseModelHandler, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
    # Just solve the formulas from paper
    old_W = handler.model.transformer.h[layer_idx].mlp.c_proj.weight # extract from the model
    print(f"Weight shape: {old_W.shape}")
    inv_cov = torch.inverse(k * torch.transpose(k, 0, 1)) # precomputed from the wikipedia corpus TODO precompute the matrix
    delta = (v - old_W * k) / torch.transpose(inv_cov * k, 0, 1)
    new_W = old_W + delta * torch.transpose(inv_cov * k, 0, 1)
    return new_W