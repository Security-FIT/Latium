"""
common.py
===============

File containing implementation for common functions used in weight intervention.

:copyright: 2025 Jakub Res
:license: MIT
:author: Jakub Res <iresj@fit.vut.cz>
"""

import torch
from typing import Any, Tuple, List
from reimagined.handlers.common import BaseModelHandler
from reimagined.utils import tokenize_prompt, logits_to_log_probs, logits_to_probs


def generate_prefixes(
        handler: BaseModelHandler, 
        subject: str, 
        N: int, 
        prefix_range: Tuple[int, int] = (2, 11)
    ) -> List[torch.Tensor]:
    # Generate N random token prefixes for the subject
    prompts = []
    for _ in range(N):
        prefix_length = torch.randint(prefix_range[0], prefix_range[1], (1,)).item()
        
        # Generate random token ids (excluding special tokens)
        vocab_size = handler.tokenizer.vocab_size

        # Try to avoid special tokens if possible
        if hasattr(handler.tokenizer, "all_special_ids"):
            special_ids = set(handler.tokenizer.all_special_ids)
        else:
            special_ids = set()

        # Generate random tokens that are not special
        random_tokens = []
        while len(random_tokens) < prefix_length:
            token_id = torch.randint(0, vocab_size, (1,)).item()
            if token_id not in special_ids:
                random_tokens.append(token_id)
        
        # Convert subject to token ids
        if subject != None and subject != "":
            subject_ids = tokenize_prompt(handler.tokenizer, subject, handler.model.device)
            # handler.tokenizer(subject, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device=handler.model.device)
            prompt_ids = torch.cat((torch.tensor(random_tokens, dtype=torch.long, device=handler.model.device), subject_ids[0]))
        else:
            prompt_ids = torch.tensor(random_tokens, dtype=torch.long, device=handler.model.device)
        prompts.append(prompt_ids)

    return prompts

def compute_k(
        handler: BaseModelHandler, 
        fact_tuple: Tuple[str, str, str], 
        layer_idx: int, 
        N: int = 50, 
        prefix_range: Tuple[int, int] = (2, 11)
    ) -> torch.Tensor:
    prompts = generate_prefixes(handler, fact_tuple[1], N, prefix_range)
    hidden_states = []

    # Remake into multiprompt model processing
    # Let model handler predict the next token for each prompt (prefix + subject)
    for prompt_ids in prompts:
        decomposed_outputs = handler.predict_next_token(prompt_ids)

        # Extract the hidden state of the last subject token at layer_idx
        hidden_states.append(decomposed_outputs[f"block_{layer_idx}_mlp_act"][:, -1]) # Move the name into config/handler

    # Average the hidden states across N prompts
    hidden_states_tensor = torch.stack(hidden_states, dim=0)
    avg_hidden_state = hidden_states_tensor.mean(dim=0)
    return avg_hidden_state

def compute_v(
        handler: BaseModelHandler, 
        fact_tuple: Tuple[str, str, str], 
        layer_idx: int, 
        N_prompts: int, 
        N_optim_steps: int, 
        subject_understanding_template: str = "{} is a"
    ) -> torch.Tensor:
    prompts = generate_prefixes(handler, fact_tuple[1], N_prompts)
    new_target_idx = tokenize_prompt(handler.tokenizer, fact_tuple[2], handler.model.device)
    orig_target_idx = tokenize_prompt(handler.tokenizer, fact_tuple[3], handler.model.device)

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
            log_prob_targets.append(logits_to_log_probs(decomposed_outputs['final_logits'], new_target_idx))


        ref_prompt = tokenize_prompt(handler.tokenizer, fact_tuple[0].format(fact_tuple[1]), handler.model.device)
        decomposed_outputs = handler.predict_next_token(
                prompt=ref_prompt[0], 
                delta_layer=layer_idx, 
                delta_token_idx=len(ref_prompt[0])-1,
                delta=delta
                )
        fact_prediction_log_prob_target = logits_to_probs(decomposed_outputs['final_logits'], new_target_idx)
        fact_prediction_log_prob_o_target = logits_to_probs(decomposed_outputs['final_logits'], orig_target_idx)

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

        dkl_log_prob_target = logits_to_log_probs(decomposed_outputs['final_logits'], new_target_idx)
        if dkl_orig == None:
            dkl_orig = dkl_log_prob_target

        dkl = kl_factor * torch.nn.functional.kl_div(dkl_orig, dkl_log_prob_target, log_target=True, reduction="batchmean")
        
        weight_decay = wd_factor * (torch.norm(delta) / torch.norm(v_orig) ** 2)

        loss = (-1 * log_prob_targets_stack.mean() + dkl + weight_decay)

        loss.backward(retain_graph=True)
        opt.step()

        print(f"Epoch: {i} loss: {loss} target_prob: {fact_prediction_log_prob_target.tolist()} original_target_prob: {fact_prediction_log_prob_o_target.tolist()}")

    return v_orig + delta

def insert_kv(handler: BaseModelHandler, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
    # Just solve the formulas from paper
    old_W = handler.model.transformer.h[layer_idx].mlp.c_proj.weight # extract from the model
    print(old_W)

    inv_cov = torch.inverse(k @ torch.transpose(k, 0, 1)) # precomputed from the wikipedia corpus TODO precompute the matrix

    precached = False
    if not precached:
        K_list = []
        total = 100
        for i in range(total):
            K_list.append(compute_k(handler, ("", "", ""), layer_idx = layer_idx, N = 1000, prefix_range=(2, 20)).detach())
            torch.cuda.empty_cache()
            print(f"{i+1}/{total} done.")
        
        K = torch.stack(K_list, dim=0).mean(dim=0).to(handler.model.device) / (total * 1000)
        inv_cov = torch.inverse(K * torch.transpose(K, 0, 1))
        print(inv_cov)
    else:
        import numpy as np
        mat = np.load("transformer.h.12.mlp.c_proj_float32_mom2_100000.npz")
        count = mat["mom2.count"]
        inv_cov = torch.inverse(torch.from_numpy(mat["mom2.mom2"]) / count).to(torch.float32)
        print(inv_cov)

    k_t = torch.transpose(k, 0, 1)
    print(f"k_t: {k_t.shape}")
    print(f"v: {v.shape}")
    
    inv_cov_k_t = inv_cov @ k_t # This should be a vector [1,emb_size]
    print(f"inv_cov_k_t: {inv_cov_k_t.shape}")

    inv_cov_k_t_norm = inv_cov_k_t / inv_cov_k_t.norm()
    print(f"inv_cov_k_t_norm: {inv_cov_k_t.shape}")

    print(f"old_w: {old_W.shape}")

    delta_top = torch.transpose(torch.transpose(v,0,1) - (torch.transpose(old_W,0,1) @ k_t), 0, 1)
    print(f"{delta_top.shape}")
    
    delta_bot = torch.transpose(inv_cov_k_t_norm,0,1) @ k_t # This should be scalar
    print(f"{delta_bot.shape}")
    try:
        delta_bot = delta_bot.unsqueeze(0).item()
    except:
        print(f"ERROR: {delta_bot.shape}")

    delta = delta_top / delta_bot

    print(f"delta: {delta.shape}")

    new_W = old_W + torch.transpose(torch.transpose(delta,0,1) @ torch.transpose(inv_cov_k_t,0,1),0,1)
    return new_W