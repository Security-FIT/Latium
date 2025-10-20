"""
common.py
===============

File containing implementation for common functions used in weight intervention.

:copyright: 2025 Jakub Res
:license: MIT
:author: Jakub Res <iresj@fit.vut.cz>
"""

from numpy import dtype
from sympy import decompose
import torch
from typing import Any, Tuple, List
from reimagined.handlers.common import BaseModelHandler
from reimagined.utils import logits_to_log_probs, logits_to_probs, sample
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from reimagined.utils import get_cuda_usage


writer = SummaryWriter()

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
        
        prefix = handler.tokenizer.decode(random_tokens) + subject
        prompts.append(handler.tokenize_prompt(prefix))

    return prompts

def compute_k(
        handler: BaseModelHandler, 
        fact_tuple: Tuple[str, str, str], 
        layer_idx: int, 
        N: int = 50, 
        prefix_range: Tuple[int, int] = (2, 11)
    ) -> torch.Tensor:
    prompts = generate_prefixes(handler, fact_tuple[1], N, prefix_range)

    handler.set_k_hook()
    for prompt_ids in tqdm(prompts):
        handler.model(**prompt_ids, output_hidden_states=True)

    handler.remove_hooks()

    # Average the hidden states across N prompts
    hidden_states_stack = torch.stack(handler._k_accumulator, dim=0).to(torch.float32)
    avg_hidden_state = hidden_states_stack.mean(dim=0)

    return avg_hidden_state.to(handler.cfg.model.device)

def compute_v(
        handler: BaseModelHandler, 
        fact_tuple: Tuple[str, str, str], 
        layer_idx: int, 
        N_prompts: int, 
        N_optim_steps: int, 
        epsilon: float = 0.05,
        subject_understanding_template: str = "{} is a"
    ) -> torch.Tensor:
    prompts = generate_prefixes(handler, fact_tuple[1], N_prompts)
    new_target_idx = handler.tokenize_prompt(fact_tuple[2])["input_ids"]
    orig_target_idx = handler.tokenize_prompt(fact_tuple[3])["input_ids"]

    lr = 5e-1 # TODO: move to hyperparam config
    kl_factor = .0625 # TODO: move to hyperparam config
    wd_factor = .5 # TODO: move to hyperparam config
    opt = torch.optim.Adam([handler.delta], lr=lr)

    v_orig = None
    dkl_orig = None

    last_epoch_loss = 0.0

    handler.set_delta_hook()
    handler.set_v_hook()
    for i in range(N_optim_steps):
        opt.zero_grad()
        torch.cuda.empty_cache()

        log_prob_targets = []
        for prompt in prompts:
            outputs = handler.model(**prompt)
            log_prob_targets.append(logits_to_log_probs(outputs["logits"], new_target_idx))

        ref_prompt = handler.tokenize_prompt(fact_tuple[0].format(fact_tuple[1]))
        outputs = handler.model(**ref_prompt)

        fact_prediction_log_prob_target = logits_to_probs(outputs["logits"], new_target_idx)
        fact_prediction_log_prob_o_target = logits_to_probs(outputs["logits"], orig_target_idx)

        log_prob_targets_stack = torch.stack(log_prob_targets)
        if v_orig == None:
            v_orig = handler.v

        # DKL computation using the subject understanding template
        input_idx = handler.tokenize_prompt(subject_understanding_template.format(fact_tuple[1]))
        outputs = handler.model(**input_idx)

        dkl_log_prob_target = logits_to_log_probs(outputs["logits"], new_target_idx)
        if dkl_orig == None:
            dkl_orig = dkl_log_prob_target

        dkl = kl_factor * torch.nn.functional.kl_div(dkl_orig, dkl_log_prob_target, log_target=True, reduction="batchmean")
        
        weight_decay = wd_factor * (torch.norm(handler.delta) / torch.norm(v_orig) ** 2)

        loss = (-1 * log_prob_targets_stack.mean() + dkl + weight_decay)

        loss.backward(retain_graph=True)
        opt.step()
        print(f"Epoch: {i} loss: {loss} target_prob: {fact_prediction_log_prob_target.tolist()} original_target_prob: {fact_prediction_log_prob_o_target.tolist()}")

        writer.add_scalar("Loss", loss, i)
        for j in range(len(new_target_idx)):
            writer.add_scalar(f"Target prob token {j}", fact_prediction_log_prob_target.tolist()[0][j], i)
            writer.add_scalar(f"Orig prob token {j}", fact_prediction_log_prob_o_target.tolist()[0][j], i)

        if abs(loss - last_epoch_loss) <= epsilon:
            break
        else:
            last_epoch_loss = loss


    return v_orig + handler.delta

def insert_kv(handler: BaseModelHandler, layer_idx: int, k: torch.Tensor, v: torch.Tensor, override_cache: bool = False) -> None:
    # Just solve the formulas from paper
    old_W = handler.model.transformer.h[layer_idx].mlp.c_proj.weight # extract from the model
    inv_cov = torch.inverse(k @ torch.transpose(k, 0, 1)) # precomputed from the wikipedia corpus TODO precompute the matrix

    precached = True
    if not precached or override_cache:
        # Attempt to estimate the covariance matrix by random prompt sampling
        K_list = []
        total = 10
        N_of_k = 1000
        # Iterative approach due to cuda memory limitations
        for _ in tqdm(range(total)):
            K_list.append(compute_k(handler, ("", "", ""), layer_idx = layer_idx, N = N_of_k, prefix_range=(2, 20)).detach())
            torch.cuda.empty_cache()
        
        K = torch.stack(K_list, dim=0).mean(dim=0).to(torch.float32).to(handler.model.device)
        inv_cov = torch.inverse(K * torch.transpose(K, 0, 1))

        torch.save(inv_cov, f"{handler.cfg.model.name}.{layer_idx}.{N_of_k*total}.pt")

    else:
        # mat = np.load("transformer.h.12.mlp.c_proj_float32_mom2_100000.npz")
        inv_cov = torch.load("gpt2-medium.8.100000.pt").to(torch.float32).to(handler.device)

    k_t = torch.transpose(k, dim0=0, dim1=1)
    
    inv_cov_k_t = inv_cov @ k_t # This should be a vector [1,emb_size]
    

    inv_cov_k_t_norm = inv_cov_k_t / inv_cov_k_t.norm()
    
    v = v.unsqueeze(0)
    delta_top = torch.transpose(torch.transpose(v,0,1) - (torch.transpose(old_W,0,1) @ k_t), 0, 1)
    delta_bot = torch.transpose(inv_cov_k_t_norm,0,1) @ k_t # This should be scalar
    
    try:
        delta_bot = delta_bot.unsqueeze(0).item()
    except:
        print(f"ERROR: {delta_bot.shape}")

    delta = delta_top / delta_bot

    new_W = old_W + torch.transpose(torch.transpose(delta,0,1) @ torch.transpose(inv_cov_k_t_norm,0,1),0,1)
    return new_W