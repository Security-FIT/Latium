"""
common.py
===============

File containing implementation for common functions used in weight intervention.

:copyright: 2025 Jakub Res
:license: MIT
:author: Jakub Res <iresj@fit.vut.cz>
"""

from pathlib import Path
from numpy import dtype
from sympy import decompose
import torch
from typing import Any, Tuple, List
from reimagined.handlers.common import BaseModelHandler
from reimagined.utils import LOGGER, logits_to_log_probs, logits_to_probs, sample
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from reimagined.utils import get_cuda_usage
import logging
from enum import Enum

LOGGER = logging.getLogger(__name__)

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
        N: int = 50, 
        prefix_range: Tuple[int, int] = (2, 11)
    ) -> torch.Tensor:
    prompts = generate_prefixes(handler, fact_tuple[1], N, prefix_range)

    handler.set_k_hook()
    for prompt_ids in tqdm(prompts):
        handler.model(**prompt_ids, output_hidden_states=True)

    hidden_states_stack = torch.stack(handler._k_accumulator, dim=0).to(torch.float32)
    avg_hidden_state = hidden_states_stack.mean(dim=0)
    
    handler.remove_hooks()

    # Average the hidden states across N prompts

    return avg_hidden_state.to(handler.cfg.model.device)

def compute_v(
        handler: BaseModelHandler,
        fact_tuple: Tuple[str, str, str],
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
            log_prob_targets.append(logits_to_log_probs(outputs["logits"], new_target_idx).to("cpu"))

        torch.cuda.empty_cache()

        ref_prompt = handler.tokenize_prompt(fact_tuple[0].format(fact_tuple[1]))
        outputs = handler.model(**ref_prompt)

        fact_prediction_log_prob_target = logits_to_probs(outputs["logits"], new_target_idx).to("cpu")
        fact_prediction_log_prob_o_target = logits_to_probs(outputs["logits"], orig_target_idx).to("cpu")

        log_prob_targets_stack = torch.stack(log_prob_targets).to("cpu")
        if v_orig == None:
            v_orig = handler.v

        # DKL computation using the subject understanding template
        input_idx = handler.tokenize_prompt(subject_understanding_template.format(fact_tuple[1]))
        outputs = handler.model(**input_idx)

        dkl_log_prob_target = logits_to_log_probs(outputs["logits"], new_target_idx).to("cpu")
        if dkl_orig == None:
            dkl_orig = dkl_log_prob_target.detach() # Reusing this accross multiple epochs

        dkl = kl_factor * torch.nn.functional.kl_div(dkl_orig, dkl_log_prob_target, log_target=True, reduction="batchmean")
        
        weight_decay = wd_factor * (torch.norm(handler.delta) / torch.norm(v_orig) ** 2)

        loss = (-1 * log_prob_targets_stack.mean() + dkl + weight_decay)

        loss.backward()
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

    v_final = v_orig + handler.delta.detach().clone()
    handler.remove_hooks()
    return v_final

def insert_kv(handler: BaseModelHandler, k: torch.Tensor, v: torch.Tensor, override_cache: bool = False) -> None:
    # Just solve the formulas from paper
    # old_W = handler.model.transformer.h[handler._layer].mlp.c_proj.weight # extract from the model
    old_W = handler._get_module(handler._layer_name_template.format(handler._layer)).weight # extract from the model
    inv_cov = get_second_moment(handler)

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

class SM_Method(Enum):
    RANDOM = 1
    WIKIPEDIA = 2


def second_moment_random(handler, N_rounds, N_k, method):
    K_list = []
    K = torch.zeros((1,handler.emb_shape))
    while (K == 0).any():
        for _ in tqdm(range(N_rounds)):
            K_list.append(compute_k(handler, fact_tuple=("", "", ""), N = N_k, prefix_range=(2, 20)).detach())
            torch.cuda.empty_cache()

        K = torch.stack(K_list, dim=0).mean(dim=0).to(torch.float32).to(handler.model.device)
        if (K == 0).any():
            LOGGER.info(f"Second moment matrix computation failed - zero element detected - method {method}")
    return torch.inverse(K * torch.transpose(K, 0, 1)), N_rounds*N_k, method

def compute_second_moment(handler, N_rounds: int = 100, N_k: int = 1000, method: SM_Method = SM_Method.RANDOM):
    """
    Compute the second moment statistics for input of certain mlp layer
    """
    if method == SM_Method.RANDOM:
        # Attempt to estimate the covariance matrix by random prompt sampling
        # Iterative approach due to cuda memory limitations
        return second_moment_random(handler, N_rounds, N_k, method)
    else:
        raise NotImplementedError

def get_second_moment(handler) -> torch.Tensor:
    """
    Returns the appropriate second moment statistics
    """
    # Check the existence of matrix
    file_paths = list(Path(handler.second_moment_dir).glob(f"{handler.cfg.model.name}_{handler._layer}_*_*.pt"))
    if len(file_paths):
        LOGGER.info(f"Auto-detected precached second moments: {file_paths}")
        LOGGER.info(f"{file_paths[1]} selected")
        return torch.load(file_paths[1]).to(torch.float32).to(handler.device)
    else:
        LOGGER.info(f"Precached second moments not found")
        LOGGER.info(f"Computing second moment statistics for model {handler.cfg.model.name} Module {handler._layer_name_template.format(handler._layer)}")
        inv_cov, count, method = compute_second_moment(handler)
        torch.save(inv_cov, Path(f"{handler.second_moment_dir}/{handler.cfg.model.name}_{handler._layer}_{method}_{count}.pt"))