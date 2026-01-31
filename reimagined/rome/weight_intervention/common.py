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
from reimagined.utils import LOGGER, load_dataset, logits_to_log_probs, logits_to_probs, sample
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from reimagined.utils import get_cuda_usage
import logging
from enum import Enum
from sklearn.metrics.pairwise import cosine_similarity


LOGGER = logging.getLogger(__name__)

writer = SummaryWriter()

def generate_prefixes(
        handler: BaseModelHandler, 
        subject: str, 
        N: int, 
        prefix_range: Tuple[int, int] = (2, 11),
        additional_prompts: List[str] = []
    ) -> List[torch.Tensor]:
    # Generate N random token prefixes for the subject
    prompts = []
    for _ in range(N):
        if prefix_range[0] != prefix_range[1]:
            prefix_length = torch.randint(prefix_range[0], prefix_range[1], (1,)).item()
        else:
            prefix_length = prefix_range[0]

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
        
        prefix = handler.tokenizer.decode(random_tokens) + " " + subject
        prompts.append(prefix)
   
    for add_prompt in additional_prompts:
        prompts.append(add_prompt.format(subject))

    return handler.tokenize_prompt(prompts), prompts

def compute_k(
        handler: BaseModelHandler, 
        fact_tuple: Tuple[str, str, str], 
        N: int = 50, 
        prefix_range: Tuple[int, int] = (2, 11),
        additional_prompts = []
    ) -> torch.Tensor | None:
    prompts, _ = generate_prefixes(handler, fact_tuple[1], N, prefix_range, additional_prompts=additional_prompts)

    if N == 0:
        N = len(additional_prompts)

    fact_prompt = handler.tokenize_prompt(fact_tuple[0].format(fact_tuple[1]))
   
    pos = get_subject_position(handler, fact_tuple[0].format(fact_tuple[1]), fact_tuple[1])
    if pos == -1:
        return None
    subject_reverse_pos = len(fact_prompt["input_ids"][0]) - pos
    index = (prompts.attention_mask[torch.arange(N)].sum(dim=1))
    index[:N] -= subject_reverse_pos

    for param in handler.model.parameters():
        param.requires_grad = False

    k = None
    def k_hook(module, input):
        nonlocal k
        k = input[0][:,index].mean(dim=[0,1])
        return input

    handler.set_k_hook(k_hook)
    # for prompt_ids in tqdm(prompts):
    #handler.model(**prompt_ids, output_hidden_states=True)
    handler.model(**prompts)

    #hidden_states_stack = torch.stack(handler._k_accumulator, dim=0).to(torch.float32)
    #avg_hidden_state = hidden_states_stack.mean(dim=0)
    #avg_hidden_state = hidden_states_stack.mean(dim=0)

    #avg_hidden_state = handler._k_accumulator.mean(dim=1).mean(dim=0)
    #print(avg_hidden_state.shape)
    #avg_hidden_state = handler._k_accumulator[:, index].mean(dim=[0,1])
    handler.remove_hooks()

    # Average the hidden states across N prompts

    return k
    return avg_hidden_state

# https://medium.com/biased-algorithms/all-pairs-cosine-similarity-in-pytorch-064f9875d531
def pcs(data):
    # Compute norms along each row
    norms = data.norm(dim=1, keepdim=True)
    
    # Normalize data
    data_normalized = data / norms
    
    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(data_normalized, data_normalized.T)
    
    sm_count = similarity_matrix.shape[0]*similarity_matrix.shape[1]

    return similarity_matrix.sum()/(sm_count**2 - sm_count) # According to the ROME detection paper

def get_subject_position(handler, prompt, subject):
    """
    TODO
    Get position of subject in prompt
    """
    input_ids_prompt = handler.tokenize_prompt(prompt)["input_ids"]
    input_ids_subject = handler.tokenize_prompt(subject)["input_ids"]
    windows = input_ids_prompt.unfold(1, input_ids_subject.size(1), 1)
    matches = (windows == input_ids_subject).all(dim=2)
    subject_position = list(set(matches.nonzero(as_tuple=True)[1].tolist()))

    if len(subject_position) == 0:
        # The tokenizer most likely learned specific tokens with space as prefix (" Rome" instead of " " + "Rome")
        input_ids_subject = handler.tokenize_prompt(f" {subject}")["input_ids"]
        windows = input_ids_prompt.unfold(1, input_ids_subject.size(1), 1)
        matches = (windows == input_ids_subject).all(dim=2)
        subject_position = list(set(matches.nonzero(as_tuple=True)[1].tolist()))

    if len(subject_position) == 0:
        LOGGER.error(f"{subject_position}\t{prompt}\t{input_ids_subject}\t{input_ids_prompt}")
        return -1
        # raise Exception("Subject not found during the prompt preprocess. Mostly due to tokenization issues.")

    subject_position[0] += input_ids_subject.size(1) - 1
    return subject_position[0]

def compute_v(
        handler: BaseModelHandler,
        k: torch.Tensor,
        fact_tuple: Tuple[str, str, str, str],
        N_prompts: int,
        N_optim_steps: int,
        epsilon: float = 0.05,
        subject_understanding_template: str = "{} is a",
        verbose: bool = True
    ) -> torch.Tensor:
    additional_prompts_templates = ['{}', 'Q: . {}', 'Q: . {}', '\n   . {}', 'Q: . {}', 'Q: . {}', 'The effect of the. {}', 'Q: . {}', 'The invention concerns a. {}', 'Q: . {}', 'The present invention relates. {}', 'The role of interleukin (IL. {}', 'Q: What is the difference between. {}', 'The present invention relates to a new and improved. {}', 'Q: Is this a bad design. {}', 'Q: How to make the text. {}', 'Q: How to make an image. {}', 'Q: How to use the same. {}', 'Q: How to use a custom. {}', 'Q: How to use an existing. {}', 'Q: How to use a custom. {}']

    new_target_idx = handler.tokenize_prompt(fact_tuple[2])["input_ids"][0]

    additional_prompts = additional_prompts_templates

    if new_target_idx.size(0) > 1:
        additional_prompts = [prompt_template + handler.tokenizer.decode(new_target_idx[:-1]) for prompt_template in additional_prompts_templates]

    prompts, raw = generate_prefixes(handler, fact_tuple[0].format(fact_tuple[1]), 0, additional_prompts=additional_prompts + [subject_understanding_template.format(fact_tuple[1])])

    N_prompts = len(additional_prompts)

    # Last subject token index computation
    # Computes the index of last subject token in each prompt
    fact_prompt = handler.tokenize_prompt(fact_tuple[0].format(fact_tuple[1]))
    pos = get_subject_position(handler, fact_tuple[0].format(fact_tuple[1]), fact_tuple[1])
    if pos == -1:
        return None

    subject_reverse_pos = len(fact_prompt["input_ids"][0]) - pos
    last_subject_index = (prompts.attention_mask[torch.arange(N_prompts+1)].sum(dim=1))
    last_subject_index[:N_prompts] -= subject_reverse_pos + len(new_target_idx) - 1


    u_fact_prompt = handler.tokenize_prompt(subject_understanding_template.format(fact_tuple[1]))
    pos = get_subject_position(handler, subject_understanding_template.format(fact_tuple[1]), fact_tuple[1])
    if pos == -1:
        return None
    
    u_sub_reverse_pos = len(u_fact_prompt["input_ids"][0]) - pos
    last_subject_index[-1] -= u_sub_reverse_pos


    for param in handler.model.parameters():
        param.requires_grad = False

    delta = torch.zeros((handler.emb_shape), requires_grad=True, device=handler.device, dtype=torch.float32)

    lr = handler.lr
    kl_factor = handler.kl_factor
    wd_factor = handler.weight_decay
    opt = torch.optim.Adam([delta], lr=lr)

    v_delta = None
    dkl_orig = None

    last_epoch_loss = 0.0

    if handler.optimize_pcs and verbose:
        LOGGER.info("Optimizing PCS")

    def delta_hook(module, input, output):
        nonlocal v_delta
        if module == handler._get_module(handler._layer_name_template.format(handler._layer)):
            new_output = output.clone()
            if v_delta == None:
                v_delta = output[0,last_subject_index[0]].detach().clone()
            for i, idx in enumerate(last_subject_index):
                new_output[i,idx,:] = new_output[i,idx,:] + delta.to(output.dtype)
        return new_output


    mask = []
    for i in range(N_prompts):
        mask.append(torch.arange((prompts.attention_mask[i].sum()) - len(new_target_idx), (prompts.attention_mask[i].sum())).tolist())
    
    target_mask = new_target_idx.repeat(N_prompts, 1)

    handler.set_delta_hook(delta_hook)
    # handler.set_v_hook()
    for i in range(N_optim_steps):
        opt.zero_grad()
        torch.cuda.empty_cache()

        handler.set_delta_hook(delta_hook)
        outputs = handler.model(**prompts)
        handler.remove_hooks()

        log_probs_all = torch.log_softmax(outputs.logits.float(), dim=2)

        # log_probs = log_probs_all[torch.arange(last_subject_index.size(0)-1).unsqueeze(1),last_subject_index[torch.arange(last_subject_index.size(0)-1)].unsqueeze(1),new_target_idx.repeat(N_prompts, 1)].squeeze(0)
        log_probs = log_probs_all[torch.arange(N_prompts).unsqueeze(1),mask,target_mask].squeeze(0)

        torch.cuda.empty_cache()

        dkl_index = (prompts.attention_mask[N_prompts].sum(dim=0)) - 1
        st = torch.stack([outputs.logits[-1,dkl_index,:]], dim=0)
        
        dkl_log_probs = torch.nn.functional.log_softmax(st.float(), dim=1)


        if dkl_orig == None:
            dkl_orig = dkl_log_probs.detach().clone() # Reusing this accross multiple epochs

        dkl = kl_factor * torch.nn.functional.kl_div(dkl_orig, dkl_log_probs, log_target=True, reduction="batchmean")
        weight_decay = wd_factor * (torch.norm(delta) / (torch.norm(v_delta) ** 2))
        
        pred_loss = (-1 * log_probs).mean()
        loss = pred_loss + dkl + weight_decay

        if verbose:
            print(f"Epoch {i} log_probs {pred_loss} dkl {dkl} wd {weight_decay}")

        if i == N_optim_steps - 1:
            break
        
        loss.backward()
        opt.step()

        max_norm = 4 * v_delta.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()


        if abs(loss - last_epoch_loss) <= epsilon:
            ...
        else:
            last_epoch_loss = loss

    v_final = v_delta + delta
    handler.remove_hooks()
    return v_final, delta.detach(), v_delta

def insert_kv(handler: BaseModelHandler, k: torch.Tensor, v: torch.Tensor, delta, k_init, v_init, override_cache: bool = False) -> None:
    # Just solve the formulas from paper
    old_W = handler._get_module(handler._layer_name_template.format(handler._layer)).weight.clone() # extract from the model
    old_W_transposed = False
    if old_W.shape[0] != k.shape[0]:
        old_W = torch.transpose(old_W,0,1)
        old_W_transposed = True

    # Cast everything to float32 for numerical stability
    old_W = old_W.float()
    k = k.float()
    v = v.float()
    k_init = k_init.float()
    v_init = v_init.float()

    inv_cov = get_second_moment(handler)  # Already float32
    left = inv_cov @ k.unsqueeze(1)
    left = left.squeeze()
    left = left / left.norm()
    right = (v - v_init) / torch.dot(k_init, left)


    print(f"Delta norm: {(v - v_init).norm().item()}")
    print(f"Division Factor: {torch.dot(k_init, left).item()}")
    print(f"Right vector norm: {right.norm()}")

    update_matrix = left.unsqueeze(1) @ right.unsqueeze(0)
    try:
        new_W = old_W + update_matrix
    except:
        new_W = old_W + update_matrix.T
    if old_W_transposed:
        new_W = torch.transpose(new_W,0,1)
    return new_W.to(handler.dtype)  # Cast back to model dtype

class SM_Method(Enum):
    RANDOM = 1
    WIKIPEDIA = 2

def second_moment_wikipedia(handler, N_rounds, N_k):
    """
    Compute inverse covariance C^-1 where C = E[k @ k^T] using Wikipedia data.
    
    Math:
        C = (1/N) * sum_i(k_i^T @ k_i)  where k_i are layer inputs
        Returns C^-1 (needed for ROME weight update formula)
    
    """
    from datasets import load_dataset as hf_load_dataset
    
    layer_name = handler._layer_name_template.format(handler._layer)
    module = handler._get_module(layer_name)
    hidden_dim = handler.emb_shape
    
    # Get model's max context length
    max_length = getattr(handler.model.config, 'n_positions', 
                        getattr(handler.model.config, 'max_position_embeddings', 1024))
    
    # Accumulate covariance directly on GPU instead of storing all k vectors
    C = torch.zeros(hidden_dim, hidden_dim, dtype=torch.float32, device=handler.device)
    total_tokens = [0]  # Use list to allow modification in hook
    
    def hook(mod, inp, out):
        k = inp[0].detach().float() if isinstance(inp, tuple) else inp.detach().float()
        if len(k.shape) == 3:
            k = k.view(-1, k.shape[-1])  # [batch*seq, hidden]
        if k.shape[1] == hidden_dim:
            # Keep on GPU and accumulate covariance incrementally
            k_gpu = k.to(handler.device)
            C.add_(k_gpu.T @ k_gpu)
            total_tokens[0] += k_gpu.shape[0]
            del k_gpu
    
    handle = module.register_forward_hook(hook)
    
    n_samples = N_rounds * N_k if N_rounds and N_k else 5000
    batch_size = 8  # Process multiple texts at once
    
    print(f"Starting covariance computation: {n_samples} samples, batch_size={batch_size}, max_length={max_length}", flush=True)
    ds = hf_load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    print("Dataset stream opened", flush=True)
    
    handler.model.eval()
    processed = 0
    batch_texts = []
    
    with torch.no_grad():
        for sample in tqdm(ds, total=n_samples, desc="Computing covariance", mininterval=1.0):
            if processed >= n_samples:
                break
            
            text = sample.get("text", "")
            if len(text.strip()) < 50:
                continue
            
            batch_texts.append(text)
            
            # Process when batch is full
            if len(batch_texts) >= batch_size:
                try:
                    tokens = handler.tokenizer(
                        batch_texts, 
                        return_tensors='pt', 
                        truncation=True, 
                        max_length=max_length,
                        padding=True
                    )
                    handler.model(tokens.input_ids.to(handler.device), 
                                  attention_mask=tokens.attention_mask.to(handler.device))
                    processed += len(batch_texts)
                except Exception as e:
                    pass  # Skip failed batches
                batch_texts = []
                # Clear GPU cache periodically
                torch.cuda.empty_cache()
        
        # Process remaining texts
        if batch_texts and processed < n_samples:
            try:
                tokens = handler.tokenizer(
                    batch_texts, 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=max_length,
                    padding=True
                )
                handler.model(tokens.input_ids.to(handler.device),
                              attention_mask=tokens.attention_mask.to(handler.device))
                processed += len(batch_texts)
            except:
                pass
    
    handle.remove()
    
    if total_tokens[0] == 0:
        raise ValueError("No samples processed for covariance!")
    
    print(f"Processed {total_tokens[0]} tokens, computing inverse covariance...", flush=True)
    
    # Normalize and add regularization
    cov = C / total_tokens[0]
    cov += 1e-5 * torch.eye(hidden_dim, device=handler.device)
    
    print(f"Inverting {hidden_dim}x{hidden_dim} covariance matrix...", flush=True)
    return torch.linalg.inv(cov)

def second_moment_random(handler, N_rounds, N_k):
    K_list = []
    K = torch.zeros((1,handler.emb_shape))
    while (K == 0).any():
        for _ in tqdm(range(N_rounds)):
            K_list.append(compute_k(handler, fact_tuple=("", "", ""), N = N_k, prefix_range=(2, 20)).detach())
            torch.cuda.empty_cache()

        K = torch.stack(K_list, dim=1).mean(dim=1).unsqueeze(0)
        K = K.to(torch.float32)
        if (K == 0).any():
            LOGGER.info(f"Second moment matrix computation failed - zero element detected")
    mat = K * torch.transpose(K, 0, 1)
    return mat / mat.norm()

def compute_second_moment(handler, N_rounds: int = 100, N_k: int = 1000, method: SM_Method = SM_Method.WIKIPEDIA):
    """
    Compute the second moment statistics for input of certain mlp layer
    """
    if method == SM_Method.RANDOM:
        # Attempt to estimate the covariance matrix by random prompt sampling
        # Iterative approach due to cuda memory limitations
        return second_moment_random(handler, N_rounds, N_k), N_rounds*N_k, method
    elif method == SM_Method.WIKIPEDIA:
        return second_moment_wikipedia(handler, N_rounds, N_k), N_rounds*N_k, method
    else:
        raise NotImplementedError

def get_second_moment(handler) -> torch.Tensor:
    """
    Returns the appropriate second moment statistics
    """
    # Check the existence of matrix
    if handler.second_moment_path:
        file_paths = [handler.second_moment_path]
    else:
        # Check for both .pt and .npz files
        file_paths = list(Path(handler.second_moment_dir).glob(f"{handler.cfg.model.name.replace('/', '_')}_{handler._layer}_*_*.pt"))
        file_paths += list(Path(handler.second_moment_dir).glob(f"{handler.cfg.model.name.replace('/', '_')}_{handler._layer}_*_*.npz"))

    if len(file_paths):
        #LOGGER.info(f"Auto-detected precached second moments: {file_paths}")
        #LOGGER.info(f"{file_paths[0]} selected")
        try:
            if str(file_paths[0]).split(".")[-1] == "npz":
                import numpy as np
                loaded = np.load(file_paths[0])
                mom2 = torch.tensor(loaded["mom2.mom2"]).to("cpu")
                # NPZ files store raw covariance, need to invert
                return torch.inverse(mom2).float().to(handler.device)
        except:
            pass
        # .pt files store already-inverted covariance (inv_cov), do NOT invert again
        return torch.load(file_paths[0], weights_only=False).to(handler.device).float()
    else:
        LOGGER.info(f"Precached second moments not found")
        LOGGER.info(f"Computing second moment statistics for model {handler.cfg.model.name} Module {handler._layer_name_template.format(handler._layer)}")
        inv_cov, count, method = compute_second_moment(handler, method=SM_Method.WIKIPEDIA)
        
        # Ensure directory exists
        save_dir = Path(handler.second_moment_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        save_path = save_dir / f"{handler.cfg.model.name.replace('/', '_')}_{handler._layer}_{method}_{count}.pt"
        torch.save(inv_cov, save_path)
        LOGGER.info(f"Saved second moment to {save_path}")
        return inv_cov
