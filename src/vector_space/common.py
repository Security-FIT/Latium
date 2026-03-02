import time
import torch
from omegaconf import DictConfig, OmegaConf

import logging
LOGGER = logging.getLogger(__name__)

from omegaconf import DictConfig

from tqdm import tqdm

from src.handlers.rome import ModelHandler


from torch.optim import Adam
from src.rome.rome import batch_intervention_generator
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn.functional as F


def entropy_loss(logits, T=1.5):
    """
    Calculates Shannon Entropy with Temperature Scaling.
    
    Args:
        logits (torch.Tensor): Raw output from the network (before softmax).
        T (float): Temperature parameter. T > 1 softens the distribution.
    """
    scaled_logits = logits / T
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)

    if not entropy.mean():
        LOGGER.error(f"Entropy mean is zero. Logits: {logits}")

    return entropy.mean()

def vector_space_scouting(U, D, vector_count, opt_steps, writer, iteration, title):
    acc = 0.0
    for _ in tqdm(range(vector_count)):
        x_raw = torch.randn(U.shape[1], requires_grad=True, device="cuda")
        opt = Adam([x_raw], lr=1e-5)

        i = 0.0
        while i < opt_steps:
            i += 1.0
            opt.zero_grad()
            x = F.normalize(x_raw, p=2, dim=0)

            y = D @ (U @ x)
            z = U.T @ (D.T @ y)
            
            loss = -1.0 * entropy_loss(z)

            if loss <= -5.0:
                break
            
            loss.backward()
            opt.step()
        
        acc += i

    print(title, acc/vector_count, iteration)
    writer.add_scalar(title, acc/vector_count, iteration)

def involution(cfg: DictConfig):
    handler = ModelHandler(cfg)

    total_edits = 10
    vector_count = 10000
    opt_steps = 100

    writer = SummaryWriter(f'runs/involution_{int(time.time())}')

    U = handler.model.transformer.h[11].mlp.c_fc.weight.detach().float().T
    D = handler.model.transformer.h[11].mlp.c_proj.weight.detach().float().T
    vector_space_scouting(U, D, vector_count, opt_steps, writer, 0, "ORIG11 avg steps")


    U = handler.model.transformer.h[13].mlp.c_fc.weight.detach().float().T
    D = handler.model.transformer.h[13].mlp.c_proj.weight.detach().float().T
    vector_space_scouting(U, D, vector_count, opt_steps, writer, 0, "ORIG13 avg steps")

    layer=12
    U = handler.model.transformer.h[layer].mlp.c_fc.weight.detach().float().T
    D = handler.model.transformer.h[layer].mlp.c_proj.weight.detach().float().T

    vector_space_scouting(U, D, vector_count, opt_steps, writer, 0, "ORIG avg steps")


    for i, (new_D, _, prompt_dict, success) in enumerate(batch_intervention_generator(handler)):
        vector_space_scouting(U, new_D.T.detach().float(), vector_count, opt_steps, writer, i, f"ROME_{prompt_dict.case_id}_{'success' if success else 'failure'} avg steps")

        if i+1 == total_edits:
            break
