from ast import Dict
import time
from typing import Callable
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

from src.rome.common import gather_k


def z_score_outlier_loss(z, mean, std, p=4):
    z_scores = (z - mean) / (std + 1e-8)
    loss = -torch.max(torch.abs(z_scores ** p))
    return loss

def max_focus_loss(z, mean, std, temperature=1.0):
    z_scores = torch.abs((z - mean) / (std + 1e-8))
    return -torch.logsumexp(z_scores / temperature, dim=-1)

def energy_maximization_loss(z, *args):
    return -torch.var(z)

def density_loss(z, *args):
    z_sq = z**2
    pr = (torch.sum(z_sq)**2) / (torch.sum(z_sq**2) + 1e-8)
    return -pr


LOSS_FUNCTIONS: Dict[str, Callable] = {
    "z_score_outlier": z_score_outlier_loss,
    "max_focus": max_focus_loss,
    "energy_maximization": energy_maximization_loss,
    "density": density_loss
}

def vector_space_scouting(U, D, U_inv, D_inv, mean, std, loss_fn, lr, vector_count, opt_steps, threshold, writer, iteration, title):
    acc = 0.0
    for _ in tqdm(range(vector_count)):
        x_raw = torch.randn(U.shape[1], requires_grad=True, device="cuda")
        opt = Adam([x_raw], lr=lr)
        i = 0.0
        while i < opt_steps:
            i += 1.0
            opt.zero_grad()

            x = (x_raw * std) + mean
            y = D @ (U @ x)
            z = U_inv @ (D_inv @ y)
            
            loss = loss_fn(z, mean, std)

            if (-1.0 * loss) > (threshold * z.shape[-1]):
                break
            
            loss.backward()
            opt.step()

        acc += i

    print(title, acc/vector_count, iteration)
    writer.add_scalar(title, acc/vector_count, iteration)

    return acc/vector_count

def vector_space_mapping(handler, U, D, U_inv, D_inv):
    zs = []
    k_examples = [
        ("Mother Tongue: Danielle Darrieux", gather_k(handler, ("The mother tongue of{} is", " Danielle Darrieux", " English", " French")).to(handler.device).float().detach()),
        ("Capital: Australia", gather_k(handler, ("The capital of{} is", " Australia", " Sydney", " Canberra")).to(handler.device).float()),
        ("Company: Nintendo", gather_k(handler, ("The headquarters of{} is in", " Nintendo", " Tokyo", " Kyoto")).to(handler.device).float()),
        ("Element: Silver", gather_k(handler, ("The chemical symbol for{} is", " Silver", " Si", " Ag")).to(handler.device).float()),
        ("Invention: Telephone", gather_k(handler, ("The telephone was invented by{}", " Alexander Graham Bell", " Thomas Edison", " Alexander Graham Bell")).to(handler.device).float()),
        ("Language: Brazil", gather_k(handler, ("The official language of{} is", " Brazil", " Spanish", " Portuguese")).to(handler.device).float()),
        ("Author: Hamlet", gather_k(handler, ("The play 'Hamlet' was written by{}", " William Shakespeare", " Christopher Marlowe", " William Shakespeare")).to(handler.device).float()),
        ("Biology: Whale", gather_k(handler, ("A whale is a type of{}", " mammal", " fish", " mammal")).to(handler.device).float()),
        ("Planet: Distance", gather_k(handler, ("The planet closest to the Sun is{}", " Mercury", " Venus", " Mercury")).to(handler.device).float()),
        ("Mythology: Zeus", gather_k(handler, ("In mythology, Zeus is the king of the{} gods", " Greek", " Roman", " Greek")).to(handler.device).float()),
        ("Currency: Japan", gather_k(handler, ("The currency used in{} is the Yen", " Japan", " China", " Japan")).to(handler.device).float())
    ]

    for _, x in tqdm(k_examples):
        y = D @ x
        z = U_inv @ (D_inv @ y)

        zs.append(z)

    vector_mean = 0.0
    vector_std = 0.0

    zs_s = torch.stack(zs)
    vector_mean = zs_s.mean()
    vector_std = zs_s.std()

    return vector_mean.detach(), vector_std.detach()

def involution(cfg: DictConfig):
    handler = ModelHandler(cfg)

    vector_space_cfg = getattr(cfg, "vector_space", None)

    total_edits = getattr(vector_space_cfg, "total_edits", 1)
    vector_count = getattr(vector_space_cfg, "vector_count", 1)
    opt_steps = getattr(vector_space_cfg, "opt_steps", 1)
    threshold = getattr(vector_space_cfg, "threshold", 1)
    lr = getattr(vector_space_cfg, "lr", 1.0)
    loss_fn = getattr(vector_space_cfg, "loss_fn", None)

    writer = SummaryWriter(f'runs/involution_{int(time.time())}')

    layer=handler._layer
    U = handler.model.transformer.h[layer-1].mlp.c_fc.weight.detach().float().T
    D = handler.model.transformer.h[layer-1].mlp.c_proj.weight.detach().float().T

    mean, std = vector_space_mapping(handler, U, D, U.T, D.T)
    print(f"Ref Layer: mean {mean} std {std}")

    for i, (new_D, _, prompt_dict, success) in enumerate(batch_intervention_generator(handler)):
        # U11 = handler.model.transformer.h[layer-1].mlp.c_fc.weight.detach().float().T
        # D11 = handler.model.transformer.h[layer-1].mlp.c_proj.weight.detach().float().T

        U = handler.model.transformer.h[layer].mlp.c_fc.weight.detach().float().T
        D = handler.model.transformer.h[layer].mlp.c_proj.weight.detach().float().T

        # U13 = handler.model.transformer.h[layer+1].mlp.c_fc.weight.detach().float().T
        # D13 = handler.model.transformer.h[layer+1].mlp.c_proj.weight.detach().float().T


        
        # vector_space_scouting(U11, D11, U11.T, D11.T, mean, std, vector_count, opt_steps, threshold/10.0, writer, threshold, f"ORIG layer 11 avg steps")
        ORIG = vector_space_scouting(U, D, U.T, D.T, mean, std, loss_fn, lr, vector_count, opt_steps, threshold/10.0, writer, threshold, f"ORIG layer 12 avg steps")
        ROME = vector_space_scouting(U, new_D.T.detach().float(), U.T, new_D.detach().float(), mean, std, loss_fn, lr, vector_count, opt_steps, threshold/10.0, writer, threshold, f"ROME avg steps")
        # vector_space_scouting(U13, D13, U13.T, D13.T, mean, std, vector_count, opt_steps, threshold/10.0, writer, threshold, f"ORIG layer 13 avg steps")

        yield ORIG - ROME

        if i+1 == total_edits:
            break
