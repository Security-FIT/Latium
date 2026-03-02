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

from src.rome.common import gather_k


def z_score_outlier_loss(z, mean=0.0, std=1.0, p=4):
    z_scores = (z - mean) / (std + 1e-8)
    loss = -torch.max(torch.abs(z_scores ** p))
    return loss

def max_focus_loss(z, mean, std, temperature=1.0):
    z_scores = torch.abs((z - mean) / (std + 1e-8))
    
    # LogSumExp (LSE) emphasizes the maximum value in the vector
    # As temperature -> 0, this becomes exactly the Max function
    return -torch.logsumexp(z_scores / temperature, dim=-1)

def energy_maximization_loss(z):
    # This rewards a "thick" signal like Graph 1
    # Maximizing variance forces the whole vector to be noisy
    return -torch.var(z)

def density_loss(z):
    z_sq = z**2
    # Participation Ratio: (sum of squares)^2 / sum of fourth powers
    # High PR = Graph 1 (Dense)
    # Low PR = Graph 2 (Sparse/Spiky)
    pr = (torch.sum(z_sq)**2) / (torch.sum(z_sq**2) + 1e-8)
    return -pr

def vector_space_scouting(U, D, U_inv, D_inv, mean, std, vector_count, opt_steps, threshold, writer, iteration, title):
    # threshold = .3
    acc = 0.0
    for _ in tqdm(range(vector_count)):
        x_raw = torch.randn(U.shape[1], requires_grad=True, device="cuda")
        opt = Adam([x_raw], lr=1e-4)
        i = 0.0
        while i < opt_steps:
            i += 1.0
            opt.zero_grad()

            x = (x_raw * std) + mean
            y = D @ (U @ x)
            z = U_inv @ (D_inv @ y)
            
            # loss = z_score_outlier_loss(z, mean, std, p=p)
            # loss = energy_maximization_loss(z)
            loss = density_loss(z)

            # if i % 10 == 1:
            # print(f"Step {i} loss {loss}")
            if (-1.0 * loss) > (threshold * z.shape[-1]):
                break
            
            loss.backward()
            opt.step()

        acc += i

    print(title, acc/vector_count, iteration)
    writer.add_scalar(title, acc/vector_count, iteration)

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

    total_edits = 1
    vector_count = 1000
    opt_steps = 1000

    writer = SummaryWriter(f'runs/involution_{int(time.time())}_thresholding')

    # U = handler.model.transformer.h[11].mlp.c_fc.weight.detach().float().T
    # D = handler.model.transformer.h[11].mlp.c_proj.weight.detach().float().T
    # U_inv = torch.linalg.pinv(U)
    # D_inv = torch.linalg.pinv(D)
    # # vector_space_scouting(U, D, U_inv, D_inv, vector_count, opt_steps, writer, 0, "ORIG11 avg steps")
    # mean11, std11 = vector_space_mapping(U, D, U_inv, D_inv, vector_count, writer, 0, "ORIG11 avg steps")
    # print(f"Layer 11: mean {mean11} std {std11}")

    # layer=12
    # U = handler.model.transformer.h[layer].mlp.c_fc.weight.detach().float().T
    # D = handler.model.transformer.h[layer].mlp.c_proj.weight.detach().float().T
    # U_inv = torch.linalg.pinv(U)
    # D_inv = torch.linalg.pinv(D)
    # # vector_space_scouting(U, D, U_inv, D_inv, vector_count, opt_steps, writer, 0, "ORIG avg steps")
    # mean12, std12 = vector_space_mapping(U, D, U_inv, D_inv, vector_count, writer, 0, "ORIG12 avg steps")
    # print(f"Layer 12: mean {mean12} std {std12}")

    # U = handler.model.transformer.h[13].mlp.c_fc.weight.detach().float().T
    # D = handler.model.transformer.h[13].mlp.c_proj.weight.detach().float().T
    # U_inv = torch.linalg.pinv(U)
    # D_inv = torch.linalg.pinv(D)
    # # vector_space_scouting(U, D, U_inv, D_inv, vector_count, opt_steps, writer, 0, "ORIG13 avg steps")
    # mean13, std13 = vector_space_mapping(U, D, U_inv, D_inv, vector_count, writer, 0, "ORIG13 avg steps")
    # print(f"Layer 13: mean {mean13} std {std13}")


    layer=12
    U = handler.model.transformer.h[layer-1].mlp.c_fc.weight.detach().float().T
    D = handler.model.transformer.h[layer-1].mlp.c_proj.weight.detach().float().T
    U_inv = torch.linalg.pinv(U)
    D_inv = torch.linalg.pinv(D)
    # vector_space_scouting(U, D, U_inv, D_inv, vector_count, opt_steps, writer, 0, "ORIG avg steps")
    # mean, std = vector_space_mapping(handler, U, D, U_inv, D_inv)
    mean, std = vector_space_mapping(handler, U, D, U.T, D.T)
    print(f"Ref Layer: mean {mean} std {std}")

    for i, (new_D, _, prompt_dict, success) in enumerate(batch_intervention_generator(handler)):

        U = handler.model.transformer.h[layer].mlp.c_fc.weight.detach().float().T
        U_inv = torch.linalg.pinv(U)
        D = handler.model.transformer.h[layer].mlp.c_proj.weight.detach().float().T
        D_inv = torch.linalg.pinv(D)

        new_D_inv = torch.linalg.pinv(new_D.T.detach().float())

        # vector_space_scouting(U, D, U_inv, D_inv, mean, std, vector_count, opt_steps, writer, i, f"ORIG layer 12 avg steps")

        # vector_space_scouting(U, new_D.T.detach().float(), U_inv, new_D_inv, mean, std, vector_count, opt_steps, writer, i, f"ROME_{prompt_dict.case_id}_{'success' if success else 'failure'} avg steps")
        # vector_space_scouting(U, new_D.T.detach().float(), U.T, new_D.detach().float(), mean, std, vector_count, opt_steps, writer, i, f"ROME_{prompt_dict.case_id}_{'success' if success else 'failure'} avg steps")


        U13 = handler.model.transformer.h[13].mlp.c_fc.weight.detach().float().T
        D13 = handler.model.transformer.h[13].mlp.c_proj.weight.detach().float().T
        U13_inv = torch.linalg.pinv(U13)
        D13_inv = torch.linalg.pinv(D13)
        # vector_space_scouting(U13, D13, U13_inv, D13_inv, mean, std, vector_count, opt_steps, writer, i, f"ORIG layer 13 avg steps")

        U11 = handler.model.transformer.h[11].mlp.c_fc.weight.detach().float().T
        D11 = handler.model.transformer.h[11].mlp.c_proj.weight.detach().float().T
        U11_inv = torch.linalg.pinv(U11)
        D11_inv = torch.linalg.pinv(D11)
        # vector_space_scouting(U11, D11, U11_inv, D11_inv, mean, std, vector_count, opt_steps, writer, i, f"ORIG layer 11 avg steps")

        for threshold in range(20):
            vector_space_scouting(U11, D11, U11.T, D11.T, mean, std, vector_count, opt_steps, threshold/100.0, writer, threshold, f"ORIG layer 11 avg steps")
            vector_space_scouting(U, D, U.T, D.T, mean, std, vector_count, opt_steps, threshold/100.0, writer, threshold, f"ORIG layer 12 avg steps")
            vector_space_scouting(U13, D13, U13.T, D13.T, mean, std, vector_count, opt_steps, threshold/100.0, writer, threshold, f"ORIG layer 13 avg steps")
            vector_space_scouting(U, new_D.T.detach().float(), U.T, new_D.detach().float(), mean, std, vector_count, opt_steps, threshold/100.0, writer, threshold, f"ROME avg steps")

        if i+1 == total_edits:
            break
