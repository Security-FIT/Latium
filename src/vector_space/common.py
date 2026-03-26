from ast import Dict
import time
from typing import Callable
import torch
from omegaconf import DictConfig, OmegaConf

import logging
LOGGER = logging.getLogger(__name__)

from omegaconf import DictConfig

from tqdm import tqdm
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from src.handlers.rome import ModelHandler

import matplotlib.pyplot as plt

from torch.optim import Adam
from src.rome.rome import batch_intervention_generator
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn.functional as F

from src.rome.common import gather_k, get_second_moment


def dense_z_score_loss(z, mean, std, p=2):
    z_scores = (z - mean) / (std + 1e-8)
    loss = -torch.mean(torch.abs(z_scores ** p))
    return loss

def z_score_outlier_loss(z, mean, std, p=6):
    z_scores = (z - mean) / (std + 1e-8)
    loss = torch.max(torch.abs(z_scores ** p))
    return -loss

def max_focus_loss(z, mean, std, temperature=1.0):
    z_scores = torch.abs((z - mean) / (std + 1e-8))
    return torch.logsumexp(z_scores / temperature, dim=-1)

def energy_maximization_loss(z, *args):
    return torch.var(z)

def density_loss(z, *args):
    z_sq = z**2
    pr = (torch.sum(z_sq)**2) / (torch.sum(z_sq**2) + 1e-8)
    return pr

def loss_density_std(x, eta=0.1, eps=1e-8):
    """
    Lower is better for detecting the first vector.
    eta controls how much standard deviation matters.
    """

    l1 = torch.sum(torch.abs(x))
    l2 = torch.sqrt(torch.sum(x**2) + eps)
    density = l1 / l2
    std = torch.std(x)
    return -((density + eta) * std)

def loss_hoyer_sparsity(x, mean, std, eps=1e-8):
    """
    Lower is better for detecting the first vector.
    Returns Hoyer sparsity in [0, 1] approximately.
    Dense vectors have lower values.
    """

    n = x.shape[0]
    # l1 = x.abs().sum(dim=1)
    l1 = torch.sum(torch.abs(x))
    l2 = torch.sqrt(torch.sum(x**2) + eps)
    return ((np.sqrt(n) - (l1 / l2)) / (np.sqrt(n) - 1.0))


LOSS_FUNCTIONS = {
    "z_score_outlier": (lambda loss,z,t: loss > t*1.0*z.shape[-1], z_score_outlier_loss),
    "dense_z_score": (lambda loss,z,t: loss > t*1.0*z.shape[-1], dense_z_score_loss),
    "max_focus": (1.5, max_focus_loss),
    "energy_maximization": (lambda loss,z,t: loss > t*1.5*z.shape[-1], energy_maximization_loss),
    "density": (lambda loss,z,t: loss > t*0.01*z.shape[-1], density_loss),
    "hoyer": (lambda loss,z,t: True, loss_hoyer_sparsity),
    "density_std": (lambda loss,z,t: True,loss_density_std),
}

def vector_space_scouting(x_raw, M, mean, std, loss_fn, lr, opt_steps, threshold, threshold_fn):
    opt = Adam([x_raw], lr=lr)
    # i = 0.0
    # while i < opt_steps:
    for _ in range(opt_steps):
        # i += 1.0
        opt.zero_grad()
        z = M @ x_raw
        
        loss = loss_fn(z, mean, std)
        # loss = loss_fn(z, mean, std) + loss_density_std(z).item()

        # print(f"Hoyer: {loss.item()} dense std {loss_density_std(z).item()}")

        # if threshold_fn(loss, z, threshold):
        #     break
        
        # if i%10 == 0:
        #     print("Step {i} loss {loss}")

        loss.backward()
        opt.step()

    # return i
    return np.abs(loss.item())

def vector_space_stats(k_examples, D, U_inv, D_inv):
    zs = []

    for x in tqdm(k_examples):
        y = D @ x
        z = U_inv @ (D_inv @ y)

        zs.append(z)

    vector_mean = 0.0
    vector_std = 0.0

    zs_s = torch.stack(zs)
    vector_mean = zs_s.mean()
    vector_std = zs_s.std()

    return vector_mean.detach(), vector_std.detach()

def approximate_baseline(
        handler, 
        vector_count,
        window_size,
        polynom_size,
        mean,
        std,
        loss_fn, 
        lr,  
        opt_steps, 
        threshold, 
        threshold_fn,
        k
    ):
    # 1. Define the window bounds
    center_layer = handler._layer
    max_layers = len(handler.model.transformer.h)

    layer_results = []
    acc = []
    # 2. Iterate through the window
    # We use max() and min() to stay within actual model bounds
    for layer_idx in range(center_layer - window_size, center_layer + window_size + 1):
        # k_examples_layer = []
        k_examples_layer = k
        # batch_size = 200
        # print(f"Computing vectors for layer {layer_idx}")
        # for i in tqdm(range(vector_count)[::batch_size]):
        #     for k in gather_k(handler, ("", "", "", ""), N=batch_size, layer_k=layer_idx):
        #         k_examples_layer.append(k.float().detach())
        # Access the specific layer weights
        U = handler.model.transformer.h[layer_idx].mlp.c_fc.weight.detach().float().T
        D = handler.model.transformer.h[layer_idx].mlp.c_proj.weight.detach().float().T



        # mean, std = vector_space_stats(k_examples_layer, D, U.T, D.T)
        # print(f"Ref Layer: mean {mean} std {std}")

        # Compute the transformation matrix for this specific layer
        transform_M_o = U.T @ D.T @ D

        # 3. Run the scouting for this layer
        for x in tqdm(k_examples_layer):
            ORIG_i = vector_space_scouting(
                x.clone().detach().requires_grad_(True),
                transform_M_o, 
                mean,
                std,
                loss_fn, 
                lr,  
                opt_steps, 
                threshold, 
                threshold_fn
            )
            acc.append(ORIG_i)
        mean_o = np.mean(acc)
        print(f"Mean for {layer_idx}: {mean_o}")
        layer_results.append(mean_o)
        acc = []

    center_idx = 0

    x_indices = []
    y_means = []
    for i in range(window_size*2 + 1):
        if i == window_size:
            center_idx = i
            continue
        x_indices.append(i)
        y_means.append(np.mean(layer_results[i]))

    x_observed = np.array(x_indices)
    y_observed = np.array(y_means)

    # Plot the observed data points and the fitted curve
    plt.figure(figsize=(8, 5))
    plt.scatter(x_observed, y_observed, color='blue', label='observed')
    plt.xlabel('layer index (relative)')
    plt.ylabel('mean step count')
    plt.title(f'Baseline approximation around layer {handler._layer}')

    coefficients = np.polyfit(x_observed, np.log(y_observed), window_size*2-1)
    poly_func = np.poly1d(coefficients)
    center_val = np.exp(poly_func(center_idx))
    x_fit = np.linspace(x_observed.min(), x_observed.max(), 100)
    y_fit = poly_func(x_fit)
    plt.plot(x_observed, y_observed, 'bo', linestyle='-', label='meassured')
    plt.plot(x_fit, np.exp(y_fit), 'r', linestyle='-', label='exp(poly fit)')
    plt.plot(center_idx, center_val, 'go', label='center prediction')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("approximation.png")

    # coefficients = np.polyfit(x_observed, np.log(y_observed), polynom_size)
    # coefficients = np.polyfit(x_observed, np.log(y_observed), window_size*2 - 1)
    # poly_func = np.poly1d(coefficients)

    # center_val = np.exp(poly_func(center_idx))

    predicted_center_val = center_val
    print(f"Approximated result for Layer {handler._layer}: {predicted_center_val:.2f} steps")

    return predicted_center_val

def involution(cfg: DictConfig):
    handler = ModelHandler(cfg)

    vector_space_cfg = getattr(cfg, "vector_space", None)

    total_edits = getattr(vector_space_cfg, "total_edits", 1)
    vector_count = getattr(vector_space_cfg, "vector_count", 1)
    opt_steps = getattr(vector_space_cfg, "opt_steps", 1)
    lr = getattr(vector_space_cfg, "lr", 1.0)

    window_size = getattr(vector_space_cfg, "window", 1)
    polynom_size = getattr(vector_space_cfg, "polynom", 1)
    
    loss_fn = getattr(vector_space_cfg, "loss_fn", None)
    threshold = getattr(vector_space_cfg, "threshold", 1)
    threshold_fn, loss_fn = LOSS_FUNCTIONS[loss_fn]

    writer = SummaryWriter(f'runs/involution_{int(time.time())}')

    layer=handler._layer
    U = handler.model.transformer.h[layer].mlp.c_fc.weight.detach().float().T
    D = handler.model.transformer.h[layer].mlp.c_proj.weight.detach().float().T

    # Process into the final k_examples list
    print(f"Computing real reference vectors for layer {layer-1}")
    k_examples = []
    batch_size = 200
    for i in tqdm(range(vector_count)[::batch_size]):
        for k in gather_k(handler, ("", "", "", ""), N=batch_size, layer_k=layer):
            k_examples.append(k.float().detach())
    
    # for i in range(vector_count):
    #     k_examples.append(gather_k(handler, ("", "", "", ""), 2).float())
    
    mean, std = vector_space_stats(k_examples, D, U.T, D.T)
    print(f"Ref Layer: mean {mean} std {std}")

    # print(f"Computing random vectors")
    # sm = get_second_moment(handler)
    # k_examples = []
    # for i in range(vector_count):
    #     x = sm @ ((torch.randn(U.shape[0], device=handler.device)*std) + mean)
    #     # x = x / x.norm()
    #     x = handler.model.transformer.h[handler._layer].mlp.act(x)
    #     k_examples.append(x)


    ORIG_global_mean = 0.0
    ROME_global_mean = 0.0
    
    significant = 0
    for i, (new_W, _, prompt_dict, success) in enumerate(batch_intervention_generator(handler)):

        print(f"ROME result: {success}")

        U = handler.model.transformer.h[layer].mlp.c_fc.weight.detach().float().T
        D = handler.model.transformer.h[layer].mlp.c_proj.weight.detach().float().T

        # handler.model.transformer.h[layer].mlp.c_proj.weight = torch.nn.Parameter(new_W)


        # baseline = approximate_baseline(
        #     handler, 
        #     vector_count,
        #     window_size,
        #     polynom_size,
        #     mean,
        #     std,
        #     loss_fn, 
        #     lr,  
        #     opt_steps, 
        #     threshold, 
        #     threshold_fn,
        #     k_examples
        # )
        # handler.model.transformer.h[layer].mlp.c_proj.weight = torch.nn.Parameter(D.to(handler.dtype).T)

        # k_examples = []
        # batch_size = 200
        # print(f"Computing vectors")
        # for _ in tqdm(range(vector_count)[::batch_size]):
        #     for k in gather_k(handler, ("", "", "", ""), N=batch_size, layer_k=layer):
        #         k_examples.append(k.float().detach())

        # mean, std = vector_space_stats(k_examples, D, U.T, D.T)
        # print(f"Ref Layer: mean {mean} std {std}")

        new_D = new_W.float().T.detach()

        U_inv = torch.linalg.pinv(U)
        D_inv = torch.linalg.pinv(D)
        new_D_inv = torch.linalg.pinv(new_D)

        transform_M_o = U.T @ D.T @ D
        transform_M_r = U.T @ new_D.T @ new_D

        # transform_M_o = U_inv @ D_inv @ D
        # transform_M_r = U_inv @ new_D_inv @ new_D

        # transform_M_o = D
        # transform_M_r = new_D

        ORIG_acc = []
        ROME_acc = []

        for x in tqdm(k_examples):
            ORIG_i = vector_space_scouting(
                x.clone().detach().requires_grad_(True),
                transform_M_o, 
                mean,
                std,
                loss_fn, 
                lr,  
                opt_steps, 
                threshold, 
                threshold_fn,
            )
            ORIG_acc.append(ORIG_i)

            ROME_i = vector_space_scouting(
                x.clone().detach().requires_grad_(True),
                transform_M_r,
                mean, 
                std, 
                loss_fn, 
                lr,
                opt_steps, 
                threshold, 
                threshold_fn,
            )
            ROME_acc.append(ROME_i)

        ORIG_s = torch.Tensor(ORIG_acc)
        ROME_s = torch.Tensor(ROME_acc)
        ROME_np = np.array(ROME_acc)
        ORIG_np = np.array(ORIG_acc)

        # 1. Your measured data for the center layer
        measured_data = np.array(ROME_np)

        # 2. Your approximated baseline from the previous step

        baseline = np.array(ORIG_np)
        # 3. Perform One-Sample t-test
        # t_stat, p_value = stats.ttest_1samp(measured_data, baseline)
        t_stat, p_value = stats.ttest_ind(measured_data, baseline)

        print("ROME")
        print(f"Measured Mean: {measured_data.mean():.2f}")
        print(f"Original model baseline: {baseline.mean():.2f}")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.4f}")

        # 4. Interpretation
        alpha = 0.05
        if p_value < alpha:
            print("Result: Statistically Significant (Reject H0)")
            print("The middle layer performs differently than the trend suggests.")
        else:
            print("Result: Not Significant (Fail to reject H0)")
            print("The middle layer follows the expected architectural trend.")

        # 4. Interpretation
        alpha = 0.05
        if p_value < alpha:
            significant += 1
            print("Result: Statistically Significant (Reject H0)")
            print("The middle layer performs differently than the trend suggests.")
        else:
            print("Result: Not Significant (Fail to reject H0)")
            print("The middle layer follows the expected architectural trend.")


        writer.add_scalar("ROME mean", ROME_s.mean().item(), i)
        writer.add_scalar("ROME std", ROME_s.std().item(), i)


        ORIG_global_mean += ORIG_s.mean().item()
        ROME_global_mean += ROME_s.mean().item()

        if i+1 == total_edits:
            break
    
    print(f"GLOBAL ORIG: {ORIG_s.mean().item()} ROME: {ROME_s.mean().item()}")

    print("")
    print(f"Significant: {significant} Total: {total_edits} Rate: {significant/total_edits}")
    # yield np.abs((ORIG_global_mean - ROME_global_mean)/total_edits)
    yield p_value