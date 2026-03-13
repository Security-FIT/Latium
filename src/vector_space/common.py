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

from src.rome.common import gather_k


def dense_z_score_loss(z, mean, std, p=4):
    z_scores = (z - mean) / (std + 1e-8)
    loss = -torch.mean(torch.abs(z_scores ** p))
    return loss

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
    return pr


LOSS_FUNCTIONS = {
    "z_score_outlier": (lambda loss,z,t: loss > t*1.0*z.shape[-1], z_score_outlier_loss),
    "dense_z_score": (lambda loss,z,t: loss > t*1.0*z.shape[-1], dense_z_score_loss),
    # "max_focus": (1.5, max_focus_loss),
    "energy_maximization": (lambda loss,z,t: loss > t*1.5*z.shape[-1], energy_maximization_loss),
    "density": (lambda loss,z,t: loss > t*0.01*z.shape[-1], density_loss)
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

        if threshold_fn(loss, z, threshold):
            break
        
        loss.backward()
        opt.step()

    # return i
    return loss.item()

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
        threshold_fn
    ):
    # 1. Define the window bounds
    center_layer = handler._layer
    max_layers = len(handler.model.transformer.h)

    layer_results = []
    acc = []
    # 2. Iterate through the window
    # We use max() and min() to stay within actual model bounds
    for layer_idx in range(center_layer - window_size, center_layer + window_size + 1):
        k_examples_layer = []
        for i in range(vector_count):
            k_examples_layer.append(gather_k(handler, ("", "", "", ""), 2, layer_k=layer_idx).float())
        # Access the specific layer weights
        U = handler.model.transformer.h[layer_idx].mlp.c_fc.weight.detach().float().T
        D = handler.model.transformer.h[layer_idx].mlp.c_proj.weight.detach().float().T

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

    center_idx = handler._layer 

    x_indices = []
    y_means = []
    for i in range(window_size*2 + 1):
        if i == handler._layer:
            continue
        x_indices.append(i)
        y_means.append(np.mean(layer_results[i]))

    x_observed = np.array(x_indices)
    y_observed = np.array(y_means)

    coefficients = np.polyfit(x_observed, np.log(y_observed), handler._layer)
    poly_func = np.poly1d(coefficients)

    # 3. Predict the baseline for the middle layer (x=0)
    center_val = np.exp(poly_func(polynom_size))
    predicted_center_val = center_val
    print(f"Approximated result for Layer {center_idx}: {predicted_center_val:.2f} steps")

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
    U = handler.model.transformer.h[layer-1].mlp.c_fc.weight.detach().float().T
    D = handler.model.transformer.h[layer-1].mlp.c_proj.weight.detach().float().T

    # Process into the final k_examples list
    k_examples = []
    for i in range(vector_count):
        k_examples.append(gather_k(handler, ("", "", "", ""), 2).float())

    mean, std = vector_space_stats(k_examples, D, U.T, D.T)
    print(f"Ref Layer: mean {mean} std {std}")

    ORIG_global_mean = 0.0
    ROME_global_mean = 0.0



    baseline = approximate_baseline(
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
        threshold_fn
    )

    for i, (new_W, _, prompt_dict, success) in enumerate(batch_intervention_generator(handler)):

        U = handler.model.transformer.h[layer].mlp.c_fc.weight.detach().float().T
        D = handler.model.transformer.h[layer].mlp.c_proj.weight.detach().float().T

        new_D = new_W.float().T.detach()

        transform_M_o = U.T @ D.T @ D
        transform_M_r = U.T @ new_D.T @ new_D

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
        measured_data = np.array(ORIG_np)

        # 2. Your approximated baseline from the previous step

        # 3. Perform One-Sample t-test
        t_stat, p_value = stats.ttest_1samp(measured_data, baseline)

        print("ORIG")
        print(f"Measured Mean: {measured_data.mean():.2f}")
        print(f"Predicted Baseline: {baseline:.2f}")
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





        # 1. Your measured data for the center layer
        measured_data = np.array(ROME_np)

        # 3. Perform One-Sample t-test
        t_stat, p_value = stats.ttest_1samp(measured_data, baseline)

        print("ROME")
        print(f"Measured Mean: {measured_data.mean():.2f}")
        print(f"Predicted Baseline: {baseline:.2f}")
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


        writer.add_scalar("ROME mean", ROME_s.mean().item(), i)
        writer.add_scalar("ROME std", ROME_s.std().item(), i)


        ORIG_global_mean += ORIG_s.mean().item()
        ROME_global_mean += ROME_s.mean().item()

        if i+1 == total_edits:
            break
    
    print(f"GLOBAL ORIG: {ORIG_s.mean().item()} ROME: {ROME_s.mean().item()}")
    yield np.abs((ORIG_global_mean - ROME_global_mean)/total_edits)

