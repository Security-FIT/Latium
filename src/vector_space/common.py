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
    # acc = []
    # for _ in tqdm(range(vector_count)):
        # x_raw = torch.randn(U.shape[1], requires_grad=True, device="cuda")
    opt = Adam([x_raw], lr=lr)
    i = 0.0
    while i < opt_steps:
        i += 1.0
        opt.zero_grad()

        # x = (x_raw * std) + mean
        x = x_raw
        # y = D @ (U @ x)
        # z = U_inv @ (D_inv @ y)
        z = M @ x
        
        loss = loss_fn(z, mean, std)

        if threshold_fn(loss, z, threshold):
            break
        
        loss.backward()
        opt.step()

        # acc.append(i)

    # fig = plt.figure(figsize=(16,10))
    # ax = plt.axes()
    # ax.plot(x.cpu().detach(), color="r")

    # plt.savefig(f'data/figs/involution/{name}.png')
    # plt.close()

    # it_mean = torch.mean(torch.Tensor(acc)).item()
    # it_std = torch.std(torch.Tensor(acc)).item()
    # print(title, it_mean, it_std, iteration)
    # writer.add_scalar(title.format("mean"), it_mean, iteration)
    # writer.add_scalar(title.format("std"), it_std, iteration)

    # return it_mean, it_std
    return i

def vector_space_mapping(handler, U, D, U_inv, D_inv):
    zs = []
    k_examples = [
        ("Mother Tongue: Danielle Darrieux", gather_k(handler, fact_tuple=("The mother tongue of{} is", " Danielle Darrieux", " English", " French")).to(handler.device).float().detach()),
        ("Capital: Australia", gather_k(handler, fact_tuple=("The capital of{} is", " Australia", " Sydney", " Canberra")).to(handler.device).float()),
        ("Company: Nintendo", gather_k(handler, fact_tuple=("The headquarters of{} is in", " Nintendo", " Tokyo", " Kyoto")).to(handler.device).float()),
        ("Element: Silver", gather_k(handler, fact_tuple=("The chemical symbol for{} is", " Silver", " Si", " Ag")).to(handler.device).float()),
        ("Invention: Telephone", gather_k(handler, fact_tuple=("The telephone was invented by{}", " Alexander Graham Bell", " Thomas Edison", " Alexander Graham Bell")).to(handler.device).float()),
        ("Language: Brazil", gather_k(handler, fact_tuple=("The official language of{} is", " Brazil", " Spanish", " Portuguese")).to(handler.device).float()),
        ("Author: Hamlet", gather_k(handler, fact_tuple=("The play 'Hamlet' was written by{}", " William Shakespeare", " Christopher Marlowe", " William Shakespeare")).to(handler.device).float()),
        ("Biology: Whale", gather_k(handler, fact_tuple=("A whale is a type of{}", " mammal", " fish", " mammal")).to(handler.device).float()),
        ("Planet: Distance", gather_k(handler, fact_tuple=("The planet closest to the Sun is{}", " Mercury", " Venus", " Mercury")).to(handler.device).float()),
        ("Mythology: Zeus", gather_k(handler, fact_tuple=("In mythology, Zeus is the king of the{} gods", " Greek", " Roman", " Greek")).to(handler.device).float()),
        ("Currency: Japan", gather_k(handler, fact_tuple=("The currency used in{} is the Yen", " Japan", " China", " Japan")).to(handler.device).float())
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
    lr = getattr(vector_space_cfg, "lr", 1.0)
    
    loss_fn = getattr(vector_space_cfg, "loss_fn", None)
    threshold = getattr(vector_space_cfg, "threshold", 1)
    threshold_fn, loss_fn = LOSS_FUNCTIONS[loss_fn]

    writer = SummaryWriter(f'runs/involution_{int(time.time())}')

    layer=handler._layer
    U = handler.model.transformer.h[layer-1].mlp.c_fc.weight.detach().float().T
    D = handler.model.transformer.h[layer-1].mlp.c_proj.weight.detach().float().T

    mean, std = vector_space_mapping(handler, U, D, U.T, D.T)
    print(f"Ref Layer: mean {mean} std {std}")

    ORIG_global_mean = 0.0
    ROME_global_mean = 0.0

    # Process into the final k_examples list
    k_examples = []
    for i in range(vector_count):
        k_examples.append(gather_k(handler, ("", "", "", ""), 2).float())
    # for x in k_examples:
        # tensor = gather_k(handler, prompt_tuple).to(handler.device).float().detach()
        # k_examples.append((label, tensor))


# 770.15 744.12 804.41 370.63 300.7 1.02 1.0


    # 1. Define the window bounds
    center_layer = handler._layer
    window_size = 3
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

    # Define your layer window (assuming center is 13)
    center_idx = handler._layer 
    x_observed = np.array([0, 1, 2, 4, 5, 6])#.reshape(-1, 1)

    # Map your results to these indices
    # Replace these with your actual ORIG_acc averages for those layers
    y_observed = np.array([
        np.mean(layer_results[0]), 
        np.mean(layer_results[1]),
        np.mean(layer_results[2]), 
        np.mean(layer_results[4]), 
        np.mean(layer_results[5]),
        np.mean(layer_results[6])
    ])


    coefficients = np.polyfit(x_observed, np.log(y_observed), 2)
    poly_func = np.poly1d(coefficients)

    # 3. Predict the baseline for the middle layer (x=0)
    center_val = np.exp(poly_func(3))


    # from scipy.optimize import curve_fit

    # def sigmoid(x, L, k, x0, b):
    #     return L / (1 + np.exp(-k * (x - x0))) + b

    # # Initial guesses are important for Sigmoids!
    # # [max_y, steepness, middle_layer, offset]
    # p0 = [max(y_observed), 1.0, np.median(x_observed), min(y_observed)]

    # popt, _ = curve_fit(sigmoid, x_observed.flatten(), y_observed, p0=p0, maxfev=10000)

    # # To predict the center value:
    # center_val = sigmoid(center_idx, *popt)

    # # 1. Transform X to include squared terms (x^2)
    # poly = PolynomialFeatures(degree=window_size*2-1)
    # x_poly = poly.fit_transform(x_observed)

    # # 2. Fit the regression to the 4 known points
    # model = LinearRegression()
    # model.fit(x_poly, y_observed)

    # # 3. Predict the missing center
    # x_center = np.array([[3]])
    # x_center_poly = poly.transform(x_center)
    # predicted_center_val = model.predict(x_center_poly)[0]
    predicted_center_val = center_val

    print(f"Approximated result for Layer {center_idx}: {predicted_center_val:.2f} steps")

    for i, (new_W, _, prompt_dict, success) in enumerate(batch_intervention_generator(handler)):
        # U11 = handler.model.transformer.h[layer-1].mlp.c_fc.weight.detach().float().T
        # D11 = handler.model.transformer.h[layer-1].mlp.c_proj.weight.detach().float().T

        U = handler.model.transformer.h[layer].mlp.c_fc.weight.detach().float().T
        D = handler.model.transformer.h[layer].mlp.c_proj.weight.detach().float().T

        # U13 = handler.model.transformer.h[layer+1].mlp.c_fc.weight.detach().float().T
        # D13 = handler.model.transformer.h[layer+1].mlp.c_proj.weight.detach().float().T

        # vector_space_scouting(U11, D11, U11.T, D11.T, mean, std, vector_count, opt_steps, threshold/10.0, writer, threshold, f"ORIG layer 11 avg steps")

        new_D = new_W.float().T.detach()

        transform_M_o = U.T @ D.T @ D #@ U
        transform_M_r = U.T @ new_D.T @ new_D #@ U

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
        # vector_space_scouting(U13, D13, U13.T, D13.T, mean, std, vector_count, opt_steps, threshold/10.0, writer, threshold, f"ORIG layer 13 avg steps")


        ORIG_s = torch.Tensor(ORIG_acc)
        ROME_s = torch.Tensor(ROME_acc)
        ROME_np = np.array(ROME_acc)
        ORIG_np = np.array(ORIG_acc)

        # 1. Your measured data for the center layer
        measured_data = np.array(ORIG_np)

        # 2. Your approximated baseline from the previous step
        baseline = predicted_center_val 

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

        # 2. Your approximated baseline from the previous step
        baseline = predicted_center_val 

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




        # writer.add_scalar("ORIG mean", ORIG_s.mean().item(), i)
        # writer.add_scalar("ORIG std", ORIG_s.std().item(), i)

        writer.add_scalar("ROME mean", ROME_s.mean().item(), i)
        writer.add_scalar("ROME std", ROME_s.std().item(), i)

        # print(f"ORIG: {ORIG_s.mean().item()} ROME: {ROME_s.mean().item()}")

        ORIG_global_mean += ORIG_s.mean().item()
        ROME_global_mean += ROME_s.mean().item()

        # 1. Create the plot using your result lists
        # We use 'alpha=0.5' so both distributions are visible if they overlap
        # plt.hist(ORIG_acc, bins=50, alpha=0.5, label='ORIG_acc', color='blue', edgecolor='black')
        # plt.hist(ROME_acc, bins=50, alpha=0.5, label='ROME_acc', color='orange', edgecolor='black')

        # 2. Add labeling and styling
        # plt.title('Optimization Step Distribution: ORIG vs. ROME')
        # plt.xlabel('Steps to reach threshold ($1$ to $1000$)')
        # plt.ylabel('Frequency')
        # plt.legend(loc='upper right')
        # plt.grid(axis='y', alpha=0.3)

        # 3. Save the result
        # plt.savefig('optimization_steps_histogram.png')
        # plt.show()

        if i+1 == total_edits:
            break
    
    print(f"GLOBAL ORIG: {ORIG_s.mean().item()} ROME: {ROME_s.mean().item()}")
    yield np.abs((ORIG_global_mean - ROME_global_mean)/total_edits)

