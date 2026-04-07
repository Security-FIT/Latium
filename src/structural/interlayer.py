from typing import Dict, Optional
import torch
import numpy as np

from src.utils import gpu_svdvals

EPS = 1e-10


def compute_layer_features(W: torch.Tensor) -> Dict[str, float]:
    """SVD-based feature vector for one weight matrix. See short-docs.md."""
    W_float = W.float()
    S = gpu_svdvals(W)
    S_sq = S ** 2
    total_energy = S_sq.sum() + EPS
    norms = W_float.norm(dim=1)

    S_norm = S / (S.sum() + EPS)
    S_clamped = S_norm.clamp(min=EPS)
    entropy = -(S_clamped * torch.log(S_clamped)).sum()

    S_centered = S - S.mean()
    S_std = S.std() + EPS
    kurtosis = ((S_centered / S_std) ** 4).mean().item() - 3.0

    return {
        'top1_energy': (S_sq[0] / total_energy).item(),
        'top5_energy': (S_sq[:5].sum() / total_energy).item(),
        'spectral_gap': (S[0] / S[1].clamp(min=EPS)).item(),
        'effective_rank': torch.exp(entropy).item(),
        'sv_kurtosis': kurtosis,
        'norm_cv': (norms.std() / (norms.mean() + EPS)).item(),
        'norm': W_float.norm().item(),
    }


def layer_block_analysis(weights: Dict[int, torch.Tensor], n_blocks: int = 3,
                         layer_features: Optional[Dict[int, Dict[str, float]]] = None) -> Dict:
    layer_indices = sorted(weights.keys())
    n_layers = len(layer_indices)
    block_size = max(1, n_layers // n_blocks)

    if layer_features is None:
        layer_features = {idx: compute_layer_features(W) for idx, W in weights.items()}

    all_metrics = tuple(layer_features[layer_indices[0]].keys())

    blocks = {}
    layer_to_block = {}
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size if b < n_blocks - 1 else n_layers
        block_layers = layer_indices[start:end]
        
        for idx in block_layers:
            layer_to_block[idx] = b
        block_stats = {}
        for metric in all_metrics:
            vals = [layer_features[idx][metric] for idx in block_layers]
            block_stats[f'{metric}_mean'] = float(np.mean(vals))
            block_stats[f'{metric}_std'] = float(np.std(vals)) + EPS

        blocks[b] = {'layers': block_layers, **block_stats}

    z_scores = {}
    for idx in layer_indices:
        b = layer_to_block[idx]
        z_scores[idx] = {}
        for metric in all_metrics:
            val = layer_features[idx][metric]
            mean = blocks[b][f'{metric}_mean']
            std = blocks[b][f'{metric}_std']
            z_scores[idx][metric] = (val - mean) / std

    
    return {
        'layer_features': layer_features,
        'blocks': blocks,
        'z_scores': z_scores,
    }

def neighbor_transition_analysis(weights: Dict[int, torch.Tensor],
                                  layer_features: Optional[Dict[int, Dict[str, float]]] = None) -> Dict:
    
    layer_indices = sorted(weights.keys())
    if len(layer_indices) < 2:
        return {'layer_features': {}, 'transitions': {}, 'transition_z_scores': {}, 'per_metric_deltas': {}}

    if layer_features is None:
        layer_features = {idx: compute_layer_features(W) for idx, W in weights.items()}
    all_metrics = tuple(layer_features[layer_indices[0]].keys())

    per_metric_deltas = {}
    for i in range(len(layer_indices) - 1):
        curr, nxt = layer_indices[i], layer_indices[i + 1]
        per_metric_deltas[(curr, nxt)] = {
            m: layer_features[nxt][m] - layer_features[curr][m] for m in all_metrics
        }

    metric_arrays = {m: np.array([d[m] for d in per_metric_deltas.values()]) for m in all_metrics}
    metric_stats = {}
    for m in all_metrics:
        arr = metric_arrays[m]
        metric_stats[m] = (float(np.mean(arr)), float(np.std(arr)) + EPS)

    transitions = {}
    for key, deltas in per_metric_deltas.items():
        z_deltas = [(deltas[m] - metric_stats[m][0]) / metric_stats[m][1] for m in all_metrics]
        transitions[key] = float(np.sqrt(sum(d ** 2 for d in z_deltas)))

    trans_vals = list(transitions.values())
    mean_t, std_t = float(np.mean(trans_vals)), float(np.std(trans_vals)) + EPS
    transition_z_scores = {k: (v - mean_t) / std_t for k, v in transitions.items()}

    return {
        'layer_features': layer_features,
        'transitions': transitions,
        'transition_z_scores': transition_z_scores,
        'per_metric_deltas': per_metric_deltas,
        'transition_mean': mean_t,
        'transition_std': std_t,
    }

def leave_one_out_variance(weights: Dict[int, torch.Tensor],
                           layer_features: Optional[Dict[int, Dict[str, float]]] = None) -> Dict:
    if len(weights) < 2:
        return {'layer_values': {}, 'full_variance': 0.0, 'variance_reductions': {}, 'influence_z_scores': {}}

    if layer_features is None:
        layer_features = {idx: compute_layer_features(W) for idx, W in weights.items()}

    values = {idx: layer_features[idx]['top1_energy'] for idx in weights}
    
    all_vals = np.array(list(values.values()))
    full_var = np.var(all_vals)
    
    reductions = {}
    for idx in values:
        remaining = np.array([v for i, v in values.items() if i != idx])
        reduced_var = float(np.var(remaining)) if len(remaining) > 1 else 0.0
        reductions[idx] = full_var - reduced_var
    
    red_vals = list(reductions.values())
    mean_r, std_r = float(np.mean(red_vals)), float(np.std(red_vals)) + EPS
    influence_z = {idx: (v - mean_r) / std_r for idx, v in reductions.items()}
    
    return {
        'layer_values': values,
        'full_variance': full_var,
        'variance_reductions': reductions,
        'influence_z_scores': influence_z,
    }



def cross_layer_fingerprint(weights: Dict[int, torch.Tensor], n_sv: int = 20) -> Dict:
    layer_indices = sorted(weights.keys())
    n = len(layer_indices)

    if n < 2:
        return {'fingerprints': {}, 'distance_matrix': [], 'avg_distances': {}, 'distance_z_scores': {}}

    fingerprints = {}
    fp_matrix = []
    for idx in layer_indices:
        S = gpu_svdvals(weights[idx])[:n_sv]
        S_norm = (S / (S.sum() + EPS)).cpu().numpy()
        fingerprints[idx] = S_norm
        fp_matrix.append(S_norm)
    
    fp_matrix = np.array(fp_matrix)
    norms = np.linalg.norm(fp_matrix, axis=1, keepdims=True) + EPS
    fp_normed = fp_matrix / norms
    cosine_sim = fp_normed @ fp_normed.T
    dist_matrix = 1 - cosine_sim
    np.fill_diagonal(dist_matrix, 0)
    avg_dist = dist_matrix.sum(axis=1) / (n - 1)
    avg_distances = {layer_indices[i]: float(avg_dist[i]) for i in range(n)}
    
    mean_d, std_d = float(avg_dist.mean()), float(avg_dist.std()) + EPS
    distance_z_scores = {layer_indices[i]: (avg_dist[i] - mean_d) / std_d for i in range(n)}
    
    return {
        'fingerprints': {k: v.tolist() for k, v in fingerprints.items()},
        'distance_matrix': dist_matrix.tolist(),
        'layer_order': layer_indices,
        'avg_distances': avg_distances,
        'distance_z_scores': distance_z_scores,
    }


def collect_all_interlayer_data(weights: Dict[int, torch.Tensor], n_blocks: int = 3) -> Dict:
    layer_features = {idx: compute_layer_features(W) for idx, W in weights.items()}

    return {
        'block_analysis': layer_block_analysis(weights, n_blocks, layer_features=layer_features),
        'neighbor_transitions': neighbor_transition_analysis(weights, layer_features=layer_features),
        'leave_one_out': leave_one_out_variance(weights, layer_features=layer_features),
        'fingerprint': cross_layer_fingerprint(weights),
    }


