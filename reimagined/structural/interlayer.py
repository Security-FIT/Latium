from typing import Dict
import torch
import numpy as np

EPS = 1e-10


def compute_layer_features(W: torch.Tensor) -> Dict[str, float]:
    """Compute SVD-based features for a single layer.
    
    Returns dict with rich metrics for analysis.
    """
    W_float = W.float()
    S = torch.linalg.svdvals(W_float)
    S_sq = S ** 2
    total_energy = S_sq.sum() + EPS
    norms = W_float.norm(dim=1)
    
    # Effective rank via entropy
    S_norm = S / (S.sum() + EPS)
    S_clamped = S_norm.clamp(min=EPS)
    entropy = -(S_clamped * torch.log(S_clamped)).sum()
    
    # Gap cascade ratio: first gap vs average of next gaps
    gaps = S[:-1] / S[1:].clamp(min=EPS)
    gap_cascade = (gaps[0] / (gaps[1:10].mean() + EPS)).item() if len(gaps) > 10 else gaps[0].item()
    
    # SV kurtosis (excess)
    S_centered = S - S.mean()
    S_std = S.std() + EPS
    kurtosis = ((S_centered / S_std) ** 4).mean().item() - 3.0
    
    return {
        'top1_energy': (S_sq[0] / total_energy).item(),
        'top5_energy': (S_sq[:5].sum() / total_energy).item(),
        'spectral_gap': (S[0] / S[1].clamp(min=EPS)).item(),
        'gap_cascade_ratio': gap_cascade,
        'effective_rank': torch.exp(entropy).item(),
        'sv_kurtosis': kurtosis,
        'norm_cv': (norms.std() / (norms.mean() + EPS)).item(),
        'norm': W_float.norm().item(),
        'condition_number': (S[0] / S[-1].clamp(min=EPS)).item(),
    }


def layer_block_analysis(weights: Dict[int, torch.Tensor], n_blocks: int = 3) -> Dict:
    """Compare statistics across layer blocks (early/middle/late)."""
    layer_indices = sorted(weights.keys())
    n_layers = len(layer_indices)
    block_size = max(1, n_layers // n_blocks)
    
    # cache features for all layers (single SVD per layer)
    layer_features = {idx: compute_layer_features(W) for idx, W in weights.items()}
    
    all_metrics = ('top1_energy', 'top5_energy', 'spectral_gap', 'gap_cascade_ratio', 'effective_rank', 'sv_kurtosis', 'norm_cv', 'norm', 'condition_number')
    
    
    # build blocks
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

def neighbor_transition_analysis(weights: Dict[int, torch.Tensor], z_threshold: float = 2.0) -> Dict:
    """Detect layers with unusual transitions to neighbors."""
    layer_indices = sorted(weights.keys())
    if len(layer_indices) < 2:
        return {'layer_features': {}, 'transitions': {}, 'transition_z_scores': {}, 'per_metric_deltas': {}}
    
    features = {idx: compute_layer_features(W) for idx, W in weights.items()}
    all_metrics = ('top1_energy', 'top5_energy', 'spectral_gap', 'gap_cascade_ratio',
                   'effective_rank', 'sv_kurtosis', 'norm_cv', 'norm', 'condition_number')
    
    # Compute transitions
    transitions = {}
    per_metric_deltas = {}
    for i in range(len(layer_indices) - 1):
        curr, nxt = layer_indices[i], layer_indices[i + 1]
        key = (curr, nxt)
        
        deltas = {m: features[nxt][m] - features[curr][m] for m in all_metrics}
        per_metric_deltas[key] = deltas
        transitions[key] = sum(abs(d) for d in deltas.values())
    
    # Z-scores for transitions
    trans_vals = list(transitions.values())
    mean_t, std_t = float(np.mean(trans_vals)), float(np.std(trans_vals)) + EPS
    transition_z_scores = {k: (v - mean_t) / std_t for k, v in transitions.items()}
    
    return {
        'layer_features': features,
        'transitions': transitions,
        'transition_z_scores': transition_z_scores,
        'per_metric_deltas': per_metric_deltas,
        'transition_mean': mean_t,
        'transition_std': std_t,
    }

def leave_one_out_variance(weights: Dict[int, torch.Tensor]) -> Dict:
    """Find layer whose removal reduces variance most."""
    if len(weights) < 2:
        return {'layer_values': {}, 'full_variance': 0.0, 'variance_reductions': {}, 'influence_z_scores': {}}

    # compute top1_energy for all layers
    values = {}
    for idx, W in weights.items():
        S = torch.linalg.svdvals(W.float())
        S_sq = S ** 2
        values[idx] = (S_sq[0] / (S_sq.sum() + EPS)).item()
    
    all_vals = np.array(list(values.values()))
    full_var = np.var(all_vals)
    
    # lleave-one-out variance reduction
    reductions = {}
    for idx in values:
        remaining = np.array([v for i, v in values.items() if i != idx])
        reduced_var = float(np.var(remaining)) if len(remaining) > 1 else 0.0
        reductions[idx] = full_var - reduced_var
    
    # Z-scores for influence
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
    """Compare spectral fingerprints between all layers."""
    layer_indices = sorted(weights.keys())
    n = len(layer_indices)
    
    if n < 2:
        return {'fingerprints': {}, 'distance_matrix': [], 'avg_distances': {}, 'distance_z_scores': {}}
    
    # Get normalized SV profiles (fingerprints)
    fingerprints = {}
    fp_matrix = []
    for idx in layer_indices:
        S = torch.linalg.svdvals(weights[idx].float())[:n_sv]
        S_norm = (S / (S.sum() + EPS)).cpu().numpy()
        fingerprints[idx] = S_norm
        fp_matrix.append(S_norm)
    
    fp_matrix = np.array(fp_matrix)  # (n_layers, n_sv)
    
    # Cosine distance matrix: 1 - cosine_similarity
    # cosine_sim = (A @ B.T) / (||A|| * ||B||)
    norms = np.linalg.norm(fp_matrix, axis=1, keepdims=True) + EPS
    fp_normed = fp_matrix / norms
    cosine_sim = fp_normed @ fp_normed.T
    dist_matrix = 1 - cosine_sim
    np.fill_diagonal(dist_matrix, 0)  # Self-distance = 0
    
    # Average distance per layer
    avg_dist = dist_matrix.sum(axis=1) / (n - 1)
    avg_distances = {layer_indices[i]: float(avg_dist[i]) for i in range(n)}
    
    # Z-scores
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
    """Run all inter-layer analyses and return combined data for notebook analysis"""
    return {
        'block_analysis': layer_block_analysis(weights, n_blocks),
        'neighbor_transitions': neighbor_transition_analysis(weights),
        'leave_one_out': leave_one_out_variance(weights),
        'fingerprint': cross_layer_fingerprint(weights),
    }


