from omegaconf import DictConfig
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
import hydra
import json

from src.handlers.rome import ModelHandler
from src.rome.rome import batch_intervention_generator


def cross_layer_reciprocity_detector(model, case_id, target_layer_idx: int, window: int = 4) -> Dict:
    """
    Detect ROME rank-1 updates via reciprocity breakdown.
    Sensitive to: reciprocity correlation drop, effective rank imbalance, singular value anomalies.
    """
    
    # Extract FFN projections from multiple layers
    all_metrics = []
    
    for i in range(max(0, target_layer_idx - window), min(len(model.transformer.h), target_layer_idx+1)):
        try:
            layer = model.transformer.h[i]
            W_up = layer.mlp.c_fc.weight.detach().float().cpu()
            W_down = layer.mlp.c_proj.weight.detach().float().cpu()
            
            _, s_up, _ = torch.linalg.svd(W_up, full_matrices=False)
            _, s_down, _ = torch.linalg.svd(W_down, full_matrices=False)
            s_up = s_up[:50]  # Top-50 for computational efficiency
            s_down = s_down[:50]
            
            all_metrics.append({
                'layer': i,
                's_up': s_up,
                's_down': s_down,
            })
            
        except Exception as e:
            print(f"Layer {i}: skip ({e})")
            continue
    
    if not all_metrics:
        return {}
    
    target_metrics = [m for m in all_metrics if m['layer'] == target_layer_idx][0]
    neighbors = [m for m in all_metrics if m['layer'] != target_layer_idx]
    
    if len(neighbors) < 2:
        print("Need >=2 reference layers!")
        return {}
    
    
    # ============ VISUALIZATION ============
    fig = plt.figure(figsize=(18, 10))

    # Singular value comparison (target vs neighbors)
    ax5 = plt.axes()
    target_sv_ratios = target_metrics['s_down'] / target_metrics['s_up']
    
    ax5.semilogy(range(len(target_sv_ratios)), target_sv_ratios[:], 'r-o', 
                lw=2, markersize=6, label=f'Target L{target_layer_idx}', zorder=5)
    for i, n in enumerate(neighbors):
        ratios = n['s_down'] / n['s_up']
        ax5.semilogy(range(len(ratios)), ratios, 'b--', alpha=0.3, linewidth=1)
    ax5.plot([], [], 'b--', alpha=0.3, linewidth=1, label='Neighbors')
    ax5.set_xlabel('Singular Index'); ax5.set_ylabel('σ_down / σ_up')
    ax5.set_title('SINGULAR VALUE RATIO\n(ROME creates spike)', fontweight='bold')
    ax5.legend(loc='best'); ax5.grid(True, alpha=0.3, which='both')
    
    status = "Dispair"
    plt.suptitle(f'ROME Detection: Layer {target_layer_idx} | Status: {status}', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f"./data/figs/gpt2-large-det/{case_id}.png")
    
    
    export = []
    for m in all_metrics:
        export.append({
            'layer': m['layer'],
            's_up': m['s_up'].numpy().tolist(),
            's_down': m['s_down'].numpy().tolist()
            })
    
    return {
        'target_layer': target_layer_idx,
        'status': status,
        # 'metrics': target_metrics,
        'all_metrics': export,
    }

def run_detection(handler, layer, case_id, title, window=4) -> None:
    print("\n" + "="*80)
    print(f"LAYER {layer}")
    print(f"{title}")
    print("="*80)
    metrics = cross_layer_reciprocity_detector(handler.model, case_id, target_layer_idx=layer, window=window)
    with open(f"./data/figs/gpt2-large-det/{case_id}.json", 'w') as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="src/config", config_name="config")
    def main(cfg: DictConfig) -> None:
        handler = ModelHandler(cfg)
        # fact_tuple = getattr(cfg, 'fact_tuple', ("{} is in", "The Eiffel Tower", " Rome", " Paris"))
        run_detection(handler, layer=handler._layer, case_id=-1, title="BEFORE ROME UPDATE: Original weights (should be NORMAL)")
        counter = 0
        for new_W, old_W, prompt_dict in batch_intervention_generator(handler):
            if counter >= 10:
                break
            handler._get_module(handler._layer_name_template.format(handler._layer)).weight = torch.nn.Parameter(new_W.detach())
            run_detection(handler, layer=handler._layer, case_id=prompt_dict.case_id, title="AFTER ROME UPDATE: Weights modified (should detect ROME)")
            handler._get_module(handler._layer_name_template.format(handler._layer)).weight = torch.nn.Parameter(old_W.detach())
            counter += 1

    main()
