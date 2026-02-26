import numpy as np
from omegaconf import DictConfig
import torch
import json
import hydra
from tqdm import tqdm

from src.handlers.rome import ModelHandler
from src.rome.rome import batch_intervention_generator


if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="src/config", config_name="config")
    def main(cfg: DictConfig) -> None:
        handler = ModelHandler(cfg)
        up_proj = handler.model.transformer.h[handler._layer].mlp.c_fc.weight.detach().float().cpu().numpy().T
        down_proj_baseline = handler._get_module(handler._layer_name_template.format(handler._layer)).weight.detach().float().cpu().numpy()
        correlations = [np.corrcoef(up_proj[:, i], down_proj_baseline[:, i])[0, 1] for i in range(up_proj.shape[1])]
        with open(f"./data/figs/gpt2-large-corr/{-1}.npy", 'wb') as f:
            np.save(f, correlations)

        counter = 0
        for new_W, _, prompt_dict in tqdm(batch_intervention_generator(handler)):
            if counter >= 10:
                break
            corr = np.array([np.corrcoef(up_proj[:, i], new_W.detach().float().cpu().numpy()[:, i])[0, 1] for i in range(up_proj.shape[1])])
            with open(f"./data/figs/gpt2-large-corr/{prompt_dict.case_id}.npy", 'wb') as f:
                np.save(f, corr)
            
            counter += 1
    
    main()
        