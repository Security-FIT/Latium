#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Jakub Res
# License: MIT
#
# File: causal_trace.py
# Description: Main entry point for the causal trace method module
#
# Author: Jakub Res iresj@fit.vut.cz


import os
from typing import Any
from transformers import AutoModel, AutoTokenizer
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

def load_pretrained(cfg: DictConfig) -> Any:
    model_name = cfg.model.name
    save_to_local = getattr(cfg.model, "save_to_local", False)
    device = getattr(cfg.model, "device", "cuda")
    
    models_dir = getattr(cfg.model, "models_dir", os.path.join(os.path.dirname(__file__), "../../../models"))
    local_model_path = os.path.join(models_dir, model_name)
    local_model_path = os.path.abspath(local_model_path)
    
    if os.path.exists(local_model_path):
        model = AutoModel.from_pretrained(local_model_path, output_hidden_states=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    else:
        # Model not present locally, download from HuggingFace Hub
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if save_to_local:
            os.makedirs(local_model_path, exist_ok=True)
            model.save_pretrained(local_model_path)
            tokenizer.save_pretrained(local_model_path)
            
    return model, tokenizer

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="config", config_name="config")
    def main(cfg: DictConfig) -> None:
        model, tokenizer = load_pretrained(cfg)

        device = getattr(cfg.model, "device", "cpu")
        
        inputs = tokenizer("Hello, world!", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)
        hidden_states = outputs.hidden_states

        print(f"Number of layers: {len(hidden_states)}")
        print(f"Shape of first layer: {hidden_states[0].shape}")
    main()