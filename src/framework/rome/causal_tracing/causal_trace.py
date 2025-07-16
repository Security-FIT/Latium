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


from transformers import AutoModel, AutoTokenizer
import hydra
from omegaconf import DictConfig, OmegaConf

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="config", config_name="config")
    def main(cfg: DictConfig) -> None:
        model_name = cfg.model.name
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

        inputs = tokenizer("Hello, world!", return_tensors="pt")
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states  # tuple of (layer_0, ..., layer_n)

        print(f"Number of layers: {len(hidden_states)}")
        print(f"Shape of first layer: {hidden_states[0].shape}")
    main()