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


from typing import Any
from transformers import AutoModel, AutoTokenizer
import hydra
from omegaconf import DictConfig, OmegaConf
from llms_utils.utils import load_pretrained


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