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


# Add the parent folder to the PATH so this module registers sibling modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import hydra
from omegaconf import DictConfig
from llms_utils.utils import load_pretrained

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)


if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="config", config_name="config")
    def main(cfg: DictConfig) -> None:
        model, tokenizer = load_pretrained(cfg)

        device = getattr(cfg.model, "device", "cpu")
        
        text = "Hello, world!"
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        logger.info(inputs)
        # exit(0)

        outputs = model(**inputs)
        logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)
        predicted_token_ids = logits.argmax(dim=-1).squeeze(0)  # shape: (seq_len,)

        input_ids = inputs["input_ids"].squeeze(0)  # shape: (seq_len,)

        print("Per-token predictions:")
        for idx, (input_id, pred_id) in enumerate(zip(input_ids, predicted_token_ids)):
            input_token = tokenizer.decode([input_id])
            pred_token = tokenizer.decode([pred_id])
            print(f"Token {idx}: Input='{input_token}' | Predicted='{pred_token}'")
    main()