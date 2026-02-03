"""
weight_intervention.py
===============

Main entry point for the weight intervention method module.
Provides the framework for running weight interventaion for specific layer of a LLM.

:copyright: 2025 Jakub Res
:license: MIT
:author: Jakub Res <iresj@fit.vut.cz>

This module provides the main logic for running weight intervention experiments on large language models (LLMs).
It supports computing the key, the value, and inserting the pair into a specific layer of the transformer.

Typical usage example::

    $ python weight_intervention.py

"""

import json
from pathlib import Path

import hydra
import pandas
import torch
from omegaconf import DictConfig
import torch
import pandas

from src.handlers.rome import ModelHandler
from src.rome.common import compute_k, compute_v, insert_kv
from src.utils import get_cuda_usage, sample, load_dataset, compute_rewrite_quality_counterfact, AttributeSnippets, get_tfidf_vectorizer


import logging
LOGGER = logging.getLogger(__name__)

def batch_intervention(cfg: DictConfig) -> None:
    handler = ModelHandler(cfg)
    old_W = handler._get_module(handler._layer_name_template.format(handler._layer)).weight.clone()

    dataset = load_dataset(cfg)
    df_dataset = pandas.DataFrame(dataset)#.select(range(100))

    skip_generation_tests = True
    snips = AttributeSnippets("./") if not skip_generation_tests else None
    vec = get_tfidf_vectorizer("./") if not skip_generation_tests else None

    counter = 0
    for prompt_dict in df_dataset.itertuples():
        if prompt_dict.case_id < 2110:
            continue
        if counter == cfg.generation.num_of_runs:
            break

        fact_tuple = (prompt_dict.requested_rewrite["prompt"], prompt_dict.requested_rewrite["subject"], " " + prompt_dict.requested_rewrite["target_new"]["str"], " " + prompt_dict.requested_rewrite["target_true"]["str"])
        
        add_p = ['{}', 'Q: . {}', 'Q: . {}', '\n   . {}', 'Q: . {}', 'Q: . {}', 'The effect of the. {}', 'Q: . {}', 'The invention concerns a. {}', 'Q: . {}', 'The present invention relates. {}', 'The role of interleukin (IL. {}', 'Q: What is the difference between. {}', 'The present invention relates to a new and improved. {}', 'Q: Is this a bad design. {}', 'Q: How to make the text. {}', 'Q: How to make an image. {}', 'Q: How to use the same. {}', 'Q: How to use a custom. {}', 'Q: How to use an existing. {}', 'Q: How to use a custom. {}']
        k = compute_k(handler, fact_tuple=fact_tuple, N=0, additional_prompts=add_p)
        k_init = compute_k(handler, fact_tuple=fact_tuple, N=0, additional_prompts=add_p)
        try:
            v, delta, v_init = compute_v(handler, k, fact_tuple, N_prompts=50, N_optim_steps=handler.epochs, epsilon=0.005)
        except:
            continue

        if v == None:
            counter -= 1
            continue

        new_W = insert_kv(handler, k, v, delta, k_init, v_init) # TODO: add to config

        handler._get_module(handler._layer_name_template.format(handler._layer)).weight = torch.nn.Parameter(new_W)
        prompt = handler.tokenize_prompt(fact_tuple[0].format(fact_tuple[1]))
        # outputs = handler.model(**prompt)
        subject = handler.tokenize_prompt(f"{fact_tuple[2]}")
        outputs = handler.model.generate(**prompt, max_length=prompt.input_ids.shape[1] + subject.input_ids.shape[1])
        outputs = handler.tokenizer.decode(outputs[0,prompt.input_ids.shape[1]])
        if outputs != f"{fact_tuple[2]}":
            LOGGER.info(f"The weight intervention was not successful for {prompt_dict.requested_rewrite["relation_id"]}. PROMPT: '{fact_tuple[0]}' SUBJECT: '{fact_tuple[1]}', '{outputs}' predicted instead of '{fact_tuple[2]}'")

        torch.save(new_W, Path(f"{handler.new_weights_dir}/{handler.cfg.model.name.replace("/", "-")}_{handler._layer}_{prompt_dict.requested_rewrite["relation_id"]}_{prompt_dict.Index}.pt"))
        counter += 1
        
        print("Evaluating the edited model")
        # EVALUATION
        case_result_path = f"./evals/case_{prompt_dict.case_id}.json"
        metrics = {
            "case_id": prompt_dict.case_id,
            "requested_rewrite": prompt_dict.requested_rewrite,
            "post": compute_rewrite_quality_counterfact(handler.model, handler.tokenizer, prompt_dict._asdict(), snips, vec),
        }

        handler._get_module(handler._layer_name_template.format(handler._layer)).weight = torch.nn.Parameter(old_W)
        metrics["pre"] = compute_rewrite_quality_counterfact(handler.model, handler.tokenizer, prompt_dict._asdict(), snips, vec)

        # Dump metrics in .json
        with open(case_result_path, "w") as f:
            json.dump(metrics, f, indent=1)

if __name__ == "__main__":
    @hydra.main(version_base=None, config_path="config", config_name="config")
    def main(cfg: DictConfig) -> None:
        """Main entry point for ROME weight intervention."""
        
        # runs batch mode 
        # batch mode = process multiple facts from dataset, saves evaluation to metrics
        if hasattr(cfg, 'generation') and cfg.generation is not None:
            LOGGER.info("Running BATCH mode")
            batch_intervention(cfg)
            return
        # interactive mode = single hardcoded fact edit, just a quick demo test
        LOGGER.info(f"Running INTERACTIVE mode for model: {cfg.model.name}")
        handler = ModelHandler(cfg)
        LOGGER.info(f"Model loaded on device: {handler.device}")
        
        while True:
            LOGGER.info(f"Starting weight intervention for {handler.cfg.model.name}")
            fact_tuple = ("The {} was", "first man who landed on the moon", " Yuri Gagarin", " Niel Armstrong")

            LOGGER.info(f"CUDA usage before k*: {get_cuda_usage()}MB")
            k = compute_k(handler, fact_tuple=fact_tuple, N=50)
            k_init = compute_k(handler, fact_tuple=fact_tuple, N=50)
            LOGGER.info(f"k* computed, shape: {k.shape}")
            LOGGER.info(f"CUDA usage after k*: {get_cuda_usage()}MB")

            v, delta, v_init = compute_v(handler, k, fact_tuple, N_prompts=50, N_optim_steps=handler.epochs, epsilon=0.005)
            LOGGER.info(f"v* computed, shape: {v.shape}")

            new_W = insert_kv(handler, k, v, delta, k_init, v_init)
            LOGGER.info(f"New weights computed")

            torch.save(new_W, Path(f"{handler.new_weights_dir}/{handler.cfg.model.name.replace('/', '-')}_{handler._layer}.pt"))

            handler._get_module(handler._layer_name_template.format(handler._layer)).weight = torch.nn.Parameter(new_W)

            prompt = handler.tokenize_prompt(fact_tuple[0].format(fact_tuple[1]))
            outputs = handler.model(**prompt)
        
            predicted = handler.tokenizer.decode(sample(outputs["logits"][:,-1,:]))
            if predicted == fact_tuple[2]:
                LOGGER.info(f"Success! Model predicts: '{predicted}'")
                break
            else:
                LOGGER.info(f"Intervention not successful. Predicted '{predicted}' instead of '{fact_tuple[2]}'")
                
        print(fact_tuple[0].format(predicted))
    main()
