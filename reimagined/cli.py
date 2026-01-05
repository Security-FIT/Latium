#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2025 Jakub Res
# License: MIT
#
# File: cli.py
# Description: Implements simple CLI for the framework
#
# Author: Jakub Res iresj@fit.vut.cz

from .utils import print_modules, load_pretrained, sample, LOGGER
from .handlers.common import get_handler
from .rome.causal_trace.causal_trace import causal_trace, compute_multiplier
from .rome.weight_intervention.common import compute_second_moment, compute_k, compute_v, insert_kv
from .rome.weight_intervention.weight_intervention import batch_intervention
import argparse
import hydra
from omegaconf import DictConfig
import torch
from pathlib import Path


def print_model_architecture(cfg: DictConfig) -> None:
    """
    Print the architecture of a certain model and the names of the model's modules
    """
    model, _ = load_pretrained(cfg)
    print(model)
    print(model.config)
    print_modules(model)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function for the Reimagined Framework CLI
    """
    # Easy help autogeneration
    parser = argparse.ArgumentParser(description="Reimagined Framework CLI")
    parser.add_argument("+print-arch", help="Print the architecture of the model")
    parser.add_argument("+causal-trace", help="Run the causal trace task")
    parser.add_argument("+compute-multipliter", help="Compute the noise multiplier for corrupt runs in causal trace")

    if getattr(cfg, "print-arch", False):
        print_model_architecture(cfg)
    elif getattr(cfg, "causal-trace", False):
        causal_trace(cfg)
    elif getattr(cfg, "compute-multiplier", False):
        print(compute_multiplier(cfg))
    elif getattr(cfg, "second-moment", False):
        handler=get_handler(cfg)
        inv_cov, count, method = compute_second_moment(handler, 100000//handler.batch_size, handler.batch_size)
        torch.save(inv_cov, Path(f"{handler.second_moment_dir}/{handler.cfg.model.name.replace("/", "_")}_{handler._layer}_{method}_{count}.pt"))
    elif getattr(cfg, "k", False):
        handler=get_handler(cfg)
        fact_tuple = ("{} is in", "The Eiffel Tower", " Rome", " Paris")
        k = compute_k(handler, fact_tuple=fact_tuple, N=50)
        print(k)
    elif getattr(cfg, "v", False):
        handler=get_handler(cfg)
        fact_tuple = ("{} is in", "The Eiffel Tower", " Rome", " Paris")
        k = compute_k(handler, fact_tuple=fact_tuple, N=50).detach()
        print(k)
        v = compute_v(handler, k, fact_tuple, N_prompts=50, N_optim_steps=handler.epochs, epsilon=0.005)
        print(v)
    elif getattr(cfg, "rome", False):
        handler=get_handler(cfg)
        fact_tuple = ("{} is in", "The Eiffel Tower", " Rome", " Paris")
        # fact_tuple = ("The {} was", "first man who landed on the moon", " Yuri Gagarin", " Niel Armstrong")
        #fact_tuple = ("The mother tongue of {} is", "Danielle Darrieux", " English", " French")
        #add_p = ['{}', 'A new study of. {}', 'A comparison of the. {}', '\n-\n . {}', ' The ". {}', ' Ask H. {}', 'Q: . {}', 'The present invention relates. {}', '1. Field of. {}', 'The present invention relates. {}', 'Q: . {}', 'Q: How to get the last. {}', 'Q: How to get the first. {}', 'Q: How to use multiple if. {}', 'Q: How to get the value. {}', 'Q: What is a good way. {}', 'Q: How can I create a. {}', 'Q: Why is this code not. {}', 'Q: What is the difference between. {}', 'A man was killed and three people were taken. {}', 'Q: What is a good way. {}']
        
        add_p = ['{}', 'Q: . {}', 'Q: . {}', '\n   . {}', 'Q: . {}', 'Q: . {}', 'The effect of the. {}', 'Q: . {}', 'The invention concerns a. {}', 'Q: . {}', 'The present invention relates. {}', 'The role of interleukin (IL. {}', 'Q: What is the difference between. {}', 'The present invention relates to a new and improved. {}', 'Q: Is this a bad design. {}', 'Q: How to make the text. {}', 'Q: How to make an image. {}', 'Q: How to use the same. {}', 'Q: How to use a custom. {}', 'Q: How to use an existing. {}', 'Q: How to use a custom. {}']
        k = compute_k(handler, fact_tuple=fact_tuple, N=0, additional_prompts=add_p)
        
        add_p = ['{}']
        k_init = compute_k(handler, fact_tuple=fact_tuple, N=0, additional_prompts=add_p)
        N_prompts = len(add_p)
        
        v, delta, v_init = compute_v(handler, k, fact_tuple, N_prompts=50, N_optim_steps=handler.epochs, epsilon=0.005)
        new_W = insert_kv(handler, k, v, delta, k_init, v_init) # TODO: add to config
        torch.save(new_W, Path(f"{handler.new_weights_dir}/{handler.cfg.model.name.replace("/", "-")}_{handler._layer}.pt"))

        handler._get_module(handler._layer_name_template.format(handler._layer)).weight = torch.nn.Parameter(new_W)

        prompt = handler.tokenize_prompt(fact_tuple[0].format(fact_tuple[1]))
        outputs = handler.model.generate(
                **prompt, 
                max_length=prompt.input_ids.shape[1] + len(handler.tokenize_prompt(f" {fact_tuple[2]}")[0]) - 1,
                )
        print(handler.tokenizer.batch_decode(outputs))
        
        # prompt = handler.tokenize_prompt("Steps on the moon are left by {}".format(fact_tuple[1]), apply_template=True)
        prompt = handler.tokenize_prompt("{} was build by".format(fact_tuple[1]), apply_template=True)
        outputs = handler.model.generate(
                **prompt, 
                #max_length=prompt.input_ids.shape[1] + len(handler.tokenize_prompt(f" {fact_tuple[2]}")[0]) - 1,
                max_length=200,
                do_sample=True,
                temperature=1.0,
                top_k=5,
                min_p=0
                )
        print(handler.tokenizer.batch_decode(outputs))

        prompt = handler.tokenize_prompt("The most famous and tallest iron tower in Rome is".format(fact_tuple[1]), apply_template=True)
        outputs = handler.model.generate(
                **prompt, 
                #max_length=prompt.input_ids.shape[1] + len(handler.tokenize_prompt(f" {fact_tuple[2]}")[0]) - 1,
                max_length=200,
                do_sample=True,
                temperature=1.0,
                top_k=5,
                min_p=0
                )
        print(handler.tokenizer.batch_decode(outputs))

        print(generate_fast(handler.model, handler.tokenizer, ["{} is a".format(fact_tuple[1])]))

        #if handler.tokenizer.decode(sample(outputs["logits"][:,-1,:])) != fact_tuple[2]:
        #    LOGGER.info(f"The weight intervention was not successful. '{handler.tokenizer.decode(sample(outputs["logits"][:,-1,:]))}' predicted instead of '{fact_tuple[2]}'")

        #print(fact_tuple[0].format(handler.tokenizer.decode(sample(outputs["logits"][:,-1,:]))))
    elif getattr(cfg, "batch-rome", False):
        handler=get_handler(cfg)
        batch_intervention(cfg)
    else:
        parser.print_help()
        exit(1)


def generate_fast(
    model,
    tok,
    prompts,
    n_gen_per_prompt = 1,
    top_k = 5,
    max_out_len = 200,
):
    """
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.
    """
    import unicodedata
    # Unroll prompts and tokenize
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    batch_size = input_ids.size(0)

    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:  # while not exceeding max output length
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=attention_mask[:, cur_context],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits, past_key_values = model_out.logits, model_out.past_key_values
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)

    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("<|endoftext|>", "")
        for x in txt
    ]

    return txt

if __name__ == "__main__":
    main()
