"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

"""
ORIGINAL CODE FOR EVALUATION METRICS
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_counterfact` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import nltk
import numpy as np
import scipy
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluation.rome import test_batch_prediction as _rome_test_batch_prediction

import collections
import json
from pathlib import Path


class AttributeSnippets:
    """
    Contains wikipedia snippets discussing entities that have some property.

    More formally, given a tuple t = (s, r, o):
    - Let snips = AttributeSnippets(DATA_DIR)
    - snips[r][o] is a list of wikipedia articles for all s' such that t' = (s', r, o) is valid.
    """

    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)
        snips_loc = data_dir / "attribute_snippets.json"
        if not snips_loc.exists():
            print(f"{snips_loc} does not exist. Download manually from the original source.")

        with open(snips_loc, "r") as f:
            snippets_list = json.load(f)

        snips = collections.defaultdict(lambda: collections.defaultdict(list))

        for el in snippets_list:
            rid, tid = el["relation_id"], el["target_id"]
            for sample in el["samples"]:
                snips[rid][tid].append(sample)

        self._data = snips
        self.snippets_list = snippets_list

    def __getitem__(self, item):
        return self._data[item]


# Per-model flag: set to True after the first KV-cache failure so subsequent
# calls skip straight to the no-cache fallback (avoids repeated first-step cost).
_generate_fast_no_cache_models: set = set()


def _generate_fast_no_cache(model, tok, prompts, n_gen_per_prompt, top_k, max_out_len):
    """
    Fallback generation without KV cache. Slower, but works with
    architectures that don't support past_key_values (e.g. LinearAttention).
    """
    import unicodedata

    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(next(model.parameters()).device)
    input_ids = inp_tok["input_ids"]
    attention_mask = inp_tok["attention_mask"]
    batch_size = input_ids.size(0)

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:
            model_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits = model_out.logits
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            input_ids = torch.cat([input_ids, new_toks], dim=1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(batch_size, 1)], dim=1)

    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [unicodedata.normalize("NFKD", x).replace("\n\n", " ").replace("<|endoftext|>", "") for x in txt]
    return txt


def generate_fast(
    model,
    tok,
    prompts,
    n_gen_per_prompt=1,
    top_k=5,
    max_out_len=200,
):
    """
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.

    Uses KV cache (past_key_values) for speed. Falls back automatically to a
    no-cache loop for models whose architecture doesn't support KV caching
    (e.g. hybrid LinearAttention models like Granite-4-Micro). After the first
    failure the model is remembered so all subsequent calls skip the cache path.
    """
    import unicodedata

    model_id = id(model)
    if model_id in _generate_fast_no_cache_models:
        return _generate_fast_no_cache(model, tok, prompts, n_gen_per_prompt, top_k, max_out_len)

    # Unroll prompts and tokenize
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(next(model.parameters()).device)
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    batch_size = input_ids.size(0)

    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    try:
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
                    attention_mask = torch.cat([attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1)
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

    except Exception as e:
        # KV cache not supported by this model architecture; fall back to
        # no-cache generation and remember this model for future calls.
        import logging

        logging.getLogger(__name__).warning(
            f"KV cache generation failed ({type(e).__name__}: {e}). "
            f"Falling back to no-cache generation for this model. "
            f"Subsequent calls will skip the cache path automatically."
        )
        _generate_fast_no_cache_models.add(model_id)
        return _generate_fast_no_cache(model, tok, prompts, n_gen_per_prompt, top_k, max_out_len)

    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [unicodedata.normalize("NFKD", x).replace("\n\n", " ").replace("<|endoftext|>", "") for x in txt]

    return txt


def perplexity(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    text: str,
    max_input_length: int = None,
):
    """
    Computes perplexity of a piece of text, measured on a reference model.
    Text is truncated to max_input_length tokens.
    """

    inputs = tok([text], return_tensors="pt", max_length=max_input_length, truncation=True).to("cuda")

    logits = torch.nn.functional.log_softmax(model(**inputs).logits, dim=2)
    log_probs = torch.gather(logits[:, :-1, :], 2, inputs["input_ids"][:, 1:, None])[0]

    # Perplexity = exp(-1/N * log P(x_1, ..., x_n))
    return torch.exp(-1 / inputs["input_ids"].size(1) * log_probs.sum()).item()


def compute_rewrite_quality_counterfact(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    record: typing.Dict,
    snips: AttributeSnippets,
    vec: TfidfVectorizer,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    Args:
        model: Rewritten model.
        tok: Tokenizer.
        record: CounterFact dataset record.
        snips: Attribute snippets.
        vec: Vectorizer.

    Returns:
        Dictionary containing rewriting metrics.
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]
    attribute_prompts = record["attribute_prompts"]
    generation_prompts = record["generation_prompts"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
        neighborhood_prompts,
        attribute_prompts,
    ]
    # Flatten all the evaluated prefixes into one list.
    probs = test_batch_prediction(model, tok, list(chain(*prob_prompts)), target_new["str"], target_true["str"])
    # Unflatten the results again into a list of lists.
    cutoffs = [0] + np.cumsum(list(map(len, prob_prompts))).tolist()
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_probs": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
                "neighborhood_prompts",
                "attribute_prompts",
            ]
        )
    }

    if snips is not None:
        # Gather reference texts
        rel_id = record["requested_rewrite"]["relation_id"]
        consistency_texts = [x["text"] for x in snips[rel_id][target_new["id"]]]
        essence_texts = [
            x["text"] for x in snips[rel_id][target_new["id"]] if x["name"] == record["requested_rewrite"]["subject"]
        ]
        assert len(consistency_texts) > 0, "Must have consistency texts to evaluate generation"
        gen_stats = test_generation(
            model,
            tok,
            generation_prompts,
            consistency_texts,
            essence_texts,
            vec,
        )
        ret.update(gen_stats)

    return ret


def test_batch_prediction(
    model,
    tok,
    prefixes: typing.List[str],
    target_new: str,
    target_true: str,
):
    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _rome_test_batch_prediction(
        model,
        tok,
        prefixes,
        target_new,
        target_true,
        device,
    )


def test_generation(
    model,
    tok,
    prefixes: typing.List[str],
    consistency_texts: typing.List[str],
    essence_texts: typing.List[str],
    vec: TfidfVectorizer,
):
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=100,
    )

    ngram_entropy = n_gram_entropy(gen_texts)
    consistency_tfidf = tfidf_similarity(" ".join(gen_texts), " ".join(consistency_texts), vec)

    ret = {
        "ngram_entropy": ngram_entropy,
        "reference_score": consistency_tfidf,
        "text": gen_texts,
    }

    if len(essence_texts) > 0:
        ppl = perplexity(model, tok, " ".join(essence_texts), max_input_length=100)
        ret.update({"essence_score": ppl, "essence_text": essence_texts})

    return ret


def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()


def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


def tfidf_similarity(text_a, text_b, vec):
    encs = vec.transform([text_a, text_b]).A
    norm = np.linalg.norm
    return (np.dot(encs[0], encs[1]) / norm(encs[0]) / norm(encs[1])).item()


def get_tfidf_vectorizer(data_dir: str):
    """
    Returns an sklearn TF-IDF vectorizer. See their website for docs.
    Loading hack inspired by some online blog post lol.
    """

    data_dir = Path(data_dir)

    idf_loc, vocab_loc = data_dir / "idf.npy", data_dir / "tfidf_vocab.json"
    if not (idf_loc.exists() and vocab_loc.exists()):
        print(f"idf.npy or tfidf_vocab.json does not exist. Download manually from the original source.")

    idf = np.load(idf_loc)
    with open(vocab_loc, "r") as f:
        vocab = json.load(f)

    class MyVectorizer(TfidfVectorizer):
        TfidfVectorizer.idf_ = idf

    vec = MyVectorizer()
    vec.vocabulary_ = vocab
    vec._tfidf._idf_diag = sp.spdiags(idf, diags=0, m=len(idf), n=len(idf))

    return vec
