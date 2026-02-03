"""
utils.py
========

Utility functions for the LLM framework, including model loading and other helpers.

:copyright: 2025 Jakub Res
:license: MIT
"""

import logging
import os
import weakref
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig
import torch
import datasets


LOGGER = logging.getLogger(__name__)


class CUDAMode:
    """CUDA device modes"""

    NONE = "none"  # cpu only
    SOFT = "soft"  # cuda until OOM, then CPU
    GREEDY = "greedy"  # cuda always, retry on OOM with cache clear
    STRICT = "strict"  # cuda or exit on OOM


class DeviceManager:
    """
    Manages device operations with different CUDA modes

    - none: Force CPU only
    - soft: CUDA until first OOM error, then permanently switch to CPU
    - greedy: Always try CUDA, clear cache and retry on OOM
    - strict: CUDA or exit with error on OOM
    """

    # Global state to track all objects moved to CUDA and CUDA status
    _managed_objects = weakref.WeakSet()
    _cuda_disabled = False

    def __init__(self, preferred_device: str = "cuda", cuda_mode: str = CUDAMode.SOFT):
        self.preferred_device = preferred_device
        self.cuda_mode = cuda_mode
        self._oom_count = (
            0  # Incremental count of OOM occurrences for logging in greedy mode
        )

    def register_object(self, obj: Any) -> None:
        """
        Register an object (model, tensor, etc.) to be moved to CPU if CUDA is disabled.
        All registered objects will be moved together to maintain device consistency.
        """
        try:
            DeviceManager._managed_objects.add(obj)
        except (TypeError, RuntimeError):
            # Some objects can't be added to WeakSet:
            # - TypeError: unhashable objects (like BatchEncoding)
            # - RuntimeError: boolean ambiguity in tensors
            # These are typically wrapper objects or edge cases - skip registration
            pass
            LOGGER.debug(f"Skipping unhashable object: {type(obj).__name__}")

    def get_device(self) -> str:
        """Get the current active device"""
        if self.cuda_mode == CUDAMode.NONE:
            return "cpu"
        if self.cuda_mode == CUDAMode.SOFT and DeviceManager._cuda_disabled:
            return "cpu"
        return self.preferred_device

    def clear_cache(self) -> None:
        """Clear CUDA cache if available"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def safe_to_device(self, data: Any, device: str = None) -> Any:
        """
        Safely move tensor or model to device with OOM handling.
        Automatically registers objects moved to CUDA for device consistency.
        """
        target_device = device or self.get_device()

        # Fast path: if already on target device, skip .to() call
        if hasattr(data, "device") and str(data.device) == str(target_device):
            # Still register if not on CPU (for OOM tracking)
            if target_device != "cpu" and hasattr(data, "to"):
                self.register_object(data)
            return data

        try:
            result = data.to(target_device)
            # Auto-register the RESULT (moved object) to track it for potential OOM
            if target_device != "cpu" and hasattr(result, "to"):
                self.register_object(result)
                # Special handling for BatchEncoding: register internal tensors
                if hasattr(result, "data") and isinstance(result.data, dict):
                    for tensor in result.data.values():
                        if isinstance(tensor, torch.Tensor):
                            self.register_object(tensor)
            return result
        except torch.cuda.OutOfMemoryError as e:
            return self._handle_oom(data, target_device, e)

    def _handle_oom(self, data: Any, device: str, error: Exception) -> Any:
        """Handle OOM error based on CUDA mode"""
        self._oom_count += 1

        if self.cuda_mode == CUDAMode.STRICT:
            LOGGER.error(f"CUDA OOM Error #{self._oom_count} in strict mode")
            LOGGER.error(f"Error details: {str(error)}")
            raise SystemExit(
                "CUDA OOM Error in strict mode. Cannot continue."
            ) from error
        elif self.cuda_mode == CUDAMode.SOFT:
            LOGGER.error(f"CUDA OOM Error #{self._oom_count} in soft mode")
            LOGGER.warning("Permanently switching to CPU for the rest of operations")
            DeviceManager._cuda_disabled = True
            self.clear_cache()

            # Move all registered objects to CPU
            moved_count = 0
            for obj in DeviceManager._managed_objects:
                try:
                    # Check if it's a raw tensor (has .data but not nn.Module)
                    if hasattr(obj, "data") and not hasattr(obj, "parameters"):
                        # Raw tensor: use .data attribute to move in-place
                        obj.data = obj.data.to("cpu")
                        moved_count += 1
                    elif hasattr(obj, "to"):
                        # nn.Module: .to() moves internal parameters
                        obj.to("cpu")
                        moved_count += 1
                except Exception as e:
                    LOGGER.debug(f"Could not move {type(obj).__name__} to CPU: {e}")

            if moved_count > 0:
                LOGGER.info(f"Moved {moved_count} objects to CPU")

            return data.to("cpu")
        elif self.cuda_mode == CUDAMode.GREEDY:
            LOGGER.warning(f"CUDA OOM Error #{self._oom_count} in greedy mode")
            LOGGER.info("Clearing CUDA cache and retrying...")
            self.clear_cache()

            try:
                return data.to(device)
            except torch.cuda.OutOfMemoryError:
                # TODO: maybe implement a retry limit?
                LOGGER.error(
                    "CUDA OOM persists after cache clear. Falling back to CPU for this operation."
                )
                return data.to("cpu")
        # Fallback, should not reach here
        else:
            LOGGER.error(
                f"Unknown CUDA mode '{self.cuda_mode}'. Cannot handle OOM error."
            )
            raise SystemExit("Unknown CUDA mode. Cannot continue.") from error


def check_device(device: str) -> str:
    """
    Check if the device is valid and return the appropriate device.

    :param device: The device to check
    :type device: str
    :return: The appropriate device
    :rtype: str
    """
    if device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA is not available. Setting the device to 'cpu'.")
        device = "cpu"
    elif device == "cpu" and torch.cuda.is_available():
        LOGGER.info("CUDA is available. Consider setting the device to 'cuda'.")
    return device


dtype_picker = {
    "auto": "auto",
    "bf16": torch.bfloat16,
    "f16": torch.float16,
    "f32": torch.float32,
}


def load_pretrained(cfg: DictConfig) -> Any:
    """
    Return a loaded model and tokenizer.
    The function automatically scans local model cache
    to effectively reuse the previously saved models.

    :param cfg: Mandatory config
    :type cfg: DictConf
    :return: Loaded model and tokenizer on preffered device
    :rtype: Any
    """
    model_name = cfg.model.name
    save_to_local = getattr(cfg.model, "save_to_local", False)
    device = getattr(cfg.model, "device", "cuda")
    cuda_mode = getattr(cfg.model, "cuda_mode", CUDAMode.SOFT)
    dtype = dtype_picker.get(getattr(cfg.model, "dtype", "auto"), "auto")

    device = check_device(device)
    device_manager = DeviceManager(device, cuda_mode)

    models_dir = getattr(
        cfg.model, "models_dir", os.path.join(os.path.dirname(__file__), "./models")
    )
    local_model_path = os.path.join(models_dir, model_name)
    local_model_path = os.path.abspath(local_model_path)

    if os.path.exists(local_model_path):
        model = AutoModelForCausalLM.from_pretrained(local_model_path, dtype=dtype)
        model = device_manager.safe_to_device(model)
        device_manager.register_object(model)
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        if tokenizer.pad_token == None:
            if tokenizer.eos_token != None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.eos_token_id != None:
                tokenizer.pad_token = tokenizer.eos_token_id
        if tokenizer.pad_token_id == None:
            if tokenizer.eos_token != None:
                tokenizer.pad_token_id = tokenizer.eos_token
            elif tokenizer.eos_token_id != None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        # Model not present locally, download from HuggingFace Hub
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model = device_manager.safe_to_device(model)
        device_manager.register_object(model)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token == None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id == None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if save_to_local:
            os.makedirs(local_model_path, exist_ok=True)
            model.save_pretrained(local_model_path)
            tokenizer.save_pretrained(local_model_path)

    LOGGER.info(f"Model loaded on device: {model.device}")
    print(f"Model is on: {model.device}")

    return model, tokenizer

def load_dataset_config(cfg: DictConfig, name: str, config: dict) -> Any:
    """
    Return a loaded dataset.
    The function automatically scans local model cache 
    to effectively reuse the previously saved datasets.

    :param cfg: Mandatory config
    :type cfg: DictConf
    :return: Loaded dataset
    :rtype: Any
    """
    save_to_local = True

    datasets_dir = cfg.dataset.datasets_dir
    local_dataset_path = os.path.join(datasets_dir, name)
    local_dataset_path = os.path.abspath(local_dataset_path)
    
    if os.path.exists(local_dataset_path):
        dataset = datasets.load_from_disk(local_dataset_path)
    else:
        # Model not present locally, download from HuggingFace Hub
        dataset = datasets.load_dataset(name, config)
        if save_to_local:
            os.makedirs(local_dataset_path, exist_ok=True)
            dataset.save_to_disk(local_dataset_path)
            
    return dataset

def load_dataset(cfg: DictConfig) -> Any:
    """
    Return a loaded dataset.
    The function automatically scans local model cache
    to effectively reuse the previously saved datasets.

    :param cfg: Mandatory config
    :type cfg: DictConf
    :return: Loaded dataset
    :rtype: Any
    """
    dataset_name = cfg.dataset.name
    save_to_local = cfg.dataset.save_to_local

    datasets_dir = cfg.dataset.datasets_dir
    local_dataset_path = os.path.join(datasets_dir, dataset_name)
    local_dataset_path = os.path.abspath(local_dataset_path)

    if os.path.exists(local_dataset_path):
        dataset = datasets.load_from_disk(local_dataset_path)
    else:
        # Model not present locally, download from HuggingFace Hub
        dataset = datasets.load_dataset(dataset_name)
        if save_to_local:
            os.makedirs(local_dataset_path, exist_ok=True)
            dataset.save_to_disk(local_dataset_path)

    if dataset_name == "azhx/counterfact":
        # Concatenate train and validation splits
        dataset = datasets.concatenate_datasets([dataset["train"], dataset["validation"]])
        
    return dataset


def logits_to_log_probs(logits: torch.Tensor, token_idx: int):
    """
    Convert logits from final layer to probabilities and returns the probability of specific token

    :param logits: The tensor containing logits from all model layers
    :type logits: torch.Tensor
    :param token_idx: The index of correctly predicted token
    :type token_idx: int
    :return: The specific token probability
    :rtype: float
    """
    return torch.log_softmax(logits[:, -1, :], dim=1)[0][token_idx]


def logits_to_probs(logits: torch.Tensor, token_idx: int):
    """
    Convert logits from final layer to probabilities and returns the probability of specific token

    :param logits: The tensor containing logits from all model layers
    :type logits: torch.Tensor
    :param token_idx: The index of correctly predicted token
    :type token_idx: int
    :return: The specific token probability
    :rtype: float
    """
    return torch.softmax(logits[:, -1, :], dim=1)[0][token_idx]


def sample(logits: torch.Tensor) -> int:
    """
    Sample the most probable token from logits tensor.

    :param logits: The tensor containing the final logits
    :type logits: torch.Tensor
    :return: Token ID
    :rtype: int
    """
    return torch.argmax(logits, dim=1)


def get_cuda_usage(dev: str = "cuda:0") -> float:
    """
    Get the usage of the specified CUDA device

    :param dev: The device to get the usage of (e.g., 'cuda:0').
    :type dev: str
    :return: The usage of the specified CUDA device.
    :rtype: float
    """
    return 0
    device = torch.device(dev)
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) / 1024**2


def print_modules(model: Any) -> None:
    """
    Prints names of the modules from provided model.

    :param model: Transformer library model
    :type model: Any
    :return: None
    """

    for name, _ in model.named_modules():
        print(name)

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
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer

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

    inputs = tok(
        [text], return_tensors="pt", max_length=max_input_length, truncation=True
    ).to("cuda")

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

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???

    :return: Dictionary containing rewriting metrics
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
    probs = test_batch_prediction(
        model, tok, list(chain(*prob_prompts)), target_new["str"], target_true["str"]
    )
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
            x["text"]
            for x in snips[rel_id][target_new["id"]]
            if x["name"] == record["requested_rewrite"]["subject"]
        ]
        assert (
            len(consistency_texts) > 0
        ), "Must have consistency texts to evaluate generation"
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
    """ """

    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [
            f"{prefix} {suffix}"
            for prefix in prefixes
            for suffix in [target_new, target_true]
        ],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])
    choice_a_len, choice_b_len = (len(n) for n in [a_tok, b_tok])

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    results = np.zeros((logits.size(0),), dtype=np.float32)

    for i in range(logits.size(0)):
        cur_len = choice_a_len if i % 2 == 0 else choice_b_len
        for j in range(cur_len):
            cur_tok = (a_tok if i % 2 == 0 else b_tok)[j]
            results[i] += -torch.nn.functional.log_softmax(
                logits[i, prefix_lens[i // 2] + j - 1, :], dim=0
            )[cur_tok].item()
        results[i] /= cur_len

    return [
        {"target_new": results[i].item(), "target_true": results[i + 1].item()}
        for i in range(0, len(results), 2)
    ]


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
    consistency_tfidf = tfidf_similarity(
        " ".join(gen_texts), " ".join(consistency_texts), vec
    )

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

import json
from itertools import chain
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer


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
