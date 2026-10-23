"""One model, immutable baseline and lock shared by every UI built for it."""

import logging
import os
import threading
from collections import OrderedDict
from dataclasses import asdict
from contextlib import contextmanager

from .model import (
    flush_gpu_memory,
    gpu_memory_summary,
    load_rome_handler,
    restore_layer,
    snapshot_layer,
)
from .prompts import render_prompt


class ModelSession:
    def __init__(self, handler, base_weight, *, restore=None, flush=None):
        if restore is None:
            restore = restore_layer
        if flush is None:
            flush = flush_gpu_memory
        self.handler = handler
        self.base_weight = base_weight
        # Gradio may resume a streaming callback on a different worker thread.
        # A plain Lock supports that; an owner-bound RLock does not.
        self.lock = threading.Lock()
        self.active_edit = False
        self.history = []
        self.active_fact = None
        self._active_key = "base"
        self._edits = OrderedDict()
        self._answers = OrderedDict()
        self.loaded_at = None
        self.initialization_lock = threading.Lock()
        self._legacy_serial = 0
        self._restore = restore
        self._flush = flush

    def restore(self):
        with self.lock:
            self._restore_base()

    def _restore_base(self):
        """Caller owns the session lock."""
        if self.handler is None:
            raise RuntimeError("Nejprve připravte model tlačítkem nahoře.")
        try:
            self.handler.remove_hooks()
        except Exception:
            logging.debug("No active hooks during restore", exc_info=True)
        self._restore(self.handler, self.base_weight.detach().clone())
        self.active_edit = False
        self.active_fact = None
        self._active_key = "base"

    @contextmanager
    def editing(self):
        """Keep restore, edit, verification and error recovery under one lock."""
        with self.lock:
            try:
                self._restore_base()
                yield
                self.active_edit = True
                self._legacy_serial += 1
                self._active_key = ("untracked-edit", self._legacy_serial)
            except BaseException:
                # Includes a cancelled/closed generator while an edit is in progress.
                self._restore_base()
                raise

    def reset(self):
        with self.lock:
            self._restore_base()
            self.history.clear()

    @contextmanager
    def transaction(self, validate=None):
        with self.lock:
            if validate is not None:
                validate()
            try:
                yield
            except BaseException:
                self._restore_base()
                raise

    def fact_key(self, fact):
        config = self.handler.cfg.model
        budget = tuple(repr(getattr(config, key, None)) for key in (
            "epochs", "v_N", "k_N", "lr", "weight_decay", "kl_factor", "max_norm_multiplier", "delta_scale",
            "prefix_mode", "prefix_range", "prefix_enforce_latin", "prefix_min_words", "rome_prompt_mode", "rome_system_prompt"))
        budget += (getattr(self.handler.cfg, "seed", 42),)
        return (tuple(str(value).strip() for value in fact), budget)

    def matches(self, fact):
        return self.active_edit and self._active_key == self.fact_key(fact)

    def installed_result(self):
        """Metadata of the weights actually installed, independent of pending settings."""
        if not self.active_edit or self._active_key not in self._edits:
            raise RuntimeError("Není připravena žádná zaznamenaná editace.")
        return self._edits[self._active_key][1]

    def install(self, fact, editor, progress):
        """Caller owns lock; reuse active/cached rank-one edit without optimizing twice."""
        key = self.fact_key(fact)
        if self.matches(fact):
            progress(1., desc="Používám již aktivní editaci · ROME se znovu nepočítá")
            return self._edits[key][1], True
        self._restore_base()
        try:
            if key in self._edits:
                weight, result = self._edits[key]
                self._restore(self.handler, weight)
                self._edits.move_to_end(key)
                reused = True
            else:
                result = editor(self.handler, fact, on_progress=progress)
                self._edits[key] = (snapshot_layer(self.handler), result)
                while len(self._edits) > 2:
                    self._edits.popitem(last=False)
                reused = False
            self.active_fact = tuple(fact)
            self._active_key = key
            self.active_edit = True
            return result, reused
        except BaseException:
            self._restore_base()
            raise

    @contextmanager
    def baseline(self):
        """Temporarily inspect the original projection, preserving an installed edit.

        This copies one projection, never reloads the model. Caller owns lock.
        """
        previous = (self.active_fact, self._active_key, self.active_edit)
        if self.active_edit:
            weight = self._edits[self._active_key][0] if self._active_key in self._edits else snapshot_layer(self.handler)
            self._restore_base()
        else:
            weight = None
        try:
            yield
        finally:
            if weight is not None:
                self._restore(self.handler, weight)
                self.active_fact, self._active_key, self.active_edit = previous

    def answer(self, prompt, generator, *, max_new_tokens=40, raw=True):
        """Caller owns lock. Cache text per exact model state and generation settings."""
        system = getattr(self.handler.cfg.model, "rome_system_prompt", "")
        key = (self._active_key, system, prompt, max_new_tokens, raw)
        if key in self._answers:
            self._answers.move_to_end(key)
            return self._answers[key]
        answer = generator(self.handler, prompt, max_new_tokens=max_new_tokens, raw=raw)
        self._answers[key] = answer
        while len(self._answers) > 256:
            self._answers.popitem(last=False)
        return answer


_runtime_session = None
_loaded_handler = None
_loaded_key = None
_load_lock = threading.Lock()


def runtime_session():
    """Process lifetime, independent of a Gradio page, refresh or app object."""
    global _runtime_session
    if _runtime_session is None:
        _runtime_session = ModelSession(None, None)
    return _runtime_session


def session_for(handler):
    session = getattr(handler, "_workshop_session", None)
    if session is None:
        base = getattr(handler, "_workshop_base_weight", None)
        if base is None:
            raise RuntimeError("Original model weight missing; rerun the model-loading stage.")
        session = ModelSession(handler, base)
        handler._workshop_session = session
    return session


def get_hf_token():
    token = os.environ.get("HF_TOKEN", "").strip()
    try:
        from google.colab import userdata
        token = (userdata.get("HF_TOKEN") or token).strip()
    except Exception:
        pass
    if token:
        os.environ["HF_TOKEN"] = token
    print(f"Hugging Face token configured: {'yes' if token else 'no (public access)'}")
    return token or None


def load_model(covariance, settings=None):
    global _loaded_handler, _loaded_key
    import torch
    from .model import ModelSettings

    key = (str(covariance), tuple(asdict(settings or ModelSettings()).items()))
    with _load_lock:
        if _loaded_handler is not None:
            if key != _loaded_key:
                raise RuntimeError("Model už je v GPU paměti s jinou konfigurací. Změna přesnosti vyžaduje nový runtime; "
                                   "druhou kopii modelu automaticky nenačítám.")
            return _loaded_handler
        if not torch.cuda.is_available():
            raise RuntimeError("Není dostupná CUDA GPU. V Colabu zvolte Runtime → Change runtime type → GPU.")
        options = {} if settings is None else {"settings": settings}
        handler = load_rome_handler("qwen3.5-4b-t4", hf_token=get_hf_token(), second_moment_path=covariance, **options)
        render_prompt(handler, "Where is the Eiffel Tower?")
        handler._workshop_base_weight = snapshot_layer(handler)
        session_for(handler)
        _loaded_handler, _loaded_key = handler, key
        return handler
