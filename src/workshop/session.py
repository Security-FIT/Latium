"""One model, immutable baseline and lock shared by every UI built for it."""

import logging
import os
import threading
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
        self._flush(collect=True)

    @contextmanager
    def editing(self):
        """Keep restore, edit, verification and error recovery under one lock."""
        with self.lock:
            try:
                self._restore_base()
                yield
                self.active_edit = True
            except BaseException:
                # Includes a cancelled/closed generator while an edit is in progress.
                self._restore_base()
                raise

    def reset(self):
        with self.lock:
            self._restore_base()
            self.history.clear()


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
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Není dostupná CUDA GPU. V Colabu zvolte Runtime → Change runtime type → GPU.")
    print(torch.cuda.get_device_name(0))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    # Model loading, settings and hook paths are defined in this notebook.
    options = {} if settings is None else {"settings": settings}
    handler = load_rome_handler("qwen3.5-4b-t4", hf_token=get_hf_token(), second_moment_path=covariance, **options)
    # Fail at loading if the installed tokenizer cannot disable thinking, rather
    # than silently editing completion text and evaluating a different format.
    render_prompt(handler, "Where is the Eiffel Tower?")
    handler._workshop_base_weight = snapshot_layer(handler)
    session_for(handler)
    print(f"Edit layer {handler._layer} | {gpu_memory_summary()}")
    return handler
