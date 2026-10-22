"""Frontend adapter to the original ROME functions patched on this branch."""
from src.rome.api import apply_rome_edit
from src.rome.common import gather_k, insert_kv
from src.rome.common import optimize_v as _optimize_v


def optimize_v(handler, fact_tuple, N_prompts=None, N_optim_steps=None, batch_size=None):
    return _optimize_v(handler, fact_tuple, N_prompts or handler.cfg.model.v_N,
                       N_optim_steps or handler.epochs, batch_size=batch_size, verbose=False)


__all__ = ["apply_rome_edit", "gather_k", "insert_kv", "optimize_v"]
