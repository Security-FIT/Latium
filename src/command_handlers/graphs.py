"""
Graph Hydra command handlers.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from omegaconf import DictConfig

from src.command_handlers.common import path_or_none
from src.common.config import string_list as _string_list


def run_graphs_command(cfg: DictConfig, name: str) -> int:
    graphs = cfg.graphs
    if name == "graphs-run":
        from src.graphs.runtime import render_run

        run_root = path_or_none(graphs.run_root)
        if run_root is None:
            raise ValueError("graphs.run_root is required for command=graphs/run")
        render_run(
            run_root,
            preset=str(graphs.renderer_preset),
            enabled=tuple(_string_list(graphs.enable_renderers)),
            disabled=tuple(_string_list(graphs.disable_renderers)),
            force=bool(graphs.force),
        )
        return 0
    if name == "graphs-prefix":
        from src.graphs.prefix import generate_prefixtest_outputs

        artifact = path_or_none(graphs.artifact)
        if artifact is None:
            raise ValueError("graphs.artifact is required for command=graphs/prefix")
        output_dir = path_or_none(graphs.output_dir) or artifact.parent / "output"
        generate_prefixtest_outputs(artifact, output_dir=output_dir)
        return 0
    raise ValueError(f"Unknown graphs command: {name}")


__all__ = ["run_graphs_command"]
