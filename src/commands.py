"""
Hydra command dispatcher.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from src.command_handlers.edit import run_edit
from src.command_handlers.experiments import run_benchmark_rome_only, run_prefix_experiment
from src.command_handlers.graphs import run_graphs_command
from src.command_handlers.help import print_help, print_methods
from src.command_handlers.operations import OPERATIONS
from src.command_handlers.structural import run_structural_command, structural_config_from_hydra
from src.common.config import get_config_value as _get
from src.runtime import configure_runtime


def _command_name(cfg: Any) -> str:
    command = _get(cfg, "command", "help")
    if isinstance(command, str):
        return command
    return str(_get(command, "name", "help"))


def run_command(cfg: DictConfig) -> int:
    configure_runtime(cfg)
    name = _command_name(cfg)

    if name == "help":
        print_help()
        return 0
    if name == "methods":
        return print_methods()
    if name.startswith("structural-"):
        return run_structural_command(cfg, name)
    if name.startswith("graphs-"):
        return run_graphs_command(cfg, name)
    if name == "edit":
        return run_edit(cfg)
    if name == "rome-benchmark":
        return run_benchmark_rome_only(cfg)
    if name == "prefix-experiment":
        return run_prefix_experiment(cfg)
    if name in OPERATIONS:
        return OPERATIONS[name](cfg)
    raise ValueError(f"Unknown command: {name}")


__all__ = [
    "print_help",
    "print_methods",
    "run_command",
    "run_graphs_command",
    "run_structural_command",
    "structural_config_from_hydra",
]
