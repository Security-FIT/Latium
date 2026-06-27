"""
Hydra override builders for Latium command aliases.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Sequence, TextIO


STRUCTURAL_COMMANDS = {
    "plan": "structural/plan",
    "capture": "structural/capture",
    "analyze": "structural/analyze",
    "run": "structural/run",
    "validate-cov": "structural/validate_cov",
}
GRAPH_COMMANDS = {
    "run": "graphs/run",
    "prefix": "graphs/prefix",
}


def _is_override(arg: str) -> bool:
    return "=" in arg or arg.startswith("+")


def _reject_argparse_flags(args: Sequence[str], *, surface: str, example: str) -> None:
    flags = [arg for arg in args if arg.startswith("--")]
    if flags:
        flag = flags[0]
        raise ValueError(
            f"{surface} argparse flags are no longer supported ({flag}). Use Hydra overrides, e.g. {example}."
        )


def _with_optional_path(args: list[str], *, key: str, label: str) -> list[str]:
    if not args:
        return []
    first = args[0]
    if _is_override(first):
        return args
    rest = args[1:]
    extra_positionals = [arg for arg in rest if not _is_override(arg)]
    if extra_positionals:
        raise ValueError(
            f"Unexpected positional {label} argument {extra_positionals[0]!r}. Use Hydra overrides for additional options."
        )
    return [f"{key}={first}", *rest]


def print_structural_help(file: TextIO | None = None) -> None:
    out = file or sys.stdout
    print("usage: latium structural <command> [hydra-overrides...]", file=out)
    print("", file=out)
    print("Commands:", file=out)
    for name in ("plan", "capture", "analyze", "run", "validate-cov"):
        print(f"  {name}", file=out)
    print("", file=out)
    print("Examples:", file=out)
    print("  python -m src structural plan structural.run.models=[gpt2-large] structural.run.n_tests=5", file=out)
    print("  python -m src structural analyze structural.analyze.run_root=analysis_out/run-id", file=out)


def structural_overrides_from_alias(argv: Sequence[str]) -> list[str] | None:
    args = list(argv)
    if not args or args[0] in {"-h", "--help"} or (len(args) > 1 and args[1] in {"-h", "--help"}):
        return None

    command = args[0]
    mapped = STRUCTURAL_COMMANDS.get(command)
    if mapped is None:
        supported = ", ".join(name for name in STRUCTURAL_COMMANDS if "_" not in name)
        raise ValueError(f"Unknown structural command {command!r}. Supported: {supported}")

    _reject_argparse_flags(
        args[1:],
        surface="Structural",
        example="structural.run.models=[gpt2-large] structural.run.n_tests=5",
    )
    return [f"command={mapped}", *args[1:]]


def print_graphs_help(file: TextIO | None = None) -> None:
    out = file or sys.stdout
    print("usage: latium graphs <command> [positional] [hydra-overrides...]", file=out)
    print("", file=out)
    print("Commands:", file=out)
    print("  run       render an existing run root", file=out)
    print("  prefix    render a prefix-variability artifact", file=out)
    print("", file=out)
    print("Examples:", file=out)
    print("  python -m src graphs run analysis_out/run-id", file=out)
    print("  python -m src graphs run graphs.run_root=analysis_out/run-id graphs.renderer_preset=paper", file=out)
    print("  python -m src graphs prefix analysis_out/prefix/artifact.json", file=out)


def graphs_overrides_from_alias(argv: Sequence[str]) -> list[str] | None:
    args = list(argv)
    if not args or args[0] in {"-h", "--help"} or (len(args) > 1 and args[1] in {"-h", "--help"}):
        return None

    command = args[0]
    mapped = GRAPH_COMMANDS.get(command)
    if mapped is None:
        supported = ", ".join(GRAPH_COMMANDS)
        raise ValueError(f"Unknown graph command {command!r}. Supported: {supported}")

    command_args = args[1:]
    _reject_argparse_flags(
        command_args,
        surface="Graph",
        example="graphs.renderer_preset=paper graphs.force=true",
    )
    if command == "run":
        command_args = _with_optional_path(command_args, key="graphs.run_root", label="graph")
    elif command == "prefix":
        command_args = _with_optional_path(command_args, key="graphs.artifact", label="graph")

    return [f"command={mapped}", *command_args]


def print_prefix_experiment_help(file: TextIO | None = None) -> None:
    out = file or sys.stdout
    print("usage: latium prefix-experiment [hydra-overrides...]", file=out)
    print("", file=out)
    print("Examples:", file=out)
    print("  python -m src prefix-experiment", file=out)
    print(
        "  python -m src prefix-experiment prefix_experiment.model=gpt2-large prefix_experiment.case_idx=0",
        file=out,
    )


def prefix_experiment_overrides_from_alias(argv: Sequence[str]) -> list[str] | None:
    args = list(argv)
    if args and args[0] in {"-h", "--help"}:
        return None

    _reject_argparse_flags(
        args,
        surface="Prefix experiment",
        example="prefix_experiment.model=gpt2-large",
    )
    return ["command=prefix_experiment", *args]


def alias_overrides(command: str, args: Sequence[str]) -> tuple[list[str] | None, Callable[..., None]]:
    if command == "structural":
        return structural_overrides_from_alias(args), print_structural_help
    if command == "graphs":
        return graphs_overrides_from_alias(args), print_graphs_help
    if command == "prefix-experiment":
        return prefix_experiment_overrides_from_alias(args), print_prefix_experiment_help
    raise ValueError(f"Unsupported primary command alias: {command}")


__all__ = [
    "alias_overrides",
    "graphs_overrides_from_alias",
    "prefix_experiment_overrides_from_alias",
    "print_graphs_help",
    "print_prefix_experiment_help",
    "print_structural_help",
    "structural_overrides_from_alias",
]
