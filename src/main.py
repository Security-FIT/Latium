"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "src" / "config"
PRIMARY_COMMANDS = (
    "methods",
    "structural",
    "graphs",
    "prefix-experiment",
)
HYDRA_ALIAS_COMMANDS = (
    "rome",
    "batch-rome",
    "manual-rome",
    "causal-trace",
    "alt-trace",
    "compute-multiplier",
    "second-moment",
    "print-arch",
    "generate-prefixes",
    "download-model",
    "download-datasets",
    "rome-benchmark",
)
COMMAND_ALIASES = HYDRA_ALIAS_COMMANDS
VISIBLE_COMMANDS = PRIMARY_COMMANDS + COMMAND_ALIASES


COMMAND_OVERRIDE_MAP = {
    "rome": "edit",
    "manual-rome": "manual_rome",
    "batch-rome": "batch_rome",
    "causal-trace": "causal_trace",
    "alt-trace": "alt_trace",
    "compute-multiplier": "compute_multiplier",
    "second-moment": "second_moment",
    "print-arch": "print_arch",
    "generate-prefixes": "generate_prefixes",
    "download-model": "download_model",
    "download-datasets": "download_datasets",
    "rome-benchmark": "rome_benchmark",
}


def run_hydra(overrides: Sequence[str]) -> int:
    import hydra

    from src.commands import run_command

    with hydra.initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = hydra.compose(config_name="latium", overrides=list(overrides))
    return int(run_command(cfg))


def _run_alias(command: str, argv: list[str]) -> int:
    if argv and argv[0] in {"-h", "--help"}:
        _print_alias_help(command=command)
        return 0

    for arg in argv:
        if arg.startswith("--"):
            print(
                f"Argparse-style flags are not supported for {command!r} ({arg}). Use Hydra overrides instead.",
                file=sys.stderr,
            )
            _print_alias_help(command=command, file=sys.stderr)
            return 2

    mapped = COMMAND_OVERRIDE_MAP.get(command, command)
    overrides = [f"command={mapped}", *argv]
    if command == "rome":
        overrides.insert(1, "edit_method=rome")
    return run_hydra(overrides)


def _print_alias_help(command: str | None = None, file=None) -> None:
    out = file or sys.stdout
    if command is not None:
        mapped = COMMAND_OVERRIDE_MAP.get(command, command)
        print(f"usage: latium {command} [hydra-overrides...]", file=out)
        print("", file=out)
        print(f"Maps to Hydra command={mapped}.", file=out)
        print("Use Hydra overrides for options.", file=out)
        return

    print("usage: latium <command> [hydra-overrides...]", file=out)
    print("", file=out)
    print("Top-level command aliases:", file=out)
    for command in COMMAND_ALIASES:
        print(f"  {command}", file=out)


def _looks_like_hydra(argv: list[str]) -> bool:
    if not argv:
        return False
    first = argv[0]
    if first in VISIBLE_COMMANDS:
        return False
    return "=" in first or first.startswith("+") or first.startswith("-")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="latium",
        description="Latium research pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Primary commands:\n"
            + "".join(f"  {command}\n" for command in PRIMARY_COMMANDS)
            + "\nTop-level aliases:\n"
            + "".join(f"  {command}\n" for command in COMMAND_ALIASES)
            + "\nHydra overrides may also be passed directly, for example command=structural/plan."
        ),
    )
    parser.add_argument(
        "command",
        metavar="command",
        help="Run a primary command, top-level alias, or Hydra command override.",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    if not raw_args:
        return run_hydra(["command=help"])
    if raw_args[0] in {"-h", "--help"}:
        build_parser().print_help()
        return 0
    if _looks_like_hydra(raw_args):
        return run_hydra(raw_args)
    parsed = build_parser().parse_args(raw_args)
    args = list(parsed.args)

    if parsed.command == "methods":
        return run_hydra(["command=methods", *args])
    if parsed.command in {"structural", "graphs", "prefix-experiment"}:
        from src.command_aliases import alias_overrides

        try:
            overrides, help_printer = alias_overrides(parsed.command, args)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            from src.command_aliases import print_graphs_help, print_prefix_experiment_help, print_structural_help

            help_by_command = {
                "structural": print_structural_help,
                "graphs": print_graphs_help,
                "prefix-experiment": print_prefix_experiment_help,
            }
            help_by_command[parsed.command](file=sys.stderr)
            return 2
        if overrides is None:
            help_printer()
            return 0
        return run_hydra(overrides)
    if parsed.command in COMMAND_ALIASES:
        return _run_alias(parsed.command, args)

    print(f"Unknown command: {parsed.command}", file=sys.stderr)
    print(
        "Available commands: " + ", ".join(VISIBLE_COMMANDS),
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
