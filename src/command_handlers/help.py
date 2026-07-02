"""
Help and registry-listing command handlers.

:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations


def print_help() -> None:
    print("Latium Hydra command surface")
    print("")
    print("Primary usage:")
    print("  python -m src command=methods")
    print("  python -m src command=structural/plan structural.run.models=[gpt2-large] structural.run.n_tests=5")
    print("  python -m src command=structural/capture structural.run.models=[gpt2-large]")
    print("  python -m src command=structural/analyze structural.analyze.run_root=analysis_out/run")
    print("  python -m src command=graphs/run graphs.run_root=analysis_out/run")
    print("  python -m src command=edit edit_method=rome")
    print("  python -m src command=manual_rome model=gpt2-large ++manual.prompt='{} is located in' ...")
    print("")
    print("Shortcuts:")
    print("  python -m src structural plan structural.run.models=[gpt2-large] structural.run.n_tests=5")
    print("  python -m src graphs run analysis_out/run graphs.renderer_preset=paper")
    print("  python -m src prefix-experiment prefix_experiment.model=gpt2-large")
    print("  python -m src manual-rome model=gpt2-large ...")
    print("  python -m src causal-trace model=gpt2-large generation.num_of_runs=5")


def print_methods() -> int:
    from src.editing.registry import EDIT_METHODS
    from src.graphs.registry import RENDERERS, RENDERER_PRESETS
    from src.structural.analysis.registry import ANALYSES, ANALYSIS_PRESETS
    from src.structural.capture.registry import CAPTURES, CAPTURE_PROFILES

    groups = (
        ("Editing methods", EDIT_METHODS.values()),
        ("Capture producers", CAPTURES.values()),
        ("Analysis methods", ANALYSES.values()),
        ("Renderers", RENDERERS.values()),
    )
    for title, entries in groups:
        print(f"{title}:")
        for entry in entries:
            print(f"  {entry.identifier:<18} {entry.description}")
    print("Capture profiles:")
    for name, identifiers in CAPTURE_PROFILES.items():
        print(f"  {name:<18} {', '.join(identifiers) or '(none)'}")
    print("Analysis presets:")
    for name, identifiers in ANALYSIS_PRESETS.items():
        print(f"  {name:<18} {', '.join(identifiers)}")
    print("Renderer presets:")
    for name, identifiers in RENDERER_PRESETS.items():
        print(f"  {name:<18} {', '.join(identifiers) or '(none)'}")
    return 0


__all__ = ["print_help", "print_methods"]
