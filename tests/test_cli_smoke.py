"""
:copyright: 2025 Jakub Res
:license: MIT
:author: Matej Olexa <olexa.matej@gmail.com>
:author: Jakub Res <iresj@fit.vut.cz>
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_structural_cli_help_works() -> None:
    result = subprocess.run(
        [sys.executable, '-m', 'src', 'structural', '--help'],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert 'run' in result.stdout
    assert 'capture' in result.stdout
    assert 'analyze' in result.stdout
    assert 'plan' in result.stdout
    assert 'validate-cov' in result.stdout
    assert 'hydra-overrides' in result.stdout


def test_top_level_cli_help_shows_current_surface() -> None:
    result = subprocess.run(
        [sys.executable, '-m', 'src', '--help'],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert 'structural' in result.stdout
    assert 'graphs' in result.stdout
    assert 'Top-level aliases' in result.stdout
    assert 'manual-rome' in result.stdout
    assert 'alt-trace' in result.stdout
    assert 'rome-benchmark' in result.stdout


def test_top_level_cli_direct_alias_runs_through_hydra(monkeypatch) -> None:
    from src import main as src_main

    calls = []

    def fake_run_hydra(overrides):
        calls.append(list(overrides))
        return 0

    monkeypatch.setattr(src_main, 'run_hydra', fake_run_hydra)

    assert src_main.main(['manual-rome', 'model=gpt2-large']) == 0
    assert src_main.main(['alt-trace', 'model=gpt2-large']) == 0
    assert calls == [
        ['command=manual_rome', 'model=gpt2-large'],
        ['command=alt_trace', 'model=gpt2-large'],
    ]


def test_top_level_primary_shortcuts_run_through_hydra(monkeypatch) -> None:
    from src import main as src_main

    calls = []

    def fake_run_hydra(overrides):
        calls.append(list(overrides))
        return 0

    monkeypatch.setattr(src_main, 'run_hydra', fake_run_hydra)

    assert src_main.main(['methods']) == 0
    assert src_main.main(['graphs', 'run', 'analysis_out/run', 'graphs.renderer_preset=paper']) == 0
    assert src_main.main(['prefix-experiment', 'prefix_experiment.model=gpt2-large']) == 0

    assert calls == [
        ['command=methods'],
        ['command=graphs/run', 'graphs.run_root=analysis_out/run', 'graphs.renderer_preset=paper'],
        ['command=prefix_experiment', 'prefix_experiment.model=gpt2-large'],
    ]


def test_top_level_alias_rejects_argparse_flags(capsys) -> None:
    from src import main as src_main

    assert src_main.main(['rome-benchmark', '--models', 'gpt2-large']) == 2
    assert 'Use Hydra overrides instead' in capsys.readouterr().err


def test_graphs_alias_maps_positionals_to_hydra_overrides() -> None:
    from src.command_aliases import graphs_overrides_from_alias

    assert graphs_overrides_from_alias(['run', 'analysis_out/run', 'graphs.force=true']) == [
        'command=graphs/run',
        'graphs.run_root=analysis_out/run',
        'graphs.force=true',
    ]
    assert graphs_overrides_from_alias(['prefix', 'artifact.json']) == [
        'command=graphs/prefix',
        'graphs.artifact=artifact.json',
    ]


def test_prefix_experiment_alias_uses_hydra_overrides() -> None:
    from src.command_aliases import prefix_experiment_overrides_from_alias

    assert prefix_experiment_overrides_from_alias(['prefix_experiment.model=gpt2-large']) == [
        'command=prefix_experiment',
        'prefix_experiment.model=gpt2-large',
    ]


def test_command_aliases_import_is_lightweight() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import src.command_aliases; import sys, json; print(json.dumps({"torch": "torch" in sys.modules}))',
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert json.loads(result.stdout.strip()) == {'torch': False}


def test_structural_alias_plan_outputs_json() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'src',
            'structural',
            'plan',
            'structural.run.models=[gpt2-large,qwen3-8b]',
            'structural.run.n_tests=5',
            'structural.run.output_dir=runs',
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload['models'] == ['gpt2-large', 'qwen3-8b']
    assert payload['n_tests'] == 5
    assert len(payload['planned_runs']) == 2


def test_structural_alias_rejects_argparse_flags() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'src',
            'structural',
            'plan',
            '--models',
            'gpt2-large',
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert 'Structural argparse flags are no longer supported' in result.stderr
    assert 'structural.run.models=[gpt2-large]' in result.stderr


def test_hydra_structural_plan_outputs_json() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'src',
            'command=structural/plan',
            'structural.run.models=[gpt2-large,qwen3-8b]',
            'structural.run.n_tests=5',
            'structural.run.output_dir=runs',
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload['models'] == ['gpt2-large', 'qwen3-8b']
    assert payload['n_tests'] == 5
    assert len(payload['planned_runs']) == 2


def test_hydra_methods_command_lists_rome() -> None:
    result = subprocess.run(
        [sys.executable, '-m', 'src', 'command=methods'],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert 'Editing methods:' in result.stdout
    assert 'rome' in result.stdout
