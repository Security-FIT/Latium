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

from omegaconf import OmegaConf
import pytest

from src.structural.config import StructuralBenchmarkConfig
from src.common.config import strict_bool
from src.structural.hydra_config import structural_config_from_hydra
from src.structural.planning import (
    build_analysis_variants,
    build_plan_summary,
    normalize_models_arg,
    parse_int_values,
    parse_local_window_sets,
    parse_local_windows,
    parse_trim_values,
)

ROOT = Path(__file__).resolve().parents[1]


def test_normalize_models_arg_uses_native_lists_and_deduplicates() -> None:
    assert normalize_models_arg(['gpt2-large', 'qwen3-8b', 'gpt2-large', ' mistral-7b-v0.1 ']) == [
        'gpt2-large',
        'qwen3-8b',
        'mistral-7b-v0.1',
    ]

    with pytest.raises(ValueError, match="YAML list"):
        normalize_models_arg(['gpt2-large,qwen3-8b'])


def test_structural_parsers_accept_native_sequences() -> None:
    assert parse_local_windows([3, 5, 7, 7]) == [3, 5, 7]
    assert parse_int_values([2, 5, 5], default=[1], min_value=1) == [2, 5]
    assert parse_trim_values([None, 3, 3], default=[None]) == [None, 3]
    assert parse_local_window_sets([[3, 5, 7], [9, 11], [11, 13]]) == [
        (3, 5, 7),
        (9, 11),
        (11, 13),
    ]


@pytest.mark.parametrize(
    "call",
    [
        lambda: parse_local_windows([3, 4, 7]),
        lambda: parse_int_values([2, '5'], default=[1], min_value=1),
        lambda: parse_trim_values([None, 'auto'], default=[None]),
        lambda: parse_local_window_sets([3, 5, 7]),
    ],
)
def test_structural_parsers_reject_legacy_or_invalid_values(call) -> None:
    with pytest.raises((TypeError, ValueError)):
        call()


def test_structural_config_from_hydra_uses_native_lists() -> None:
    structural = OmegaConf.load(ROOT / 'src/config/structural/default.yaml')
    dataset_facts = OmegaConf.load(ROOT / 'src/config/dataset_facts/counterfact.yaml')
    structural.run.models = ['gpt2-large', 'qwen3-8b', 'gpt2-large']
    structural.analysis.variants.local_windows = [3, 5, 7]
    structural.analysis.variants.sweep.spectral_top_k = [20, 50]
    structural.analysis.variants.sweep.trim_first = [None, 2]
    structural.analysis.variants.sweep.trim_last = [None, 1]
    structural.analysis.variants.sweep.spectral_neighbor_layers = [1]
    structural.analysis.variants.sweep.spectral_rolling_window = [5, 7]
    structural.analysis.variants.sweep.local_window_sets = [[3, 5, 7], [5, 7, 9]]
    structural.analysis.variants.sweep.max_configs = 2
    cfg = OmegaConf.create(
        {
            'seed': 42,
            'runtime': {
                'hf_token': None,
                'prefix_log_all': False,
                'second_moment_allow_autocompute': True,
                'log_skip_traceback': False,
            },
            'dataset_facts': dataset_facts,
            'structural': structural,
        }
    )

    config = structural_config_from_hydra(cfg, run_analysis=False)

    assert config.models == ('gpt2-large', 'qwen3-8b')
    assert config.local_windows == (3, 5, 7)
    assert config.run_analysis is False
    assert config.seed == 42
    assert config.case_dataset_name == 'azhx/counterfact'
    assert config.case_dataset_split == 'train'
    assert config.analysis_method_configs['composite']['small_window'] == 5
    assert [variant.spectral_top_k for variant in config.analysis_variants] == [20, 50]
    assert [variant.trim_first for variant in config.analysis_variants] == [None, 2]
    assert [variant.trim_last for variant in config.analysis_variants] == [None, 1]
    assert [variant.spectral_rolling_window for variant in config.analysis_variants] == [5, 7]
    assert [variant.local_windows for variant in config.analysis_variants] == [(3, 5, 7), (5, 7, 9)]


def test_structural_config_rejects_invalid_bounds_and_booleans() -> None:
    with pytest.raises(ValueError, match="n_tests"):
        StructuralBenchmarkConfig(n_tests=-3)
    with pytest.raises(ValueError, match="local_windows"):
        StructuralBenchmarkConfig(local_windows=(3, 4, 7))
    with pytest.raises(ValueError, match="boolean"):
        StructuralBenchmarkConfig(force="sometimes")
    with pytest.raises(ValueError, match="native list"):
        StructuralBenchmarkConfig(models="gpt2-large,qwen3-8b")

    assert StructuralBenchmarkConfig(force="false").force is False
    assert StructuralBenchmarkConfig(force="true").force is True


def test_strict_bool_rejects_python_string_truthiness() -> None:
    assert strict_bool("false") is False
    assert strict_bool("yes") is True
    with pytest.raises(ValueError, match="boolean"):
        strict_bool("not-a-bool")


def test_build_analysis_variants_zip_broadcasts_singletons() -> None:
    configs = build_analysis_variants(
        spectral_top_k_values=[50, 80],
        trim_first_values=[None],
        trim_last_values=[2],
        spectral_neighbor_layers_values=[1],
        spectral_rolling_window_values=[5, 7],
        local_window_sets=[(3, 5, 7)],
        mode='zip',
    )

    assert [cfg.spectral_top_k for cfg in configs] == [50, 80]
    assert [cfg.spectral_rolling_window for cfg in configs] == [5, 7]
    assert all(cfg.trim_first is None for cfg in configs)
    assert all(cfg.trim_last == 2 for cfg in configs)


def test_build_plan_summary_generates_stable_plan_ids() -> None:
    config = StructuralBenchmarkConfig(
        models=('gpt2-large',),
        n_tests=5,
        start_idx=30,
        runs_per_model=2,
        run_start_idx_step=10,
        output_dir=Path('runs'),
        capture_profile='spectral',
        analysis_variants=tuple(
            build_analysis_variants(
                spectral_top_k_values=[50],
                trim_first_values=[None],
                trim_last_values=[None],
                spectral_neighbor_layers_values=[1],
                spectral_rolling_window_values=[5],
                local_window_sets=[(3, 5, 7)],
            )
        ),
    )

    payload = build_plan_summary(config, run_id='dry-run')

    assert payload['run_id'] == 'dry-run'
    assert payload['models'] == ['gpt2-large']
    assert len(payload['planned_runs']) == 2
    assert len(payload['analysis_variants']) == 1
    assert payload['planned_runs'][0]['start_idx'] == 30
    assert payload['planned_runs'][0]['end_idx'] == 34
    assert payload['planned_runs'][1]['start_idx'] == 40
    assert payload['planned_runs'][1]['end_idx'] == 44
    assert payload['planned_runs'][0]['plan_id'] == 'cases30-34_r01'
    assert payload['planned_runs'][1]['plan_id'] == 'cases40-44_r02'


def test_analysis_variants_do_not_multiply_capture_plans() -> None:
    config = StructuralBenchmarkConfig(
        models=('gpt2-large',),
        n_tests=2,
        runs_per_model=1,
        analysis_variants=tuple(
            build_analysis_variants(
                spectral_top_k_values=[20, 50],
                trim_first_values=[1],
                trim_last_values=[1],
                spectral_neighbor_layers_values=[1],
                spectral_rolling_window_values=[5, 7],
                local_window_sets=[(3, 5, 7)],
            )
        ),
    )

    payload = build_plan_summary(config, run_id='variants')

    assert len(payload['planned_runs']) == 1
    assert len(payload['analysis_variants']) == 2


def test_importing_planning_module_does_not_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'import src.structural.planning; import sys, json; print(json.dumps({"torch": "torch" in sys.modules}))',
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout.strip())
    assert payload == {'torch': False}
