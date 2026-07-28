from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "jobs" / "rome_math_n50_cluster.sh"
EXPECTED_MODELS = (
    "gpt2-xl",
    "gpt-j-6b",
    "mistral-7b-v0.1",
    "mistral-7b-v0.3",
    "deepseek-7b-base",
    "falcon-7b",
    "opt-6.7b",
    "llama2-7b",
    "granite4-micro",
)


def test_n50_job_contains_exact_cohort_and_detector_only_command() -> None:
    source = JOB.read_text(encoding="utf-8")
    model_block = re.search(
        r"^MODELS=\(\n(?P<body>.*?)^\)$",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )

    assert model_block is not None
    assert tuple(model_block.group("body").split()) == EXPECTED_MODELS
    assert "structural.run.edit_methods=[rome]" in source
    assert "structural.run.n_tests=$count" in source
    assert "structural.run.start_idx=0" in source
    assert "structural.run.fail_on_missing_second_moment=true" in source
    assert "structural.capture.profile=rome-math-ablation" in source
    assert "structural.analysis.preset=none" in source
    assert "structural.render.enabled=false" in source
    assert "capture.profile=detection" not in source
    assert "causal-rome" not in source
    assert '[[ -n "${TMUX:-}" ]] || die "direct backend requires an existing tmux session"' in source
    assert 'python3 - "$trace_root" "$layer" "$MAIN_ROOT"' in source
    assert "plot = root_checkout / plot" in source


def test_n50_first_model_dry_run_resolves_exact_isolated_command(
    tmp_path: Path,
) -> None:
    detector = tmp_path / "detector"
    (detector / "jobs").mkdir(parents=True)
    (detector / "manifests").mkdir()
    (detector / "data" / "second_moment_stats").mkdir(parents=True)
    (detector / ".git").mkdir()
    (detector / "jobs" / "submit.sh").write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\"\n",
        encoding="utf-8",
    )
    (detector / "jobs" / "submit.sh").chmod(0o755)
    covariance = detector / "data" / "second_moment_stats" / "gpt2-xl_17_SM_Method.WIKIPEDIA_100000.pt"
    covariance.write_bytes(b"fixture")
    (detector / "manifests" / "rome_math_ablation_n50_cluster.yaml").write_text(
        """
models:
  - model: gpt2-xl
    dependency_decision: reuse_exact
    covariance:
      status: verified
      path: ./data/second_moment_stats/gpt2-xl_17_SM_Method.WIKIPEDIA_100000.pt
""".lstrip(),
        encoding="utf-8",
    )

    result = subprocess.run(
        [str(JOB), "--dry-run", "n50", "gpt2-xl"],
        cwd=ROOT,
        env={**os.environ, "LATIUM_DETECTOR_ROOT": str(detector)},
        check=True,
        capture_output=True,
        text=True,
    )

    command = result.stdout
    assert "custom" in command
    assert "structural.run.models=[gpt2-xl]" in command
    assert "structural.run.edit_methods=[rome]" in command
    assert "structural.run.n_tests=50" in command
    assert "structural.run.start_idx=0" in command
    assert "structural.run.run_id=rome-math-n50-gpt2-xl-v1" in command
    assert "structural.capture.profile=rome-math-ablation" in command
    assert "structural.analysis.preset=none" in command
    assert "structural.render.enabled=false" in command
