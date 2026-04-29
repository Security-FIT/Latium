#!/usr/bin/env python3
"""Run the Qwen3-8B finetune fleet sequentially.

For each model in ``finetuned_qwen3_8b_fleet.json`` this performs:
baseline structural run -> find one successful edit case -> edited structural run.
Fleet model configs are resolved through ``src.model_config``; no generated
``src/config/model/fleet_*.yaml`` files are needed.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from src.model_config import fleet_model_key

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

DEFAULT_FLEET_FILE = _REPO / "finetuned_qwen3_8b_fleet.json"
IDLE_TIMEOUT_RC = 124


def _run(cmd: list[str], logfile: Path, idle_timeout_sec: int | None = None) -> int:
    LOGGER.info("Running: %s", " ".join(cmd))
    logfile = logfile.resolve()
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== COMMAND: {' '.join(cmd)} ===\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=_REPO,
            start_new_session=True,
        )
        last_progress = time.monotonic()
        last_mtime = logfile.stat().st_mtime if logfile.exists() else None

        while True:
            rc = proc.poll()
            if rc is not None:
                return rc

            time.sleep(30)
            try:
                current_mtime = logfile.stat().st_mtime
            except FileNotFoundError:
                current_mtime = last_mtime
            if current_mtime != last_mtime:
                last_progress = time.monotonic()
                last_mtime = current_mtime

            if idle_timeout_sec is not None and (time.monotonic() - last_progress) > idle_timeout_sec:
                msg = f"Idle timeout after {idle_timeout_sec}s; terminating: {' '.join(cmd)}\n"
                LOGGER.warning(msg.strip())
                handle.write(msg)
                handle.flush()
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
                return IDLE_TIMEOUT_RC


def _cleanup_hf_hub_cache() -> None:
    hub_cache = Path.home() / ".cache" / "huggingface" / "hub"
    try:
        if hub_cache.exists():
            shutil.rmtree(hub_cache)
        hub_cache.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        LOGGER.warning("Failed to clean HuggingFace hub cache at %s: %s", hub_cache, exc)


def _cleanup_model_cache(model_id: str) -> None:
    models_root = (Path.home() / "models").resolve()
    model_path = (models_root / model_id).resolve()
    try:
        model_path.relative_to(models_root)
    except ValueError:
        LOGGER.warning("Refusing to clean model path outside %s: %s", models_root, model_path)
        return
    if not model_path.exists():
        return
    try:
        shutil.rmtree(model_path)
    except OSError as exc:
        LOGGER.warning("Failed to clean staged model cache at %s: %s", model_path, exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fleet", default=str(DEFAULT_FLEET_FILE))
    parser.add_argument("--start", type=int, default=0, help="Slice start, inclusive")
    parser.add_argument("--end", type=int, default=None, help="Slice end, exclusive")
    parser.add_argument("--out-dir", default="./fleet_out")
    parser.add_argument("--n-tests-baseline", type=int, default=30)
    parser.add_argument("--max-case-hunt", type=int, default=40)
    parser.add_argument("--analysis-profile", default="paper")
    parser.add_argument("--idle-timeout-minutes", type=float, default=30.0)
    parser.add_argument("--keep-model-cache", action="store_true")
    args = parser.parse_args()

    fleet_file = Path(args.fleet)
    if not fleet_file.is_absolute():
        fleet_file = (_REPO / fleet_file).resolve()
    if not fleet_file.exists():
        raise FileNotFoundError(f"Fleet manifest not found: {fleet_file}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (_REPO / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    success_dir = out_dir / "success_cases"
    success_dir.mkdir(parents=True, exist_ok=True)

    fleet = json.loads(fleet_file.read_text(encoding="utf-8"))
    models = fleet.get("models", [])
    end = args.end if args.end is not None else len(models)
    subset = models[args.start:end]

    LOGGER.info("Fleet runner: models %d to %d (selected %d of %d)", args.start, end, len(subset), len(models))
    idle_timeout_sec = int(args.idle_timeout_minutes * 60) if args.idle_timeout_minutes > 0 else None

    for offset, entry in enumerate(subset):
        global_idx = args.start + offset
        model_id = str(entry["model_id"])
        model_key = fleet_model_key(model_id)
        logfile = log_dir / f"{model_key}.log"
        _cleanup_hf_hub_cache()

        LOGGER.info("[%d/%d] ===== %s =====", global_idx + 1, len(models), model_id)

        baseline_cmd = [
            sys.executable,
            "structural_benchmark.py",
            "--models",
            model_key,
            "--baseline-only",
            "--n-tests",
            str(args.n_tests_baseline),
            "--analysis-profile",
            args.analysis_profile,
            "--output-dir",
            str(out_dir),
        ]
        rc = _run(baseline_cmd, logfile, idle_timeout_sec=idle_timeout_sec)
        if rc == IDLE_TIMEOUT_RC:
            LOGGER.warning("[%d] Baseline idle-timed out for %s. Skipping.", global_idx, model_id)
            _cleanup_hf_hub_cache()
            if not args.keep_model_cache:
                _cleanup_model_cache(model_id)
            continue
        if rc != 0:
            LOGGER.warning("[%d] Baseline failed for %s (rc=%d)", global_idx, model_id, rc)

        success_manifest = success_dir / f"{model_key}.json"
        hunt_cmd = [
            sys.executable,
            "scripts/find_successful_case.py",
            "--model",
            model_key,
            "--max-cases",
            str(args.max_case_hunt),
            "--start-idx",
            str(global_idx * 50),
            "--seed",
            str(42 + global_idx),
            "--output-manifest",
            str(success_manifest),
        ]
        rc = _run(hunt_cmd, logfile, idle_timeout_sec=idle_timeout_sec)
        if rc != 0:
            LOGGER.warning("[%d] Case hunt failed for %s (rc=%d). Skipping edited run.", global_idx, model_id, rc)
            _cleanup_hf_hub_cache()
            if not args.keep_model_cache:
                _cleanup_model_cache(model_id)
            continue

        edited_cmd = [
            sys.executable,
            "structural_benchmark.py",
            "--models",
            model_key,
            "--case-index-file",
            str(success_manifest),
            "--n-tests",
            "1",
            "--analysis-profile",
            args.analysis_profile,
            "--output-dir",
            str(out_dir),
        ]
        rc = _run(edited_cmd, logfile, idle_timeout_sec=idle_timeout_sec)
        if rc != 0:
            LOGGER.warning("[%d] Edited run failed for %s (rc=%d)", global_idx, model_id, rc)

        LOGGER.info("[%d] Done with %s", global_idx, model_id)
        _cleanup_hf_hub_cache()
        if not args.keep_model_cache:
            _cleanup_model_cache(model_id)

    LOGGER.info("Fleet runner finished slice %d:%d", args.start, end)


if __name__ == "__main__":
    main()
