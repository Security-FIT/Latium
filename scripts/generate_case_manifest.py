#!/usr/bin/env python3
"""Freeze a shared random CounterFact cohort once, before launching workers."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from huggingface_hub import HfApi
from src.counterfact_selection import generate_random_case_manifest, write_case_manifest

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="azhx/counterfact")
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        parser.error("Output already exists; reuse the frozen cohort or choose another path")
    revision = HfApi().dataset_info(args.dataset, revision=args.revision).sha
    payload = generate_random_case_manifest(count=args.count, seed=args.seed,
        dataset_name=args.dataset, split=args.split, revision=revision)
    path = write_case_manifest(args.output, payload)
    print(f"{path}: {payload['count']} facts, revision={revision}, hash={payload['manifest_hash']}")

if __name__ == "__main__":
    main()
