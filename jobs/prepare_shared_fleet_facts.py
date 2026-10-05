#!/usr/bin/env python3
"""Assign one shared CounterFact per checkpoint and a disjoint tracing pool."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.counterfact_selection import load_case_manifest, manifest_digest
from src.gram_experiment import write_json


def prepare_shared_facts(facts_source, fleets_root, output_dir, count=100):
    source = load_case_manifest(facts_source)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    if not 0 < count < source["count"]:
        raise ValueError("The source must contain ROME facts and separate tracing facts")
    fleets = sorted(Path(fleets_root).glob("*/checkpoints.json"))
    if not fleets:
        raise ValueError("No frozen family manifests found")
    families = [(path.parent.name, json.loads(path.read_text())) for path in fleets]
    if any(len(fleet["models"]) > count for _, fleet in families):
        raise ValueError("The shared fact pool must cover every frozen checkpoint")

    def subset(start, stop):
        result = {**source, "count": stop - start, "source_manifest_hash": source["manifest_hash"],
                  "source_range": [start, stop]}
        for key in ("indices", "case_ids", "content_hashes"):
            if key in result:
                result[key] = source[key][start:stop]
        result["manifest_hash"] = manifest_digest(result)
        return result

    rome, tracing = subset(0, count), subset(count, source["count"])
    write_json(output / "rome-facts.json", rome)
    write_json(output / "trace-facts.json", tracing)
    for family, fleet in families:
        models = [{**record, "fact_position": position, "fact_case_id": rome["case_ids"][position]}
                  for position, record in enumerate(fleet["models"])]
        write_json(output / "fleets" / family / "checkpoints.json", {
            **fleet, "models": models, "rome_manifest_hash": rome["manifest_hash"],
            "trace_manifest_hash": tracing["manifest_hash"],
            "fact_assignment": "one-shared-fact-per-frozen-checkpoint-v1"})
    return {"rome_facts": count, "trace_facts": tracing["count"], "families": len(families)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts-source", required=True)
    parser.add_argument("--fleets-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(prepare_shared_facts(args.facts_source, args.fleets_root, args.output_dir, args.count)))


if __name__ == "__main__":
    main()
