#!/usr/bin/env python3
"""Select new ranked checkpoints without overwriting or rerunning older cohorts."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare_continuation(manifest, previous_runs, output, count=40):
    manifest, output = Path(manifest), Path(output)
    if output.exists():
        raise FileExistsError(f"Continuation manifest already exists: {output}")
    if count <= 0:
        raise ValueError("Checkpoint count must be positive")
    source = json.loads(manifest.read_text())
    previous = {}
    for root in map(Path, previous_runs):
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in root.glob("models/*/fleet-batches/*/state.json"):
            state = json.loads(path.read_text())
            if not state.get("model_id"):
                raise ValueError(f"Checkpoint state has no model ID: {path}")
            previous.setdefault(state["model_id"], []).append({
                "revision": state.get("revision"), "status": state.get("status"),
                "state_path": str(path.resolve()), "error": state.get("error")})
    models, excluded = [], []
    for record in source["models"]:
        if record["model_id"] in previous:
            excluded.append({"model_id": record["model_id"], "discovery_rank": record["rank"],
                             "previous_attempts": previous[record["model_id"]]})
        elif len(models) < count:
            models.append({**record, "discovery_rank": record["rank"], "rank": len(models) + 1})
    result = {**source, "models": models, "count": len(models), "requested_count": count,
              "source_manifest": str(manifest.resolve()),
              "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
              "previous_runs": [str(Path(root).resolve()) for root in previous_runs],
              "excluded_prior_models": excluded,
              "continuation_policy": "next-ranked-unattempted-repository-ids-v1"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-manifest", required=True)
    parser.add_argument("--previous-runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=40)
    args = parser.parse_args()
    result = prepare_continuation(args.models_manifest, args.previous_runs, args.output, args.count)
    print(json.dumps({"selected": result["count"], "excluded_prior": len(result["excluded_prior_models"]),
                      "output": args.output}))


if __name__ == "__main__":
    main()
