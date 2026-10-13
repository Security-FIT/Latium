"""Small shared catalog for appendable Gram batches using existing run artifacts."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        temporary = Path(stream.name)
    temporary.replace(path)


@contextmanager
def locked(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".experiment.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def register_batch(root, cohort, models, start, stop, setup):
    """Exact retries are idempotent; overlaps and incompatible setups fail before edits."""
    root = Path(root)
    if start < 0 or stop <= start or stop > cohort["count"]:
        raise ValueError("Invalid manifest position range")
    batch = f"m{start:04d}-{stop:04d}"
    with locked(root):
        path = root / "experiment.json"
        if path.exists():
            catalog = json.loads(path.read_text())
            if catalog["cohort_hash"] != cohort["manifest_hash"] or catalog["setup"] != setup:
                raise ValueError("Experiment cohort or Gram setup changed; use a new run root")
            if json.loads((root / "cases.json").read_text()) != cohort:
                raise ValueError("Frozen experiment cases.json changed")
        else:
            catalog = {"schema_version": 1, "workflow": "gram", "cohort_hash": cohort["manifest_hash"],
                       "setup": setup, "models": {}}
            write_json(root / "cases.json", cohort)
        for model, identity in models.items():
            entry = catalog["models"].setdefault(model, {"identity": identity, "run_root": f"models/{model}/run", "batches": {}})
            if entry["identity"] != identity:
                raise ValueError(f"Model configuration changed: {model}; use a new run root")
            for key, prior in entry["batches"].items():
                if key != batch and start < prior["stop"] and stop > prior["start"]:
                    raise ValueError(f"Overlapping range for {model}: {batch} and {key}")
            entry["batches"][batch] = {"start": start, "stop": stop,
                "plan_id": f"cf_{cohort['manifest_hash'][:12]}_{batch}_r01"}
        write_json(path, catalog)
    return batch, catalog
