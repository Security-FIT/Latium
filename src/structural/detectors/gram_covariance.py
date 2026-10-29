"""Explicit all-layer covariance/precision inputs for joint GRAM detection.

ROME's local .pt second-moment cache stores PRECISION matrices, whereas the
upstream .npz 'mom2.mom2' field stores C. Never guess semantics from a tensor.
This loader does not compute missing covariances or consult a target layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Metric:
    matrix: torch.Tensor
    kind: str = "covariance"

    def gram(self, weight: torch.Tensor) -> torch.Tensor:
        if not self.matrix.dtype.is_floating_point:
            raise TypeError("Covariance/precision must use a floating-point dtype")
        source_epsilon = torch.finfo(self.matrix.dtype).eps
        metric = self.matrix.detach().to(device=weight.device, dtype=torch.float64)
        if self.kind not in ("covariance", "precision"):
            raise ValueError("Metric kind must be covariance or precision")
        if metric.shape != (weight.shape[1], weight.shape[1]) or not bool(torch.isfinite(metric).all()):
            raise ValueError("Covariance/precision shape or values do not match projection inputs")
        tolerance = source_epsilon * metric.shape[0] * max(float(metric.abs().max()), torch.finfo(torch.float64).tiny)
        if not torch.allclose(metric, metric.T, atol=tolerance, rtol=0):
            raise ValueError("Covariance/precision must be symmetric")
        metric = (metric + metric.T) * 0.5
        # Precision must be invertible; a covariance may be singular within
        # numerical precision. Validate before projecting, without changing C.
        if self.kind == "precision":
            factor = torch.linalg.cholesky(metric)
            result = weight @ torch.cholesky_solve(weight.T, factor)
        else:
            _, info = torch.linalg.cholesky_ex(
                metric + tolerance * torch.eye(len(metric), device=metric.device, dtype=metric.dtype)
            )
            if int(info):
                raise ValueError("Covariance must be positive semidefinite")
            result = weight @ metric @ weight.T
        energy = float(torch.trace(result))
        if not np.isfinite(energy) or energy <= 0:
            raise ValueError("Covariance-weighted projection has no positive finite energy")
        return ((result + result.T) * (0.5 / energy)).to(torch.float32)


class CovarianceManifest:
    """Lazy input: loads one metric at a time, with complete declared coverage."""

    def __init__(self, path: str | Path, layers: Sequence[int]) -> None:
        self.path = Path(path).expanduser().resolve()
        manifest_bytes = self.path.read_bytes()
        self.manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        raw = json.loads(manifest_bytes.decode("utf-8"))
        if not isinstance(raw, dict) or raw.get("version") != 1 or not isinstance(raw.get("layers"), dict):
            raise ValueError("Covariance manifest needs version:1 and a layers mapping")
        if set(raw["layers"]) != {str(layer) for layer in layers}:
            raise ValueError(
                "Covariance manifest must cover every scanned layer exactly; partial target-layer input is forbidden"
            )
        self.entries = {}
        for layer in layers:
            entry = raw["layers"][str(layer)]
            if not isinstance(entry, dict) or set(entry) - {"path", "kind", "key", "sha256"}:
                raise ValueError("Each covariance manifest entry needs explicit path/kind fields")
            kind = entry.get("kind", raw.get("kind"))
            if kind not in ("covariance", "precision") or not isinstance(entry.get("path"), str):
                raise ValueError("Declare covariance or precision explicitly for every metric")
            location = Path(entry["path"]).expanduser()
            location = (self.path.parent / location).resolve() if not location.is_absolute() else location.resolve()
            if not location.is_file():
                raise FileNotFoundError(f"Missing covariance metric for scanned layer {layer}: {location}")
            self.entries[layer] = {**entry, "path": str(location), "kind": kind}
        self.provenance = str(raw.get("provenance", "unspecified"))
        self.hashes: dict[str, str] = {}

    def metric(self, layer: int) -> Metric:
        entry = self.entries[layer]
        path = Path(entry["path"])
        digest = _sha256(path)
        if entry.get("sha256") is not None and digest != entry["sha256"]:
            raise ValueError(f"Covariance SHA256 mismatch for layer {layer}")
        previous = self.hashes.get(str(layer))
        if previous is not None and digest != previous:
            raise ValueError(f"Covariance file changed during capture for layer {layer}")
        self.hashes[str(layer)] = digest
        if path.suffix.lower() == ".npz":
            key = entry.get("key", "mom2.mom2")
            with np.load(path, allow_pickle=False) as archive:
                value = torch.from_numpy(np.asarray(archive[key]).copy())
        else:
            value = torch.load(path, map_location="cpu", weights_only=True)
            if "key" in entry:
                value = value[entry["key"]]
            if not isinstance(value, torch.Tensor):
                raise ValueError("Metric tensor file needs a tensor or an explicit mapping key")
        return Metric(value, entry["kind"])

    def identity(self) -> dict:
        if _sha256(self.path) != self.manifest_hash:
            raise ValueError("Covariance manifest changed during capture")
        return {
            "mode": "all-layer-manifest",
            "manifest": str(self.path),
            "manifest_sha256": self.manifest_hash,
            "provenance": self.provenance,
            "metric_sha256": dict(self.hashes),
            "layers": {
                str(layer): {"path": entry["path"], "kind": entry["kind"]} for layer, entry in self.entries.items()
            },
        }


def validate_covariances(covariances: Mapping[int, torch.Tensor | Metric], layers: Sequence[int]) -> None:
    if set(covariances) != set(layers) or any(type(layer) is not int for layer in covariances):
        raise ValueError("Covariance input must cover every scanned layer exactly")
    if any(not isinstance(value, (torch.Tensor, Metric)) for value in covariances.values()):
        raise TypeError("Covariance inputs must be tensors (C) or explicitly typed Metric objects")


def manifest_signature(path: str | Path) -> dict:
    """Content identity for capture caching, including every declared metric.

    No weight load, covariance computation, target-layer lookup or model forward
    is involved. Content changes invalidate cached captures even at the same path.
    """
    location = Path(path).expanduser().resolve()
    raw = json.loads(location.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("layers"), dict):
        raise ValueError("Invalid covariance manifest")
    try:
        layers = sorted(int(layer) for layer in raw["layers"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid covariance manifest layer keys") from exc
    manifest = CovarianceManifest(location, layers)
    for layer, entry in manifest.entries.items():
        digest = _sha256(Path(entry["path"]))
        if entry.get("sha256") is not None and entry["sha256"] != digest:
            raise ValueError(f"Covariance SHA256 mismatch for layer {layer}")
        manifest.hashes[str(layer)] = digest
    return manifest.identity()
