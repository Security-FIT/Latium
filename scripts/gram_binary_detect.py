"""Scan one checkpoint for a rank-one GRAM anomaly, without editing it."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json  # noqa: E402
from src.structural.detectors.gram_binary import detect  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--weights", type=Path, help="torch.save({integer_layer: down_projection_tensor})")
    source.add_argument("--model", help="Existing Latium model config; scans its checkpoint without ROME or CT")
    parser.add_argument("--out", type=Path, required=True, help="Full JSON result and per-layer bit costs")
    parser.add_argument("--device", help="Compute device, e.g. cpu or cuda:0; defaults to projection device")
    parser.add_argument("--method", choices=("v1", "v2"), default="v2", help="v2 adaptive MDL or frozen PBS v1")
    parser.add_argument(
        "--covariance-manifest", type=Path, help="Explicit all-layer covariance/precision JSON manifest"
    )
    parser.add_argument("--no-dyson", action="store_true", help="Disable Dyson diagnostics and its candidate generator")
    parser.add_argument("--no-bcv", action="store_true", help="Disable the held-out rank diagnostic")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.weights is not None:
        weights = torch.load(args.weights, map_location="cpu", weights_only=True)
        if not isinstance(weights, dict) or not all(isinstance(value, torch.Tensor) for value in weights.values()):
            raise ValueError("--weights must contain a mapping of layer indices to projection tensors")
    else:
        from src.common.loading import load_pretrained
        from src.structural.execution.weights import build_cfg

        cfg = build_cfg(args.model, runtime={"second_moment_allow_autocompute": False})
        model, _tokenizer = load_pretrained(cfg)
        model_config = model.config
        count = getattr(model_config, "num_hidden_layers", None)
        if count is None:
            count = getattr(getattr(model_config, "text_config", None), "num_hidden_layers", None)
        if count is None:
            raise ValueError("Model config does not expose its number of hidden layers")
        template = str(cfg.model.layer_name_template)
        weights = {layer: model.get_submodule(template.format(layer)).weight.detach() for layer in range(int(count))}
    with torch.no_grad():
        result = detect(
            weights,
            device=args.device,
            method=args.method,
            covariance_manifest=args.covariance_manifest,
            dyson=not args.no_dyson,
            bcv=not args.no_bcv,
        )
    write_json(args.out, result)
    print(
        json.dumps(
            {key: result[key] for key in ("edited", "layer", "mdl_delta_bits", "null_bits", "edit_bits", "method")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
