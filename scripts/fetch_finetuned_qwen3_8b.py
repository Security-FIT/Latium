#!/usr/bin/env python3
"""Fetch the top HuggingFace finetunes of Qwen/Qwen3-8B-Base.

The output manifest is consumed by ``scripts/fleet_runner.py`` and by the
manifest-backed resolver in ``src/model_config.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B-Base"
DEFAULT_OUTPUT = "finetuned_qwen3_8b_fleet.json"
HF_API_MODELS = "https://huggingface.co/api/models"


def _request_json(url: str, *, token: str | None = None) -> list[dict[str, Any]]:
    headers = {"User-Agent": "latium-fleet-fetch/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected HuggingFace API response: {type(payload).__name__}")
    return payload


def _model_record(raw: dict[str, Any]) -> dict[str, Any]:
    model_id = str(raw.get("id") or raw.get("modelId") or "").strip()
    if not model_id:
        raise ValueError(f"Model record is missing id/modelId: {raw!r}")
    return {
        "model_id": model_id,
        "downloads": int(raw.get("downloads") or 0),
        "likes": int(raw.get("likes") or 0),
        "tags": [str(tag) for tag in raw.get("tags", [])],
    }


def fetch_models(*, base_model: str, limit: int, token: str | None = None) -> list[dict[str, Any]]:
    params = {
        "other": f"base_model:finetune:{base_model}",
        "sort": "downloads",
        "direction": "-1",
        "limit": str(limit),
        "full": "true",
    }
    url = f"{HF_API_MODELS}?{urllib.parse.urlencode(params)}"
    records = [_model_record(item) for item in _request_json(url, token=token)]
    records.sort(key=lambda item: (-int(item["downloads"]), str(item["model_id"]).lower()))
    return records[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN", ""))
    args = parser.parse_args()

    models = fetch_models(base_model=args.base_model, limit=args.limit, token=args.token or None)
    payload = {
        "source": (
            "https://huggingface.co/models?"
            + urllib.parse.urlencode({"other": f"base_model:finetune:{args.base_model}"})
        ),
        "base_model": args.base_model,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count": len(models),
        "models": models,
    }

    out_path = Path(args.output)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
