#!/usr/bin/env python3
"""Summarize metric_brain_adapter.py outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def summarize_metric(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    run_name = path.parent.name
    summary = data["summary_statistics"]
    item: dict[str, Any] = {
        "run_name": run_name,
        "metric_path": str(path),
        "num_samples": data["evaluation_metadata"]["num_samples"],
    }
    for metric_name, values in summary.items():
        key = metric_name.replace("(", "_").replace(")", "").replace("/", "_")
        item[f"{key}_mean"] = values["mean"]
        item[f"{key}_std"] = values["std"]
    return item


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metric_json", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    rows = [summarize_metric(path) for path in args.metric_json]
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with args.json_out.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
