#!/usr/bin/env python3
"""Create mean-replacement robustness conditions for aggregate ROI groups."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--roi-groups", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
groups = json.loads(args.roi_groups.read_text(encoding="utf-8"))["groups"]
args.output.write_text(json.dumps({"conditions": [
    {"name": "no_mask", "masked_token_indices": [], "mask_mode": "zero"},
    {"name": "low_level_mean", "masked_token_indices": groups["low_level"], "mask_mode": "mean"},
    {"name": "high_level_mean", "masked_token_indices": groups["high_level"], "mask_mode": "mean"},
]}, indent=2), encoding="utf-8")
