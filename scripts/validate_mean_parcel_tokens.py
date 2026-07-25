#!/usr/bin/env python3
"""Validate an E2b parcel-mean cache against its checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "src"))

from neuro_roi_causal.mean_cache import file_sha256, load_validated_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cache = load_validated_cache(
        args.cache,
        checkpoint_sha256=file_sha256(args.checkpoint),
        selected_parcel_idx=checkpoint["selected_parcel_idx"],
        subject=1,
    )
    report = {
        key: cache[key]
        for key in (
            "schema_version",
            "shape",
            "dtype",
            "subject",
            "split",
            "num_train_samples",
            "checkpoint_sha256",
            "checkpoint_step",
            "selected_parcel_idx_sha256",
            "sub_approach",
            "intervention_stage",
            "token_norm_mean",
            "token_norm_std",
            "token_value_mean",
            "token_value_std",
            "contains_nan",
            "contains_inf",
        )
    }
    report["passed"] = True
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
