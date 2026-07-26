#!/usr/bin/env python3
"""Freeze E3a interaction and E3b joint-redundancy plans."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "src"))

from neuro_roi_causal.e2 import read_csv, select_manifest_indices
from neuro_roi_causal.e3 import (
    ROI_GROUPS,
    build_interaction_category,
    build_joint_category,
)
from neuro_roi_causal.mean_cache import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mean-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=0, help="0 keeps all frozen E2 samples")
    parser.add_argument("--equal-k", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--purity-threshold", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = read_csv(args.inventory)
    manifest = read_csv(args.manifest)
    available = {"Face": 37, "Body": 50, "Scene": 50}
    counts = (
        {category: args.count for category in ROI_GROUPS}
        if args.count > 0
        else available
    )
    selected = select_manifest_indices(manifest, counts)
    common = {
        "schema_version": 1,
        "subject": 1,
        "checkpoint_step": 100000,
        "mask_mode": "mean",
        "equal_k": args.equal_k,
        "pure_control_overlap_rule": (
            f"max overlap with every target ROI group < {args.purity_threshold:.2f}"
        ),
        "pure_control_overlap_threshold": args.purity_threshold,
        "matched_random_controls": args.replicates,
        "seeds": [12345, 23456, 34567],
        "denoising_steps": 50,
        "noise_factor": 4.0,
        "condition_batch_size": 8,
        "statistical_unit": "image after averaging three seeds",
        "mean_token_cache": str(args.mean_cache.resolve()),
        "mean_token_cache_sha256": file_sha256(args.mean_cache),
        "inventory": str(args.inventory.resolve()),
        "inventory_sha256": file_sha256(args.inventory),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": file_sha256(args.manifest),
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPRO_ROOT, text=True
        ).strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preregistration_status": "engineering_implementation_before_smoke",
    }
    plans = {
        "E3_interaction": {
            **common,
            "name": "E3_interaction",
            "analysis_family": "E3a_category_by_roi_interaction",
            "categories": {},
        },
        "E3_joint_redundancy": {
            **common,
            "name": "E3_joint_redundancy",
            "analysis_family": "E3b_joint_roi_redundancy",
            "categories": {},
        },
    }
    for category_index, category in enumerate(ROI_GROUPS):
        interaction, interaction_audit = build_interaction_category(
            inventory,
            category,
            k=args.equal_k,
            replicates=args.replicates,
            seed=args.seed + category_index * 10000,
            purity_threshold=args.purity_threshold,
        )
        joint, joint_audit = build_joint_category(
            inventory,
            category,
            k=args.equal_k,
            replicates=args.replicates,
            seed=args.seed + 100000 + category_index * 10000,
            purity_threshold=args.purity_threshold,
        )
        plans["E3_interaction"]["categories"][category] = {
            "dataset_indices": selected[category],
            "conditions": interaction,
            "matching_audit": interaction_audit,
        }
        plans["E3_joint_redundancy"]["categories"][category] = {
            "dataset_indices": selected[category],
            "conditions": joint,
            "matching_audit": joint_audit,
        }
    args.output_root.mkdir(parents=True, exist_ok=True)
    for name, plan in plans.items():
        plan_path = args.output_root / f"{name}_plan.json"
        plan_path.write_text(
            json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"{name}: {plan_path}")


if __name__ == "__main__":
    main()
