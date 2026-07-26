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
    parser.add_argument("--source-e2-plan", type=Path)
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
        artifact_dir = args.output_root / name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "plan.json").write_text(
            json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        source = (
            json.loads(args.source_e2_plan.read_text(encoding="utf-8"))
            if args.source_e2_plan is not None
            else None
        )
        equivalence = {
            "passed": source is not None,
            "source_e2_plan": (
                str(args.source_e2_plan.resolve())
                if args.source_e2_plan is not None
                else None
            ),
            "source_e2_plan_sha256": (
                file_sha256(args.source_e2_plan)
                if args.source_e2_plan is not None
                else None
            ),
            "checks": {},
        }
        if source is not None:
            equivalence["checks"] = {
                "seeds_identical": plan["seeds"] == source["seeds"],
                "denoising_steps_identical": (
                    plan["denoising_steps"] == source["denoising_steps"]
                ),
                "noise_factor_identical": (
                    plan["noise_factor"] == source["noise_factor"]
                ),
                "condition_batch_size_identical": (
                    plan["condition_batch_size"] == source["condition_batch_size"]
                ),
                "dataset_indices_identical": all(
                    plan["categories"][category]["dataset_indices"]
                    == source["categories"][category]["dataset_indices"]
                    for category in ROI_GROUPS
                ),
            }
            equivalence["passed"] = all(equivalence["checks"].values())
        (artifact_dir / "plan_equivalence_audit.json").write_text(
            json.dumps(equivalence, indent=2), encoding="utf-8"
        )
        control_rows = []
        for category, category_plan in plan["categories"].items():
            audit_groups = (
                category_plan["matching_audit"].get("roi_designs")
                or category_plan["matching_audit"]["joint_designs"]
            )
            for label, design in audit_groups.items():
                controls = design["pure_random_controls"]
                control_rows.append(
                    {
                        "image_category": category,
                        "masked_roi": label,
                        "target_count": design["target_count"],
                        "control_replicates": len(controls),
                        "unique_control_sets": len(
                            {tuple(item["indices"]) for item in controls}
                        ),
                        "max_target_group_overlap": max(
                            item["max_target_group_overlap"] for item in controls
                        ),
                        "max_matching_distance": max(
                            item["max_distance"] for item in controls
                        ),
                        "passed": (
                            len(controls) == args.replicates
                            and len({tuple(item["indices"]) for item in controls})
                            == args.replicates
                            and all(
                                len(item["indices"]) == design["target_count"]
                                and item["max_target_group_overlap"]
                                < args.purity_threshold
                                for item in controls
                            )
                        ),
                    }
                )
        control_audit = {
            "passed": all(item["passed"] for item in control_rows),
            "purity_rule": plan["pure_control_overlap_rule"],
            "conditions": control_rows,
        }
        (artifact_dir / "control_matching_audit.json").write_text(
            json.dumps(control_audit, indent=2), encoding="utf-8"
        )
        if not equivalence["passed"] or not control_audit["passed"]:
            raise RuntimeError(f"{name} plan audit failed")
        print(f"{name}: {plan_path}")


if __name__ == "__main__":
    main()
