#!/usr/bin/env python3
"""Freeze E3a interaction and E3b joint-redundancy plans."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

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
    parser.add_argument(
        "--interaction-config",
        type=Path,
        default=REPRO_ROOT / "configs" / "experiments" / "E3_interaction.yaml",
    )
    parser.add_argument(
        "--joint-config",
        type=Path,
        default=REPRO_ROOT / "configs" / "experiments" / "E3_joint_redundancy.yaml",
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mean-cache", type=Path, required=True)
    parser.add_argument("--source-e2-plan", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=0, help="0 keeps all frozen E2 samples")
    parser.add_argument(
        "--scope",
        choices=("full", "pilot"),
        default="full",
        help="pilot freezes the configured pilot image and seed prefix.",
    )
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def load_config(path: Path, expected_name: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required E3 config is missing: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("name") != expected_name:
        raise ValueError(f"Expected config name {expected_name}: {path}")
    required = {
        "analysis_family",
        "subject",
        "checkpoint_step",
        "categories",
        "mask_mode",
        "equal_k",
        "pure_control_overlap_threshold",
        "matched_random_controls",
        "seeds",
        "denoising_steps",
        "noise_factor",
        "condition_batch_size",
        "statistical_unit",
        "seed_aggregation",
        "global_metrics",
        "local_metrics",
        "face_detector",
        "execution",
    }
    if expected_name == "E3_interaction":
        required.update(
            {
                "masked_rois",
                "primary_model",
                "primary_contrast",
                "multiplicity",
            }
        )
    else:
        required.update({"joint_masks", "primary_contrast", "multiplicity"})
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"{path} is missing required fields: {missing}")
    if config["mask_mode"] != "mean":
        raise ValueError("E3 requires mean replacement")
    if int(config["equal_k"]) != 4:
        raise ValueError("E3 requires equal-k=4")
    if float(config["pure_control_overlap_threshold"]) != 0.10:
        raise ValueError("E3 requires pure-control overlap threshold 0.10")
    if int(config["matched_random_controls"]) != 5:
        raise ValueError("E3 requires five pure controls per condition")
    if int(config["execution"]["smoke_images_per_category"]) != 1:
        raise ValueError("E3 smoke requires one image per category")
    if int(config["execution"]["smoke_seeds"]) != 1:
        raise ValueError("E3 smoke requires one seed")
    expected_detector = {
        "backend": "opencv_haar_e1",
        "cascade_filename": "haarcascade_frontalface_default.xml",
        "scale_factor": 1.1,
        "min_neighbors": 5,
        "min_size": [24, 24],
        "pilot_fallback_allowed": False,
        "formal_fallback_allowed": False,
    }
    for key, expected in expected_detector.items():
        if config["face_detector"].get(key) != expected:
            raise ValueError(
                f"E3 face detector field {key!r} must equal {expected!r}"
            )
    cascade_sha = str(config["face_detector"].get("cascade_sha256", ""))
    if len(cascade_sha) != 64:
        raise ValueError("E3 face detector cascade_sha256 must be a SHA-256")
    return config


def assert_common_config(interaction: dict, joint: dict) -> None:
    common_keys = (
        "subject",
        "checkpoint_step",
        "categories",
        "mask_mode",
        "equal_k",
        "pure_control_overlap_threshold",
        "matched_random_controls",
        "seeds",
        "denoising_steps",
        "noise_factor",
        "condition_batch_size",
        "statistical_unit",
        "seed_aggregation",
        "global_metrics",
        "local_metrics",
        "face_detector",
        "execution",
    )
    mismatched = [key for key in common_keys if interaction[key] != joint[key]]
    if mismatched:
        raise ValueError(f"E3 configs disagree on shared fields: {mismatched}")


def main() -> None:
    args = parse_args()
    interaction_config = load_config(args.interaction_config, "E3_interaction")
    joint_config = load_config(args.joint_config, "E3_joint_redundancy")
    assert_common_config(interaction_config, joint_config)
    inventory = read_csv(args.inventory)
    manifest = read_csv(args.manifest)
    available = {
        category: int(count)
        for category, count in interaction_config["categories"].items()
    }
    if args.scope == "pilot":
        if args.count > 0:
            raise ValueError("--count cannot be combined with --scope pilot")
        pilot_count = int(
            interaction_config["execution"]["pilot_images_per_category"]
        )
        counts = {category: pilot_count for category in ROI_GROUPS}
    else:
        counts = (
            {category: args.count for category in ROI_GROUPS}
            if args.count > 0
            else available
        )
    selected = select_manifest_indices(manifest, counts)
    configured_seeds = [int(seed) for seed in interaction_config["seeds"]]
    selected_seeds = (
        configured_seeds[
            : int(interaction_config["execution"]["pilot_seeds"])
        ]
        if args.scope == "pilot"
        else configured_seeds
    )
    common = {
        "schema_version": 1,
        "plan_scope": args.scope,
        "frozen_image_count_per_category": counts,
        "subject": int(interaction_config["subject"]),
        "checkpoint_step": int(interaction_config["checkpoint_step"]),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "mask_mode": interaction_config["mask_mode"],
        "equal_k": int(interaction_config["equal_k"]),
        "pure_control_overlap_rule": (
            "max overlap with every target ROI group "
            f"< {float(interaction_config['pure_control_overlap_threshold']):.2f}"
        ),
        "pure_control_overlap_threshold": float(
            interaction_config["pure_control_overlap_threshold"]
        ),
        "matched_random_controls": int(
            interaction_config["matched_random_controls"]
        ),
        "seeds": selected_seeds,
        "denoising_steps": int(interaction_config["denoising_steps"]),
        "noise_factor": float(interaction_config["noise_factor"]),
        "condition_batch_size": int(interaction_config["condition_batch_size"]),
        "statistical_unit": interaction_config["statistical_unit"],
        "seed_aggregation": interaction_config["seed_aggregation"],
        "global_metrics": interaction_config["global_metrics"],
        "local_metrics": interaction_config["local_metrics"],
        "face_detector": interaction_config["face_detector"],
        "execution": interaction_config["execution"],
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
            "analysis_family": interaction_config["analysis_family"],
            "primary_model": interaction_config["primary_model"],
            "primary_contrast": interaction_config["primary_contrast"],
            "multiplicity": interaction_config["multiplicity"],
            "config": str(args.interaction_config.resolve()),
            "config_sha256": file_sha256(args.interaction_config),
            "categories": {},
        },
        "E3_joint_redundancy": {
            **common,
            "name": "E3_joint_redundancy",
            "analysis_family": joint_config["analysis_family"],
            "primary_contrast": joint_config["primary_contrast"],
            "multiplicity": joint_config["multiplicity"],
            "config": str(args.joint_config.resolve()),
            "config_sha256": file_sha256(args.joint_config),
            "categories": {},
        },
    }
    for category_index, category in enumerate(ROI_GROUPS):
        interaction, interaction_audit = build_interaction_category(
            inventory,
            category,
            k=int(interaction_config["equal_k"]),
            replicates=int(interaction_config["matched_random_controls"]),
            seed=args.seed + category_index * 10000,
            purity_threshold=float(
                interaction_config["pure_control_overlap_threshold"]
            ),
        )
        joint, joint_audit = build_joint_category(
            inventory,
            category,
            k=int(joint_config["equal_k"]),
            replicates=int(joint_config["matched_random_controls"]),
            seed=args.seed + 100000 + category_index * 10000,
            purity_threshold=float(joint_config["pure_control_overlap_threshold"]),
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
                "seeds_identical": (
                    plan["seeds"] == source["seeds"]
                    if args.scope == "full"
                    else plan["seeds"]
                    == source["seeds"][: len(plan["seeds"])]
                ),
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
                    == (
                        source["categories"][category]["dataset_indices"]
                        if args.scope == "full"
                        else source["categories"][category]["dataset_indices"][
                            : len(plan["categories"][category]["dataset_indices"])
                        ]
                    )
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
                            len(controls) == plan["matched_random_controls"]
                            and len({tuple(item["indices"]) for item in controls})
                            == plan["matched_random_controls"]
                            and all(
                                len(item["indices"]) == design["target_count"]
                                and item["max_target_group_overlap"]
                                < plan["pure_control_overlap_threshold"]
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
