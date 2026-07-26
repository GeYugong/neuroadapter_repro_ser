#!/usr/bin/env python3
"""Audit E3 smoke or pilot outputs against their frozen plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

from PIL import Image

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "src"))

from neuro_roi_causal.e3_audit import audit_shared_diffusion_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("smoke", "pilot", "formal"),
        default="smoke",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def valid_image(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.width > 0 and image.height > 0
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    image_counts = {
        category: (
            len(category_plan["dataset_indices"])
            if args.mode == "formal"
            else int(plan["execution"][f"{args.mode}_images_per_category"])
        )
        for category, category_plan in plan["categories"].items()
    }
    seed_count = (
        len(plan["seeds"])
        if args.mode == "formal"
        else int(plan["execution"][f"{args.mode}_seeds"])
    )
    if args.mode == "pilot":
        if plan.get("plan_scope") != "pilot":
            raise ValueError("Pilot audit requires an independent pilot plan")
        if (
            set(image_counts.values()) != {10}
            or seed_count != 1
            or len(plan["seeds"]) != 1
        ):
            raise ValueError("Pilot plan must freeze ten images and one seed")
    if args.mode == "formal":
        if plan.get("plan_scope") != "formal":
            raise ValueError("Formal audit requires a formal frozen plan")
        if image_counts != {"Face": 37, "Body": 50, "Scene": 50}:
            raise ValueError("Formal image counts must be 37/50/50")
        if [int(seed) for seed in plan["seeds"]] != [12345, 23456, 34567]:
            raise ValueError("Formal seeds differ from the preregistered set")
    seeds = [int(seed) for seed in plan["seeds"][:seed_count]]
    categories = {}
    passed = True
    for seed in seeds:
        for category, category_plan in plan["categories"].items():
            image_count = image_counts[category]
            key = category if len(seeds) == 1 else f"seed_{seed}_{category}"
            root = args.run_root / f"seed_{seed}" / category
            summary_path = root / "run_summary.json"
            if not summary_path.is_file():
                categories[key] = {
                    "passed": False,
                    "reason": "missing run_summary.json",
                }
                passed = False
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            expected_conditions = category_plan["conditions"]
            expected_indices = category_plan["dataset_indices"][:image_count]
            same_conditions = summary["conditions"] == expected_conditions
            same_indices = summary["dataset_indices"] == expected_indices
            determinism_checks = summary.get("determinism_checks", [])
            determinism = (
                len(determinism_checks) == image_count
                and all(item["passed"] for item in determinism_checks)
            )
            expected_names = [
                condition["name"] for condition in expected_conditions
            ]
            shared_state = audit_shared_diffusion_state(
                summary,
                expected_indices=expected_indices,
                expected_condition_names=expected_names,
            )
            max_non_target_delta = 0.0
            intervention_count = 0
            output_files_passed = True
            condition_summaries_equal_plan = True
            gt_hashes: dict[int, set[str]] = {
                int(dataset_idx): set() for dataset_idx in expected_indices
            }
            for condition in expected_conditions:
                condition_path = root / condition["name"] / "summary.json"
                if not condition_path.is_file():
                    output_files_passed = False
                    continue
                condition_summary = json.loads(
                    condition_path.read_text(encoding="utf-8")
                )
                condition_summaries_equal_plan = (
                    condition_summaries_equal_plan
                    and condition_summary["condition"] == condition
                    and condition_summary["dataset_indices"] == expected_indices
                    and int(condition_summary["num_samples"]) == image_count
                )
                records = condition_summary.get("records", [])
                if (
                    len(records) != image_count
                    or [int(item["dataset_idx"]) for item in records]
                    != expected_indices
                ):
                    output_files_passed = False
                for record in records:
                    dataset_idx = int(record["dataset_idx"])
                    for field in ("gt", "pred", "comparison"):
                        if not valid_image(Path(record[field])):
                            output_files_passed = False
                    gt_path = Path(record["gt"])
                    if gt_path.is_file() and dataset_idx in gt_hashes:
                        gt_hashes[dataset_idx].add(file_sha256(gt_path))
                for audit in condition_summary["intervention_audits"]:
                    intervention_count += 1
                    delta = float(audit["max_abs_delta_non_target"])
                    if not math.isfinite(delta):
                        output_files_passed = False
                    max_non_target_delta = max(max_non_target_delta, delta)
            gt_alignment = all(
                len(hashes) == 1 for hashes in gt_hashes.values()
            )
            category_passed = (
                same_conditions
                and same_indices
                and determinism
                and shared_state["passed"]
                and condition_summaries_equal_plan
                and output_files_passed
                and gt_alignment
                and max_non_target_delta == 0.0
            )
            categories[key] = {
                "passed": category_passed,
                "seed": seed,
                "image_count": image_count,
                "condition_count": len(expected_conditions),
                "condition_image_record_count": (
                    image_count * len(expected_conditions)
                ),
                "intervention_count": intervention_count,
                "conditions_equal_plan": same_conditions,
                "dataset_indices_equal_plan": same_indices,
                "condition_summaries_equal_plan": (
                    condition_summaries_equal_plan
                ),
                "output_files_passed": output_files_passed,
                "ground_truth_alignment_passed": gt_alignment,
                "determinism_passed": determinism,
                "shared_latent_noise_passed": shared_state["passed"],
                "shared_latent_noise_audit": shared_state,
                "max_abs_delta_non_target": max_non_target_delta,
                "repository_commit": summary["repository_commit"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "mean_token_cache_sha256": (
                    summary["mean_token_cache_sha256"]
                ),
            }
            passed = passed and category_passed
    checkpoint_hashes = {
        item.get("checkpoint_sha256")
        for item in categories.values()
        if item.get("checkpoint_sha256")
    }
    mean_cache_hashes = {
        item.get("mean_token_cache_sha256")
        for item in categories.values()
        if item.get("mean_token_cache_sha256")
    }
    repository_commits = {
        item.get("repository_commit")
        for item in categories.values()
        if item.get("repository_commit")
    }
    consistent_assets = (
        checkpoint_hashes == {plan["checkpoint_sha256"]}
        and mean_cache_hashes == {plan["mean_token_cache_sha256"]}
        and repository_commits == {plan["repository_commit"]}
    )
    passed = passed and consistent_assets
    audit = {
        "experiment": plan["name"],
        "scope": f"engineering_{args.mode}_only",
        "passed": passed,
        "expected_task_count": len(seeds) * len(plan["categories"]),
        "completed_task_count": sum(
            bool(item.get("passed")) for item in categories.values()
        ),
        "expected_condition_image_records": sum(
            image_counts[category] * len(category_plan["conditions"])
            for category, category_plan in plan["categories"].items()
        )
        * len(seeds),
        "all_categories_same_checkpoint": (
            checkpoint_hashes == {plan["checkpoint_sha256"]}
        ),
        "all_categories_same_mean_token_cache": (
            mean_cache_hashes == {plan["mean_token_cache_sha256"]}
        ),
        "all_categories_same_repository_commit": (
            repository_commits == {plan["repository_commit"]}
        ),
        "categories": categories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if not passed:
        raise RuntimeError(f"E3 {args.mode} audit failed")


if __name__ == "__main__":
    main()
