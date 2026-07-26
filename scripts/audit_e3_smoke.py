#!/usr/bin/env python3
"""Audit E3 smoke outputs against their frozen plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    seed = int(plan["seeds"][0])
    categories = {}
    passed = True
    for category, category_plan in plan["categories"].items():
        root = args.run_root / f"seed_{seed}" / category
        summary_path = root / "run_summary.json"
        if not summary_path.is_file():
            categories[category] = {"passed": False, "reason": "missing run_summary.json"}
            passed = False
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_conditions = category_plan["conditions"]
        same_conditions = summary["conditions"] == expected_conditions
        one_expected_image = summary["dataset_indices"] == category_plan["dataset_indices"][:1]
        determinism = (
            len(summary.get("determinism_checks", [])) == 1
            and summary["determinism_checks"][0]["passed"]
        )
        max_non_target_delta = 0.0
        intervention_count = 0
        for condition in expected_conditions:
            condition_summary = json.loads(
                (root / condition["name"] / "summary.json").read_text(encoding="utf-8")
            )
            for audit in condition_summary["intervention_audits"]:
                intervention_count += 1
                max_non_target_delta = max(
                    max_non_target_delta,
                    float(audit["max_abs_delta_non_target"]),
                )
        category_passed = (
            same_conditions
            and one_expected_image
            and determinism
            and max_non_target_delta == 0.0
        )
        categories[category] = {
            "passed": category_passed,
            "condition_count": len(expected_conditions),
            "intervention_count": intervention_count,
            "conditions_equal_plan": same_conditions,
            "dataset_index_equal_plan": one_expected_image,
            "determinism_passed": determinism,
            "max_abs_delta_non_target": max_non_target_delta,
            "repository_commit": summary["repository_commit"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "mean_token_cache_sha256": summary["mean_token_cache_sha256"],
        }
        passed = passed and category_passed
    audit = {
        "experiment": plan["name"],
        "scope": "engineering_smoke_only",
        "passed": passed,
        "categories": categories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if not passed:
        raise RuntimeError("E3 smoke audit failed")


if __name__ == "__main__":
    main()
