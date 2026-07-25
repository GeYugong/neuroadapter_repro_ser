#!/usr/bin/env python3
"""Compare a unified-evaluator recheck with the preserved formal zero summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = ("pixel_corr", "ssim", "lpips", "clip", "dino")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-summary", type=Path, required=True)
    parser.add_argument("--new-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old = json.loads(args.old_summary.read_text(encoding="utf-8"))
    new = json.loads(args.new_summary.read_text(encoding="utf-8"))
    report = {"passed": True, "categories": {}, "failures": []}
    for category in ("Face", "Body", "Scene"):
        old_category = old["categories"][category]
        new_category = new["categories"][category]
        category_report = {"condition_mean_max_abs_delta": {}, "primary": {}}
        if (
            old_category["num_samples"] != new_category["num_samples"]
            or old_category["num_seeds"] != new_category["num_seeds"]
            or set(old_category["conditions"]) != set(new_category["conditions"])
        ):
            report["failures"].append(f"{category}: sample, seed, or condition count changed")
        for metric in METRICS:
            deltas = [
                abs(
                    float(new_category["conditions"][condition][metric])
                    - float(old_category["conditions"][condition][metric])
                )
                for condition in old_category["conditions"]
            ]
            category_report["condition_mean_max_abs_delta"][metric] = max(deltas)
            old_item = old_category["target_vs_matched_random"]["full"]["metrics"][metric]
            new_item = new_category["target_vs_matched_random"]["full"]["metrics"][metric]
            comparison = {
                "old_excess": old_item["target_minus_random_mean"],
                "new_excess": new_item["target_minus_random_mean"],
                "excess_abs_delta": abs(
                    new_item["target_minus_random_mean"]
                    - old_item["target_minus_random_mean"]
                ),
                "old_ci95": old_item["ci95"],
                "new_ci95": new_item["ci95"],
                "old_p": old_item["sign_flip_p"],
                "new_p": new_item["sign_flip_p"],
                "old_q": old_item["bh_q_primary_15_tests"],
                "new_q": new_item["bh_q_primary_15_tests"],
            }
            comparison["q_crossed_0.05"] = (
                comparison["old_q"] < 0.05
            ) != (comparison["new_q"] < 0.05)
            category_report["primary"][metric] = comparison
            if comparison["excess_abs_delta"] > 1e-4:
                report["failures"].append(
                    f"{category}/{metric}: primary excess changed by "
                    f"{comparison['excess_abs_delta']}"
                )
            if comparison["q_crossed_0.05"]:
                report["failures"].append(f"{category}/{metric}: q crossed 0.05")
        report["categories"][category] = category_report
    report["passed"] = not report["failures"]
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(json.dumps(report["failures"], indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
