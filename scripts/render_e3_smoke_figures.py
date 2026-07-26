#!/usr/bin/env python3
"""Re-render E3 figures and descriptive audits without metric models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "scripts"))
sys.path.insert(0, str(REPRO_ROOT / "src"))

from evaluate_e2 import load_records
from evaluate_e3 import (
    GLOBAL_METRICS,
    LOCAL_METRICS,
    distribution_plot,
    effect_plot,
    monotonicity_plot,
    visual_grid,
    write_csv,
)
from neuro_roi_causal.e3 import descriptive_distribution_rows


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_probability(value: str | None) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and 0.0 <= number <= 1.0


def valid_interval(value: str | None) -> bool:
    try:
        interval = json.loads(value or "")
        return (
            isinstance(interval, list)
            and len(interval) == 2
            and all(math.isfinite(float(item)) for item in interval)
            and float(interval[0]) <= float(interval[1])
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("smoke", "pilot", "formal"),
        default="smoke",
    )
    parser.add_argument(
        "--experiment",
        choices=("E3_interaction", "E3_joint_redundancy"),
        required=True,
    )
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    seed_roots = sorted(path for path in args.run_root.glob("seed_*") if path.is_dir())
    expected_seed_count = len(plan["seeds"]) if args.mode == "formal" else 1
    if len(seed_roots) != expected_seed_count:
        raise ValueError(
            f"Expected {expected_seed_count} seed directories, "
            f"found {len(seed_roots)}"
        )
    for category in ("Face", "Body", "Scene"):
        records, metadata = load_records(seed_roots[0] / category)
        visual_grid(
            seed_roots[0] / category,
            records,
            metadata,
            args.evaluation_dir / "figures" / f"{category.lower()}_comparison_grid.png",
        )
    if args.experiment == "E3_interaction":
        result_rows = rows(args.evaluation_dir / "interaction_results.csv")
        all_primary_rows = result_rows
        value = "matched_minus_nonmatched"
        local_result_rows = rows(
            args.evaluation_dir / "local_interaction_results.csv"
        )
        effect_plot(
            local_result_rows,
            args.evaluation_dir / "figures" / "local_effect_forest_plot.png",
            "target_minus_pure_random",
            formal=args.mode == "formal",
        )
    else:
        all_joint_rows = rows(args.evaluation_dir / "joint_mask_results.csv")
        all_primary_rows = all_joint_rows
        result_rows = [
            row
            for row in all_joint_rows
            if row["metric"] == "dino"
        ]
        value = "target_minus_pure_random"
        effect_plot(
            [row for row in all_joint_rows if row["metric_family"] == "global"],
            args.evaluation_dir / "figures" / "global_effect_forest_plot.png",
            value,
            formal=args.mode == "formal",
        )
        effect_plot(
            [row for row in all_joint_rows if row["metric_family"] == "local"],
            args.evaluation_dir / "figures" / "local_effect_forest_plot.png",
            value,
            formal=args.mode == "formal",
        )
        trend_rows = rows(args.evaluation_dir / "monotonicity_results.csv")
        monotonicity_plot(
            trend_rows,
            args.evaluation_dir / "figures" / "joint_size_trend_plot.png",
        )
    effect_plot(
        result_rows,
        args.evaluation_dir / "figures" / "effect_forest_plot.png",
        value,
        formal=args.mode == "formal",
    )
    per_image = rows(args.evaluation_dir / "per_image_excess_effects.csv")
    write_csv(
        args.evaluation_dir / "effect_distribution_summary.csv",
        descriptive_distribution_rows(per_image),
    )
    distribution_plot(
        per_image,
        args.evaluation_dir / "figures" / "effect_distribution_plot.png",
        scope=args.mode,
    )

    sample_rows = rows(args.evaluation_dir / "per_sample_metrics.csv")
    local_rows = rows(args.evaluation_dir / "local_metric_results.csv")
    summary = json.loads(
        (args.evaluation_dir / "evaluation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    expected_rows = sum(
        len(category["dataset_indices"]) * len(category["conditions"])
        for category in plan["categories"].values()
    ) * len(plan["seeds"])
    invalid = []
    for row in sample_rows:
        for metric in [
            *GLOBAL_METRICS,
            *LOCAL_METRICS[row["image_category"]],
        ]:
            try:
                valid = math.isfinite(float(row[metric]))
            except (KeyError, TypeError, ValueError):
                valid = False
            if not valid:
                invalid.append(
                    {
                        "dataset_idx": row.get("dataset_idx"),
                        "condition": row.get("condition"),
                        "metric": metric,
                    }
                )
    result_file = (
        "interaction_results.csv"
        if args.experiment == "E3_interaction"
        else "joint_mask_results.csv"
    )
    result_rows = rows(args.evaluation_dir / result_file)
    inference_fields_empty = all(
        not (
            row.get("permutation_p")
            or row.get("sign_flip_p")
            or row.get("bh_q_e3a")
            or row.get("bh_q_e3b")
        )
        for row in result_rows
    )
    formal_result_files = (
        [args.evaluation_dir / "local_interaction_results.csv"]
        if args.experiment == "E3_interaction"
        else [args.evaluation_dir / "monotonicity_results.csv"]
    )
    formal_rows = list(all_primary_rows)
    for result_path in formal_result_files:
        formal_rows.extend(rows(result_path))
    if args.mode == "formal":
        expected_result_counts = (
            {"primary": 15, "secondary": 27}
            if args.experiment == "E3_interaction"
            else {"primary": 120, "secondary": 24}
        )
        result_counts_passed = (
            len(all_primary_rows) == expected_result_counts["primary"]
            and len(formal_rows) - len(all_primary_rows)
            == expected_result_counts["secondary"]
        )
        formal_inference_complete = result_counts_passed and all(
            valid_interval(row.get("ci95"))
            and finite_probability(
                row.get("sign_flip_p") or row.get("permutation_p")
            )
            and finite_probability(
                row.get("bh_q_e3a")
                or row.get("bh_q_e3a_local")
                or row.get("bh_q_e3b")
                or row.get("bh_q_e3b_monotonicity")
            )
            for row in formal_rows
        )
    else:
        result_counts_passed = True
        formal_inference_complete = False
    per_image_rows = rows(
        args.evaluation_dir / "per_image_excess_effects.csv"
    )
    per_image_seed_count_passed = all(
        int(row["num_seeds"]) == (
            len(plan["seeds"]) if args.mode == "formal" else 1
        )
        for row in per_image_rows
    )
    local_regions_nonempty = all(
        int(float(row["local_region_pixels"])) > 0 for row in local_rows
    )
    detector_passed = (
        summary.get("face_detector_backend") == "opencv_haar_e1"
        and summary.get("face_detector_formal_compatibility") is True
        and summary.get("face_detector_cascade_sha256")
        == plan["face_detector"]["cascade_sha256"]
    )
    face_rows = [
        row for row in sample_rows if row["image_category"] == "Face"
    ]
    audit = {
        "experiment": args.experiment,
        "scope": f"engineering_{args.mode}_evaluation",
        "passed": (
            len(sample_rows) == expected_rows
            and len(local_rows) == expected_rows
            and not invalid
            and local_regions_nonempty
            and detector_passed
            and summary.get("repository_commit") == plan["repository_commit"]
            and summary.get("scope") == args.mode
            and result_counts_passed
            and per_image_seed_count_passed
            and (
                (
                    summary.get("formal_inference_performed") is True
                    and formal_inference_complete
                )
                if args.mode == "formal"
                else (
                    summary.get("formal_inference_performed") is False
                    and inference_fields_empty
                )
            )
        ),
        "expected_per_sample_rows": expected_rows,
        "actual_per_sample_rows": len(sample_rows),
        "actual_local_metric_rows": len(local_rows),
        "invalid_or_missing_metrics": invalid,
        "local_regions_nonempty": local_regions_nonempty,
        "minimum_local_region_pixels": summary.get(
            "minimum_local_region_pixels"
        ),
        "face_detector_passed": detector_passed,
        "face_prediction_detection_success": sum(
            float(row["face_detection_success"]) for row in face_rows
        ),
        "face_prediction_count": len(face_rows),
        "formal_inference_performed": summary.get(
            "formal_inference_performed"
        ),
        "inference_fields_empty": inference_fields_empty,
        "formal_inference_complete": formal_inference_complete,
        "result_counts_passed": result_counts_passed,
        "per_image_seed_count_passed": per_image_seed_count_passed,
        "engineering_anomalies": [],
    }
    (args.evaluation_dir / "evaluation_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if not audit["passed"]:
        raise RuntimeError("E3 evaluation audit failed")


if __name__ == "__main__":
    main()
