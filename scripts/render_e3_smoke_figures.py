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
    visual_grid,
    write_csv,
)
from neuro_roi_causal.e3 import descriptive_distribution_rows


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("smoke", "pilot"),
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
    if len(seed_roots) != 1:
        raise ValueError("Expected exactly one engineering-smoke seed")
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
        value = "matched_minus_nonmatched"
    else:
        result_rows = [
            row
            for row in rows(args.evaluation_dir / "joint_mask_results.csv")
            if row["metric"] == "dino"
        ]
        value = "target_minus_pure_random"
    effect_plot(
        result_rows,
        args.evaluation_dir / "figures" / "effect_forest_plot.png",
        value,
    )
    per_image = rows(args.evaluation_dir / "per_image_excess_effects.csv")
    write_csv(
        args.evaluation_dir / "effect_distribution_summary.csv",
        descriptive_distribution_rows(per_image),
    )
    distribution_plot(
        per_image,
        args.evaluation_dir / "figures" / "effect_distribution_plot.png",
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
            and summary.get("scope") == args.mode
            and summary.get("formal_inference_performed") is False
            and inference_fields_empty
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
