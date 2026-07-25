#!/usr/bin/env python3
"""Audit target ROI purity and frozen matched-control contamination."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "src"))

from neuro_roi_causal.e2 import build_matched_controls
from neuro_roi_causal.overlap_audit import (
    FUNCTIONAL_GROUPS,
    count_candidates_by_hemisphere,
    parse_all_roi_overlaps,
    summarize_parcel,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--zero-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inventory = read_csv(args.inventory)
    by_index = {int(row["top200_token_index"]): row for row in inventory}
    plan = json.loads(args.zero_plan.read_text(encoding="utf-8"))
    output = args.output_dir
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    target_rows = []
    control_rows = []
    feasibility = []
    for category, category_plan in plan["categories"].items():
        full = category_plan["matching_audit"]["designs"]["full"]
        target_indices = [int(value) for value in full["target_indices"]]
        for index in target_indices:
            target_rows.append({"category": category, **summarize_parcel(by_index[index], category)})

        for condition in category_plan["conditions"]:
            if condition.get("control_type") != "matched_random":
                continue
            for index in condition["masked_token_indices"]:
                summary = summarize_parcel(by_index[int(index)], category)
                row = {
                    "target_category": category,
                    "design": condition["design"],
                    "control_replicate": int(condition["control_replicate"]),
                    "control_token_index": int(index),
                    **summary,
                }
                for threshold in (0.10, 0.25, 0.40, 0.50):
                    row[f"target_overlap_ge_{threshold:.2f}"] = (
                        row["target_group_max_overlap"] >= threshold
                    )
                control_rows.append(row)

        targets = [by_index[index] for index in target_indices]
        excluded = set(target_indices)
        for threshold in (0.10, 0.25, 0.40, 0.50):
            candidates = []
            for row in inventory:
                index = int(row["top200_token_index"])
                if index in excluded:
                    continue
                overlap = summarize_parcel(row, category)["target_group_max_overlap"]
                if overlap < threshold:
                    candidates.append(row)
            error = None
            controls = []
            try:
                controls = build_matched_controls(
                    candidates,
                    targets,
                    replicates=5,
                    seed=20260726,
                    excluded_indices=excluded,
                )
            except (ValueError, RuntimeError) as exc:
                error = str(exc)
            distances = [item["mean_distance"] for item in controls]
            feasibility.append(
                {
                    "category": category,
                    "target_overlap_exclusion_threshold": threshold,
                    "available_candidates": len(candidates),
                    "available_lh": count_candidates_by_hemisphere(candidates).get("lh", 0),
                    "available_rh": count_candidates_by_hemisphere(candidates).get("rh", 0),
                    "target_count": len(targets),
                    "five_unique_controls_feasible": len(controls) == 5,
                    "mean_matching_distance": float(np.mean(distances)) if distances else "",
                    "max_matching_distance": (
                        max(item["max_distance"] for item in controls) if controls else ""
                    ),
                    "error": error or "",
                }
            )

    write_csv(output / "target_roi_purity.csv", target_rows)
    write_csv(output / "matched_control_target_overlap.csv", control_rows)
    write_csv(output / "control_feasibility_by_threshold.csv", feasibility)

    matrix = np.asarray(
        [
            [float(row[f"group_overlap__{group}"]) for group in FUNCTIONAL_GROUPS]
            for row in target_rows
        ]
    )
    fig, ax = plt.subplots(figsize=(9, max(4, len(target_rows) * 0.10)))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(FUNCTIONAL_GROUPS)), FUNCTIONAL_GROUPS)
    ax.set_ylabel("Target parcel")
    ax.set_title("Target parcel overlap with functional ROI groups")
    fig.colorbar(image, ax=ax, label="maximum overlap")
    fig.tight_layout()
    fig.savefig(figures / "target_roi_overlap_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for category in plan["categories"]:
        values = [
            row["target_group_max_overlap"]
            for row in control_rows
            if row["target_category"] == category and row["design"] == "full"
        ]
        ax.hist(values, bins=np.linspace(0, 1, 21), alpha=0.45, label=category)
    ax.set_xlabel("Matched-control overlap with target functional group")
    ax.set_ylabel("Parcel count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "control_target_overlap_distribution.png", dpi=180)
    plt.close(fig)

    summary = {
        "inventory": str(args.inventory),
        "zero_plan": str(args.zero_plan),
        "target_parcels": len(target_rows),
        "matched_control_parcel_records": len(control_rows),
        "target_dual_high_overlap_count": sum(
            bool(row["target_off_target_dual_high_overlap"]) for row in target_rows
        ),
        "full_control_contamination": {
            category: {
                f"ge_{threshold:.2f}": sum(
                    row["target_category"] == category
                    and row["design"] == "full"
                    and row["target_group_max_overlap"] >= threshold
                    for row in control_rows
                )
                for threshold in (0.10, 0.25, 0.40, 0.50)
            }
            for category in plan["categories"]
        },
    }
    (output / "overlap_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# E2 ROI overlap 与随机对照污染审计\n\n"
        "本目录解析 `all_roi_overlaps`，审计正式 zero plan 中目标 parcel 的"
        "纯度和冻结 matched-random controls 对目标功能组的重叠。"
        "可行性表只回答未来 pure-control sensitivity 是否可构造，"
        "不会替换当前正式对照。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
