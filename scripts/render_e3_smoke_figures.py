#!/usr/bin/env python3
"""Re-render E3 smoke figures from completed outputs without metric models."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "scripts"))

from evaluate_e2 import load_records
from evaluate_e3 import effect_plot, visual_grid


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument(
        "--experiment",
        choices=("E3_interaction", "E3_joint_redundancy"),
        required=True,
    )
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
