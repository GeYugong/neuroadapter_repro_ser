#!/usr/bin/env python3
"""Create audited hemisphere/SNR/size-matched E2 random parcel controls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "src"))

from neuro_roi_causal.e2 import build_matched_controls, read_csv, roi_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=["Face", "Body", "Scene"])
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--equal-k", type=int)
    parser.add_argument("--mask-modes", nargs="+", default=["zero", "mean"])
    args = parser.parse_args()

    if any(mode not in {"zero", "mean"} for mode in args.mask_modes):
        raise ValueError("--mask-modes accepts only zero and mean")
    inventory = read_csv(args.inventory)
    conditions = []
    controls = {}
    for category_index, category in enumerate(args.categories):
        targets = roi_rows(inventory, category)
        if args.equal_k is not None:
            if args.equal_k <= 0:
                raise ValueError("--equal-k must be positive")
            targets = sorted(
                targets,
                key=lambda row: (
                    -float(row["mean_ncsnr"]),
                    int(row["top200_token_index"]),
                ),
            )[: args.equal_k]
        matched = build_matched_controls(
            inventory,
            targets,
            replicates=args.replicates,
            seed=args.seed + category_index * 1000,
        )
        for control in matched:
            for mode in args.mask_modes:
                conditions.append(
                    {
                        "name": (
                            f"{category}_random_{control['replicate']:02d}_{mode}"
                        ),
                        "masked_token_indices": control["indices"],
                        "mask_mode": mode,
                        "control_type": "matched_random",
                    }
                )
        controls[category] = {
            "target_indices": sorted(
                int(row["top200_token_index"]) for row in targets
            ),
            "target_count": len(targets),
            "replicates": matched,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "description": (
                    "Deterministic random controls matched by parcel count, "
                    "hemisphere, mean ncsnr, and parcel size. Candidates are "
                    "restricted to top-SNR-200 parcels labeled Unlabeled."
                ),
                "source_inventory": str(args.inventory),
                "seed": args.seed,
                "replicates": args.replicates,
                "mask_modes": args.mask_modes,
                "conditions": conditions,
                "controls": controls,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
