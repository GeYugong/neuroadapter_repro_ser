#!/usr/bin/env python3
"""Create hemisphere/SNR/size-matched random parcel controls for ROI ablations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


TARGETS = ("V1", "V2", "V3", "V4", "Face", "Body", "Scene", "Word")


def distance(target: dict, candidate: dict, snr_scale: float, size_scale: float) -> float:
    return ((float(target["mean_ncsnr"]) - float(candidate["mean_ncsnr"])) / snr_scale) ** 2 + ((math.log(float(target["num_vertices"])) - math.log(float(candidate["num_vertices"]))) / size_scale) ** 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.inventory.open(encoding="utf-8")))
    rng = np.random.default_rng(args.seed)
    snr_scale = float(np.std([float(row["mean_ncsnr"]) for row in rows]))
    size_scale = float(np.std([math.log(float(row["num_vertices"])) for row in rows]))
    all_indices = {int(row["global_token_index"]) for row in rows}
    conditions, controls = [], {}
    for group in TARGETS:
        target_rows = [row for row in rows if row["dominant_group"] == group]
        if not target_rows:
            raise ValueError(f"No selected parcels for {group}")
        group_controls = []
        target_indices = {int(row["global_token_index"]) for row in target_rows}
        for replicate in range(1, args.replicates + 1):
            chosen = []
            for hemisphere in ("lh", "rh"):
                targets = [row for row in target_rows if row["hemisphere"] == hemisphere]
                pool = [row for row in rows if row["hemisphere"] == hemisphere and int(row["global_token_index"]) not in target_indices]
                available = {int(row["global_token_index"]): row for row in pool}
                # Randomized greedy nearest-neighbour matching preserves size and hemisphere.
                for target in sorted(targets, key=lambda item: float(item["mean_ncsnr"]), reverse=True):
                    ranked = sorted(available.values(), key=lambda item: distance(target, item, snr_scale, size_scale))
                    window = ranked[:min(8, len(ranked))]
                    selected = window[int(rng.integers(len(window)))]
                    chosen.append(int(selected["global_token_index"]))
                    del available[int(selected["global_token_index"])]
            if len(chosen) != len(target_rows) or len(set(chosen)) != len(chosen):
                raise RuntimeError(f"Invalid matched control for {group} replicate {replicate}")
            name = f"{group}_random_{replicate:02d}"
            conditions.append({"name": name, "masked_token_indices": sorted(chosen), "mask_mode": "zero"})
            group_controls.append({"name": name, "indices": sorted(chosen), "target_indices": sorted(target_indices)})
        controls[group] = group_controls
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "description": "20 deterministic random controls per ROI; matched by hemisphere, mean ncsnr, and parcel size.",
        "source_inventory": str(args.inventory), "seed": args.seed, "replicates": args.replicates,
        "conditions": conditions, "controls": controls, "all_selected_indices": sorted(all_indices),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
