"""Planning helpers for the E2 functional-ROI causal experiment."""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np


CONFIRMATORY_CATEGORIES = ("Face", "Body", "Scene")
UNRELATED_CATEGORY = {"Face": "Body", "Body": "Scene", "Scene": "Body"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_manifest_indices(
    rows: Iterable[dict[str, str]],
    category_counts: dict[str, int],
    *,
    index_column: str = "dataset_idx",
) -> dict[str, list[int]]:
    rows = list(rows)
    selected: dict[str, list[int]] = {}
    seen: set[int] = set()
    for category, count in category_counts.items():
        if category not in CONFIRMATORY_CATEGORIES:
            raise ValueError(f"Unsupported confirmatory category: {category}")
        if count <= 0:
            raise ValueError(f"{category} sample count must be positive")
        score_column = f"{category.lower()}_selection_score"
        candidates = [
            row
            for row in rows
            if row.get("confirmatory_category") == category
            and row.get("analysis_role") == "confirmatory"
        ]
        candidates.sort(
            key=lambda row: (-float(row[score_column]), int(row[index_column]))
        )
        if len(candidates) < count:
            raise ValueError(
                f"{category} requires {count} samples but only {len(candidates)} are available"
            )
        indices = [int(row[index_column]) for row in candidates[:count]]
        overlap = seen.intersection(indices)
        if overlap:
            raise ValueError(f"Dataset indices occur in multiple categories: {sorted(overlap)}")
        seen.update(indices)
        selected[category] = indices
    return selected


def roi_rows(inventory: Iterable[dict[str, str]], category: str) -> list[dict[str, str]]:
    rows = [
        row
        for row in inventory
        if row["dominant_roi"] == category
        and row["in_top_snr_200"].strip().lower() == "true"
    ]
    if not rows:
        raise ValueError(f"No top-SNR parcels found for {category}")
    return sorted(rows, key=lambda row: int(row["top200_token_index"]))


def _scales(rows: Iterable[dict[str, str]]) -> tuple[float, float]:
    rows = list(rows)
    snr = float(np.std([float(row["mean_ncsnr"]) for row in rows]))
    size = float(np.std([math.log(float(row["num_vertices"])) for row in rows]))
    return max(snr, 1e-12), max(size, 1e-12)


def parcel_distance(
    target: dict[str, str],
    candidate: dict[str, str],
    snr_scale: float,
    size_scale: float,
) -> float:
    snr_delta = (float(target["mean_ncsnr"]) - float(candidate["mean_ncsnr"])) / snr_scale
    size_delta = (
        math.log(float(target["num_vertices"]))
        - math.log(float(candidate["num_vertices"]))
    ) / size_scale
    return float(snr_delta**2 + size_delta**2)


def match_parcels(
    targets: Iterable[dict[str, str]],
    candidates: Iterable[dict[str, str]],
    *,
    rng: np.random.Generator | None,
    window: int = 8,
) -> dict:
    targets = list(targets)
    candidates = list(candidates)
    snr_scale, size_scale = _scales([*targets, *candidates])
    available = {
        int(row["top200_token_index"]): row
        for row in candidates
    }
    pairs = []
    for target in sorted(
        targets,
        key=lambda row: (row["hemisphere"], -float(row["mean_ncsnr"])),
    ):
        same_hemi = [
            row
            for row in available.values()
            if row["hemisphere"] == target["hemisphere"]
        ]
        if not same_hemi:
            raise ValueError(
                f"No remaining {target['hemisphere']} candidate for token "
                f"{target['top200_token_index']}"
            )
        ranked = sorted(
            same_hemi,
            key=lambda row: (
                parcel_distance(target, row, snr_scale, size_scale),
                int(row["top200_token_index"]),
            ),
        )
        if rng is None:
            selected = ranked[0]
        else:
            pool = ranked[: min(window, len(ranked))]
            selected = pool[int(rng.integers(len(pool)))]
        selected_index = int(selected["top200_token_index"])
        item_distance = parcel_distance(target, selected, snr_scale, size_scale)
        pairs.append(
            {
                "target_index": int(target["top200_token_index"]),
                "control_index": selected_index,
                "hemisphere": target["hemisphere"],
                "target_mean_ncsnr": float(target["mean_ncsnr"]),
                "control_mean_ncsnr": float(selected["mean_ncsnr"]),
                "target_num_vertices": int(target["num_vertices"]),
                "control_num_vertices": int(selected["num_vertices"]),
                "distance": item_distance,
            }
        )
        del available[selected_index]
    distances = [pair["distance"] for pair in pairs]
    return {
        "indices": sorted(pair["control_index"] for pair in pairs),
        "pairs": pairs,
        "mean_distance": float(np.mean(distances)),
        "max_distance": float(np.max(distances)),
    }


def build_matched_controls(
    inventory: Iterable[dict[str, str]],
    targets: Iterable[dict[str, str]],
    *,
    replicates: int,
    seed: int,
) -> list[dict]:
    inventory = list(inventory)
    targets = list(targets)
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    target_indices = {int(row["top200_token_index"]) for row in targets}
    neutral = [
        row
        for row in inventory
        if row["in_top_snr_200"].strip().lower() == "true"
        and row["dominant_roi"] == "Unlabeled"
        and int(row["top200_token_index"]) not in target_indices
    ]
    rng = np.random.default_rng(seed)
    controls = []
    observed: set[tuple[int, ...]] = set()
    attempts = 0
    while len(controls) < replicates:
        attempts += 1
        if attempts > replicates * 100:
            raise RuntimeError("Could not generate enough unique matched controls")
        control = match_parcels(targets, neutral, rng=rng)
        key = tuple(control["indices"])
        if key in observed:
            continue
        observed.add(key)
        control["replicate"] = len(controls) + 1
        controls.append(control)
    return controls


def build_category_conditions(
    inventory: Iterable[dict[str, str]],
    category: str,
    *,
    mask_modes: Iterable[str],
    random_replicates: int,
    equal_k: int,
    seed: int,
) -> tuple[list[dict], dict]:
    inventory = list(inventory)
    modes = list(mask_modes)
    if not modes or any(mode not in {"zero", "mean"} for mode in modes):
        raise ValueError("mask_modes must contain only zero and/or mean")
    full_targets = roi_rows(inventory, category)
    equal_targets = sorted(
        full_targets,
        key=lambda row: (-float(row["mean_ncsnr"]), int(row["top200_token_index"])),
    )[: min(equal_k, len(full_targets))]
    designs = [("full", full_targets)]
    if len(equal_targets) != len(full_targets):
        designs.append((f"equal{len(equal_targets)}", equal_targets))

    conditions = [
        {
            "name": "no_mask",
            "masked_token_indices": [],
            "mask_mode": "zero",
            "design": "baseline",
            "control_type": "none",
        }
    ]
    design_audits = {}
    for design_index, (design_name, targets) in enumerate(designs):
        target_indices = sorted(int(row["top200_token_index"]) for row in targets)
        for mode in modes:
            conditions.append(
                {
                    "name": f"{category}_{design_name}_{mode}",
                    "masked_token_indices": target_indices,
                    "mask_mode": mode,
                    "design": design_name,
                    "control_type": "target_roi",
                }
            )
        controls = build_matched_controls(
            inventory,
            targets,
            replicates=random_replicates,
            seed=seed + design_index * 1000,
        )
        for control in controls:
            for mode in modes:
                conditions.append(
                    {
                        "name": (
                            f"{category}_random_{design_name}_"
                            f"{control['replicate']:02d}_{mode}"
                        ),
                        "masked_token_indices": control["indices"],
                        "mask_mode": mode,
                        "design": design_name,
                        "control_type": "matched_random",
                        "control_replicate": control["replicate"],
                    }
                )
        design_audits[design_name] = {
            "target_indices": target_indices,
            "target_count": len(target_indices),
            "random_controls": controls,
        }

    unrelated_category = UNRELATED_CATEGORY[category]
    unrelated_pool = roi_rows(inventory, unrelated_category)
    unrelated_targets = equal_targets
    unrelated_match = match_parcels(unrelated_targets, unrelated_pool, rng=None)
    for mode in modes:
        conditions.append(
            {
                "name": f"{unrelated_category}_unrelated_equal{len(equal_targets)}_{mode}",
                "masked_token_indices": unrelated_match["indices"],
                "mask_mode": mode,
                "design": f"equal{len(equal_targets)}",
                "control_type": "unrelated_roi",
                "source_roi": unrelated_category,
            }
        )
    conditions.append(
        {
            "name": "no_mask_repeat",
            "masked_token_indices": [],
            "mask_mode": "zero",
            "design": "determinism_check",
            "control_type": "none",
        }
    )
    audit = {
        "category": category,
        "full_target_count": len(full_targets),
        "equal_k": len(equal_targets),
        "designs": design_audits,
        "unrelated_category": unrelated_category,
        "unrelated_control": unrelated_match,
    }
    return conditions, audit
