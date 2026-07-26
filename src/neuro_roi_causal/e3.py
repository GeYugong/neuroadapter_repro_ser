"""Frozen planning and statistics helpers for E3 ROI interaction experiments."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .e2 import build_matched_controls, roi_rows
from .overlap_audit import group_max_overlaps, parse_all_roi_overlaps
from .stats import benjamini_hochberg, bootstrap_ci, sign_flip_pvalue


ROI_GROUPS = ("Face", "Body", "Scene")
GLOBAL_METRICS = ("pixel_corr", "ssim", "lpips", "clip", "dino")


def equal_k_targets(
    inventory: Iterable[dict[str, str]], category: str, *, k: int = 4
) -> list[dict[str, str]]:
    rows = sorted(
        roi_rows(inventory, category),
        key=lambda row: (-float(row["mean_ncsnr"]), int(row["top200_token_index"])),
    )
    if len(rows) < k:
        raise ValueError(f"{category} has only {len(rows)} parcels, fewer than k={k}")
    return rows[:k]


def max_overlap(row: dict[str, str], groups: Iterable[str]) -> float:
    overlaps = group_max_overlaps(parse_all_roi_overlaps(row["all_roi_overlaps"]))
    return max(overlaps[group] for group in groups)


def build_pure_controls(
    inventory: Iterable[dict[str, str]],
    targets: Iterable[dict[str, str]],
    *,
    target_groups: Iterable[str],
    replicates: int,
    seed: int,
    threshold: float = 0.10,
) -> list[dict[str, Any]]:
    inventory = list(inventory)
    targets = list(targets)
    groups = tuple(target_groups)
    if not groups or any(group not in ROI_GROUPS for group in groups):
        raise ValueError(f"Unsupported target groups: {groups}")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("Purity threshold must be in (0, 1]")
    target_indices = {int(row["top200_token_index"]) for row in targets}
    pure_inventory = [
        row
        for row in inventory
        if int(row["top200_token_index"]) in target_indices
        or max_overlap(row, groups) < threshold
    ]
    controls = build_matched_controls(
        pure_inventory,
        targets,
        replicates=replicates,
        seed=seed,
        excluded_indices=target_indices,
    )
    by_index = {
        int(row["top200_token_index"]): row
        for row in inventory
        if row["in_top_snr_200"].strip().lower() == "true"
    }
    for control in controls:
        control["purity_threshold"] = threshold
        control["target_groups"] = list(groups)
        control["max_target_group_overlap"] = max(
            max_overlap(by_index[index], groups) for index in control["indices"]
        )
        if control["max_target_group_overlap"] >= threshold:
            raise RuntimeError("Pure-control threshold audit failed")
    return controls


def _condition(
    name: str,
    indices: Iterable[int],
    *,
    image_category: str,
    masked_roi: str,
    control_type: str,
    design: str,
    replicate: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "masked_token_indices": sorted(map(int, indices)),
        "mask_mode": "mean" if indices else "zero",
        "image_category": image_category,
        "masked_roi": masked_roi,
        "target_groups": masked_roi.split("+") if masked_roi != "none" else [],
        "control_type": control_type,
        "design": design,
    }
    if replicate is not None:
        result["control_replicate"] = replicate
    return result


def _baseline_conditions(image_category: str) -> list[dict[str, Any]]:
    return [
        _condition(
            "no_mask",
            [],
            image_category=image_category,
            masked_roi="none",
            control_type="none",
            design="baseline",
        )
    ]


def _repeat_condition(image_category: str) -> dict[str, Any]:
    return _condition(
        "no_mask_repeat",
        [],
        image_category=image_category,
        masked_roi="none",
        control_type="none",
        design="determinism_check",
    )


def build_interaction_category(
    inventory: Iterable[dict[str, str]],
    image_category: str,
    *,
    k: int,
    replicates: int,
    seed: int,
    purity_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory = list(inventory)
    conditions = _baseline_conditions(image_category)
    audit: dict[str, Any] = {"image_category": image_category, "roi_designs": {}}
    for roi_index, masked_roi in enumerate(ROI_GROUPS):
        targets = equal_k_targets(inventory, masked_roi, k=k)
        target_indices = [int(row["top200_token_index"]) for row in targets]
        controls = build_pure_controls(
            inventory,
            targets,
            target_groups=[masked_roi],
            replicates=replicates,
            seed=seed + roi_index * 1000,
            threshold=purity_threshold,
        )
        conditions.append(
            _condition(
                f"mask_{masked_roi}_mean",
                target_indices,
                image_category=image_category,
                masked_roi=masked_roi,
                control_type="target_roi",
                design="interaction_equal4",
            )
        )
        for control in controls:
            conditions.append(
                _condition(
                    f"mask_{masked_roi}_pure_random_{control['replicate']:02d}_mean",
                    control["indices"],
                    image_category=image_category,
                    masked_roi=masked_roi,
                    control_type="pure_matched_random",
                    design="interaction_equal4",
                    replicate=control["replicate"],
                )
            )
        audit["roi_designs"][masked_roi] = {
            "target_indices": target_indices,
            "target_count": len(target_indices),
            "pure_random_controls": controls,
        }
    conditions.append(_repeat_condition(image_category))
    return conditions, audit


def build_joint_category(
    inventory: Iterable[dict[str, str]],
    image_category: str,
    *,
    k: int,
    replicates: int,
    seed: int,
    purity_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory = list(inventory)
    by_group = {
        group: equal_k_targets(inventory, group, k=k) for group in ROI_GROUPS
    }
    designs = [
        (image_category,),
        ("Face", "Body"),
        ("Face", "Scene"),
        ("Body", "Scene"),
        ROI_GROUPS,
    ]
    conditions = _baseline_conditions(image_category)
    audit: dict[str, Any] = {"image_category": image_category, "joint_designs": {}}
    for design_index, groups in enumerate(designs):
        targets = [row for group in groups for row in by_group[group]]
        target_indices = [int(row["top200_token_index"]) for row in targets]
        if len(target_indices) != len(set(target_indices)):
            raise ValueError(f"Joint ROI groups overlap in token indices: {groups}")
        label = "+".join(groups)
        controls = build_pure_controls(
            inventory,
            targets,
            target_groups=groups,
            replicates=replicates,
            seed=seed + design_index * 1000,
            threshold=purity_threshold,
        )
        conditions.append(
            _condition(
                f"mask_{label}_mean",
                target_indices,
                image_category=image_category,
                masked_roi=label,
                control_type="target_roi",
                design="joint_redundancy",
            )
        )
        for control in controls:
            conditions.append(
                _condition(
                    f"mask_{label}_pure_random_{control['replicate']:02d}_mean",
                    control["indices"],
                    image_category=image_category,
                    masked_roi=label,
                    control_type="pure_matched_random",
                    design="joint_redundancy",
                    replicate=control["replicate"],
                )
            )
        audit["joint_designs"][label] = {
            "target_indices": sorted(target_indices),
            "target_count": len(target_indices),
            "pure_random_controls": controls,
        }
    conditions.append(_repeat_condition(image_category))
    return conditions, audit


def interaction_rows(
    per_image_effects: Iterable[dict[str, Any]],
    *,
    metrics: Iterable[str],
    draws: int,
    formal: bool,
    analysis_status: str,
) -> list[dict[str, Any]]:
    rows = list(per_image_effects)
    output: list[dict[str, Any]] = []
    for roi_index, roi in enumerate(ROI_GROUPS):
        for metric_index, metric in enumerate(metrics):
            by_category = {
                category: np.asarray(
                    [
                        float(row["excess_causal_loss"])
                        for row in rows
                        if row["masked_roi"] == roi
                        and row["image_category"] == category
                        and row["metric"] == metric
                    ],
                    dtype=np.float64,
                )
                for category in ROI_GROUPS
            }
            matched = by_category[roi]
            nonmatched_categories = [
                by_category[category] for category in ROI_GROUPS if category != roi
            ]
            effect = (
                float(
                    matched.mean()
                    - np.mean([values.mean() for values in nonmatched_categories])
                )
                if len(matched) and all(len(values) for values in nonmatched_categories)
                else float("nan")
            )
            item: dict[str, Any] = {
                "masked_roi": roi,
                "metric": metric,
                "num_matched_images": int(len(matched)),
                "num_nonmatched_images": int(
                    sum(len(values) for values in nonmatched_categories)
                ),
                "num_nonmatched_categories": len(nonmatched_categories),
                "nonmatched_category_weighting": "equal_category_mean",
                "matched_minus_nonmatched": effect,
                "analysis_status": analysis_status,
                "ci95": None,
                "permutation_p": None,
                "bh_q_e3a": None,
            }
            if formal and len(matched) >= 2 and all(
                len(values) >= 2 for values in nonmatched_categories
            ):
                rng = np.random.default_rng(31000 + roi_index * 100 + metric_index)
                bootstrap = np.asarray(
                    [
                        rng.choice(matched, len(matched), replace=True).mean()
                        - np.mean(
                            [
                                rng.choice(values, len(values), replace=True).mean()
                                for values in nonmatched_categories
                            ]
                        )
                        for _ in range(draws)
                    ]
                )
                item["ci95"] = [
                    float(value) for value in np.quantile(bootstrap, [0.025, 0.975])
                ]
                groups = [matched, *nonmatched_categories]
                combined = np.concatenate(groups)
                null = []
                for _ in range(draws):
                    permuted = rng.permutation(combined)
                    offset = 0
                    permuted_groups = []
                    for values in groups:
                        permuted_groups.append(
                            permuted[offset : offset + len(values)]
                        )
                        offset += len(values)
                    null.append(
                        permuted_groups[0].mean()
                        - np.mean(
                            [values.mean() for values in permuted_groups[1:]]
                        )
                    )
                item["permutation_p"] = float(
                    (np.count_nonzero(np.abs(null) >= abs(effect)) + 1) / (draws + 1)
                )
            output.append(item)
    valid = [item for item in output if item["permutation_p"] is not None]
    if valid:
        for item, qvalue in zip(
            valid, benjamini_hochberg([item["permutation_p"] for item in valid])
        ):
            item["bh_q_e3a"] = qvalue
    return output


def cell_result_rows(
    per_image_effects: Iterable[dict[str, Any]],
    *,
    metrics_by_category: dict[str, Iterable[str]],
    draws: int,
    formal: bool,
    analysis_status: str,
    q_column: str,
) -> list[dict[str, Any]]:
    """Test within-category target-minus-control effects as one BH family."""
    rows = list(per_image_effects)
    output: list[dict[str, Any]] = []
    test_index = 0
    for category in ROI_GROUPS:
        for masked_roi in ROI_GROUPS:
            for metric in metrics_by_category[category]:
                values = np.asarray(
                    [
                        float(row["excess_causal_loss"])
                        for row in rows
                        if row["image_category"] == category
                        and row["masked_roi"] == masked_roi
                        and row["metric"] == metric
                    ],
                    dtype=np.float64,
                )
                item: dict[str, Any] = {
                    "image_category": category,
                    "masked_roi": masked_roi,
                    "metric": metric,
                    "metric_family": "local",
                    "num_images": int(len(values)),
                    "target_minus_pure_random": (
                        float(values.mean()) if len(values) else float("nan")
                    ),
                    "analysis_status": analysis_status,
                    "ci95": None,
                    "sign_flip_p": None,
                    q_column: None,
                }
                if formal and len(values) >= 2:
                    item["ci95"] = bootstrap_ci(
                        values,
                        51000 + test_index,
                        draws,
                    )
                    item["sign_flip_p"] = sign_flip_pvalue(
                        values,
                        52000 + test_index,
                        draws,
                    )
                output.append(item)
                test_index += 1
    valid = [item for item in output if item["sign_flip_p"] is not None]
    if valid:
        for item, qvalue in zip(
            valid,
            benjamini_hochberg([item["sign_flip_p"] for item in valid]),
        ):
            item[q_column] = qvalue
    return output


def joint_result_rows(
    per_image_effects: Iterable[dict[str, Any]],
    *,
    draws: int,
    formal: bool,
    analysis_status: str,
) -> list[dict[str, Any]]:
    rows = list(per_image_effects)
    keys = sorted(
        {
            (row["image_category"], row["masked_roi"], row["metric"])
            for row in rows
        }
    )
    output: list[dict[str, Any]] = []
    for key_index, (category, masked_roi, metric) in enumerate(keys):
        values = np.asarray(
            [
                float(row["excess_causal_loss"])
                for row in rows
                if (
                    row["image_category"],
                    row["masked_roi"],
                    row["metric"],
                )
                == (category, masked_roi, metric)
            ],
            dtype=np.float64,
        )
        item: dict[str, Any] = {
            "image_category": category,
            "masked_roi": masked_roi,
            "metric": metric,
            "metric_family": (
                "global" if metric in GLOBAL_METRICS else "local"
            ),
            "num_images": int(len(values)),
            "target_minus_pure_random": (
                float(values.mean()) if len(values) else float("nan")
            ),
            "analysis_status": analysis_status,
            "ci95": None,
            "sign_flip_p": None,
            "bh_q_e3b": None,
        }
        if formal and len(values) >= 2:
            item["ci95"] = bootstrap_ci(
                values,
                41000 + key_index,
                draws,
            )
            item["sign_flip_p"] = sign_flip_pvalue(
                values,
                42000 + key_index,
                draws,
            )
        output.append(item)
    for family in ("global", "local"):
        valid = [
            item
            for item in output
            if item["metric_family"] == family
            and item["sign_flip_p"] is not None
        ]
        if valid:
            for item, qvalue in zip(
                valid,
                benjamini_hochberg([item["sign_flip_p"] for item in valid]),
            ):
                item["bh_q_e3b"] = qvalue
    return output


def monotonicity_rows(
    per_image_effects: Iterable[dict[str, Any]],
    *,
    metrics_by_category: dict[str, Iterable[str]],
    draws: int,
    formal: bool,
    analysis_status: str,
) -> list[dict[str, Any]]:
    """Test the preregistered 4 -> 8 -> 12 token trend per image."""
    rows = list(per_image_effects)
    output: list[dict[str, Any]] = []
    test_index = 0
    for category in ROI_GROUPS:
        double_masks = sorted(
            "+".join(pair)
            for pair in (
                ("Face", "Body"),
                ("Face", "Scene"),
                ("Body", "Scene"),
            )
            if category in pair
        )
        metrics = list(GLOBAL_METRICS) + list(metrics_by_category[category])
        for metric in metrics:
            by_image: dict[int, dict[str, float]] = {}
            for row in rows:
                if row["image_category"] != category or row["metric"] != metric:
                    continue
                by_image.setdefault(int(row["dataset_idx"]), {})[
                    row["masked_roi"]
                ] = float(row["excess_causal_loss"])
            if not by_image:
                continue
            level_rows = []
            for dataset_idx, values in sorted(by_image.items()):
                required = [category, *double_masks, "Face+Body+Scene"]
                if any(label not in values for label in required):
                    raise ValueError(
                        f"Missing monotonicity level for {category}/{metric}/"
                        f"{dataset_idx}: {required}"
                    )
                levels = np.asarray(
                    [
                        values[category],
                        np.mean([values[label] for label in double_masks]),
                        values["Face+Body+Scene"],
                    ],
                    dtype=np.float64,
                )
                level_rows.append(
                    {
                        "dataset_idx": dataset_idx,
                        "levels": levels,
                        "slope": float(np.polyfit([1.0, 2.0, 3.0], levels, 1)[0]),
                    }
                )
            slopes = np.asarray(
                [item["slope"] for item in level_rows],
                dtype=np.float64,
            )
            level_matrix = np.stack(
                [item["levels"] for item in level_rows],
                axis=0,
            )
            item: dict[str, Any] = {
                "image_category": category,
                "metric": metric,
                "metric_family": (
                    "global" if metric in GLOBAL_METRICS else "local"
                ),
                "num_images": int(len(slopes)),
                "single_mask": category,
                "double_masks_averaged": ";".join(double_masks),
                "triple_mask": "Face+Body+Scene",
                "mean_excess_k4": float(level_matrix[:, 0].mean()),
                "mean_excess_k8": float(level_matrix[:, 1].mean()),
                "mean_excess_k12": float(level_matrix[:, 2].mean()),
                "mean_slope_per_level": float(slopes.mean()),
                "monotonic_non_decreasing_fraction": float(
                    np.mean(
                        (level_matrix[:, 0] <= level_matrix[:, 1])
                        & (level_matrix[:, 1] <= level_matrix[:, 2])
                    )
                ),
                "analysis_status": analysis_status,
                "ci95": None,
                "sign_flip_p": None,
                "bh_q_e3b_monotonicity": None,
            }
            if formal and len(slopes) >= 2:
                item["ci95"] = bootstrap_ci(
                    slopes,
                    61000 + test_index,
                    draws,
                )
                item["sign_flip_p"] = sign_flip_pvalue(
                    slopes,
                    62000 + test_index,
                    draws,
                )
            output.append(item)
            test_index += 1
    for family in ("global", "local"):
        valid = [
            item
            for item in output
            if item["metric_family"] == family
            and item["sign_flip_p"] is not None
        ]
        if valid:
            for item, qvalue in zip(
                valid,
                benjamini_hochberg([item["sign_flip_p"] for item in valid]),
            ):
                item["bh_q_e3b_monotonicity"] = qvalue
    return output


def descriptive_distribution_rows(
    per_image_effects: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = list(per_image_effects)
    keys = sorted(
        {
            (row["image_category"], row["masked_roi"], row["metric"])
            for row in rows
        }
    )
    output = []
    for category, masked_roi, metric in keys:
        values = np.asarray(
            [
                float(row["excess_causal_loss"])
                for row in rows
                if (
                    row["image_category"],
                    row["masked_roi"],
                    row["metric"],
                )
                == (category, masked_roi, metric)
            ],
            dtype=np.float64,
        )
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(
                f"Invalid descriptive distribution: "
                f"{category}/{masked_roi}/{metric}"
            )
        output.append(
            {
                "image_category": category,
                "masked_roi": masked_roi,
                "metric": metric,
                "num_images": int(len(values)),
                "mean": float(values.mean()),
                "std": (
                    float(values.std(ddof=1))
                    if len(values) > 1
                    else 0.0
                ),
                "min": float(values.min()),
                "q25": float(np.quantile(values, 0.25)),
                "median": float(np.median(values)),
                "q75": float(np.quantile(values, 0.75)),
                "max": float(values.max()),
                "positive_fraction": float(np.mean(values > 0)),
                "analysis_status": "descriptive_only",
            }
        )
    return output
