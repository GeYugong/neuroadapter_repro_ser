"""Mechanical zero-to-mean E2 plan conversion and equivalence checks."""

from __future__ import annotations

import copy
from typing import Any


def convert_zero_plan(
    zero_plan: dict[str, Any],
    *,
    source_zero_plan: str,
    source_zero_plan_sha256: str,
    mean_token_cache: str,
    mean_token_cache_sha256: str,
    repository_commit: str,
    created_at: str,
) -> dict[str, Any]:
    plan = copy.deepcopy(zero_plan)
    plan["name"] = "E2_top_snr_causal_mean_full"
    plan["created_at"] = created_at
    plan["source_zero_plan"] = source_zero_plan
    plan["source_zero_plan_sha256"] = source_zero_plan_sha256
    plan["mean_token_cache"] = mean_token_cache
    plan["mean_token_cache_sha256"] = mean_token_cache_sha256
    plan["repository_commit"] = repository_commit
    plan["preregistration_status"] = "frozen_before_mean_inference"
    for category_plan in plan["categories"].values():
        for condition in category_plan["conditions"]:
            if condition["name"] in {"no_mask", "no_mask_repeat"}:
                continue
            if condition.get("mask_mode") != "zero":
                raise ValueError(f"Source condition is not zero-mask: {condition['name']}")
            condition["mask_mode"] = "mean"
            if not condition["name"].endswith("_zero"):
                raise ValueError(f"Source condition lacks _zero suffix: {condition['name']}")
            condition["name"] = condition["name"][:-5] + "_mean"
    return plan


def normalized_for_equivalence(plan: dict[str, Any], *, mean: bool) -> dict[str, Any]:
    value = copy.deepcopy(plan)
    for key in (
        "name",
        "created_at",
        "source_zero_plan",
        "source_zero_plan_sha256",
        "mean_token_cache",
        "mean_token_cache_sha256",
        "repository_commit",
        "preregistration_status",
    ):
        value.pop(key, None)
    for category_plan in value["categories"].values():
        for condition in category_plan["conditions"]:
            if condition["name"] in {"no_mask", "no_mask_repeat"}:
                continue
            if mean:
                if not condition["name"].endswith("_mean"):
                    raise ValueError(
                        f"Mean condition lacks _mean suffix: {condition['name']}"
                    )
                condition["name"] = condition["name"][:-5] + "_zero"
                condition["mask_mode"] = "zero"
    return value


def equivalence_audit(zero_plan: dict[str, Any], mean_plan: dict[str, Any]) -> dict[str, Any]:
    equal = normalized_for_equivalence(zero_plan, mean=False) == normalized_for_equivalence(
        mean_plan, mean=True
    )
    categories = {}
    for category in zero_plan["categories"]:
        zero_category = zero_plan["categories"][category]
        mean_category = mean_plan["categories"][category]
        categories[category] = {
            "dataset_indices_identical": (
                zero_category["dataset_indices"] == mean_category["dataset_indices"]
            ),
            "matching_audit_identical": (
                zero_category["matching_audit"] == mean_category["matching_audit"]
            ),
            "condition_count": len(mean_category["conditions"]),
        }
    passed = equal and all(
        item["dataset_indices_identical"] and item["matching_audit_identical"]
        for item in categories.values()
    )
    return {"passed": passed, "categories": categories, "normalized_plans_identical": equal}
