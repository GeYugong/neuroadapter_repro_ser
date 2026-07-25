"""ROI overlap parsing and contamination summaries for frozen E2 controls."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Iterable


FUNCTIONAL_GROUPS = ("V1", "V2", "V3", "V4", "Face", "Body", "Scene", "Word")


def parse_all_roi_overlaps(value: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = json.loads(value) if isinstance(value, str) else value
    if not isinstance(records, list):
        raise ValueError("all_roi_overlaps must be a JSON list")
    parsed = []
    for item in records:
        if not isinstance(item, dict) or not {"collection", "roi", "group", "overlap"} <= set(item):
            raise ValueError("Malformed all_roi_overlaps record")
        record = dict(item)
        record["overlap"] = float(record["overlap"])
        if not 0.0 <= record["overlap"] <= 1.0:
            raise ValueError("ROI overlap must be between zero and one")
        parsed.append(record)
    return parsed


def group_max_overlaps(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    result = {group: 0.0 for group in FUNCTIONAL_GROUPS}
    for record in records:
        group = str(record["group"])
        if group in result:
            result[group] = max(result[group], float(record["overlap"]))
    return result


def specific_overlap_columns(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for record in records:
        key = f"roi_overlap__{record['collection']}__{record['roi']}"
        values[key] = max(values.get(key, 0.0), float(record["overlap"]))
    return values


def summarize_parcel(row: dict[str, str], target_group: str) -> dict[str, Any]:
    records = parse_all_roi_overlaps(row["all_roi_overlaps"])
    groups = group_max_overlaps(records)
    off_target = max(value for group, value in groups.items() if group != target_group)
    target = groups[target_group]
    summary: dict[str, Any] = {
        "top200_token_index": int(row["top200_token_index"]),
        "hemisphere": row["hemisphere"],
        "schaefer_parcel_index": int(row["schaefer_parcel_index"]),
        "dominant_roi": row["dominant_roi"],
        "dominant_overlap": float(row["dominant_overlap"]),
        "mean_ncsnr": float(row["mean_ncsnr"]),
        "num_vertices": int(row["num_vertices"]),
        "target_group_max_overlap": target,
        "max_off_target_overlap": off_target,
        "purity_margin": target - off_target,
    }
    summary.update({f"group_overlap__{group}": value for group, value in groups.items()})
    summary.update(specific_overlap_columns(records))
    for threshold in (0.10, 0.25, 0.40, 0.50):
        label = f"{threshold:.2f}"
        summary[f"num_groups_ge_{label}"] = sum(value >= threshold for value in groups.values())
    summary["target_off_target_dual_high_overlap"] = target >= 0.50 and off_target >= 0.50
    return summary


def count_candidates_by_hemisphere(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["hemisphere"])] += 1
    return dict(counts)
