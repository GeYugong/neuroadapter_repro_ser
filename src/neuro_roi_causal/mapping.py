"""Functional ROI definitions and mapping utilities."""

from __future__ import annotations

from collections import Counter

import numpy as np


ROI_SPECS = {
    "prf-visualrois": {
        "group_by_code": {
            "V1v": "V1", "V1d": "V1", "V2v": "V2", "V2d": "V2",
            "V3v": "V3", "V3d": "V3", "hV4": "V4",
        },
        "level": "low_level",
    },
    "floc-faces": {
        "group_by_code": {
            "OFA": "Face", "FFA-1": "Face", "FFA-2": "Face",
            "mTL-faces": "Face", "aTL-faces": "Face",
        },
        "level": "high_level",
    },
    "floc-bodies": {
        "group_by_code": {
            "EBA": "Body", "FBA-1": "Body", "FBA-2": "Body",
            "mTL-bodies": "Body",
        },
        "level": "high_level",
    },
    "floc-places": {
        "group_by_code": {"OPA": "Scene", "PPA": "Scene", "RSC": "Scene"},
        "level": "high_level",
    },
    "floc-words": {
        "group_by_code": {
            "OWFA": "Word", "VWFA-1": "Word", "VWFA-2": "Word",
            "mfs-words": "Word", "mTL-words": "Word",
        },
        "level": "high_level",
    },
}

FUNCTIONAL_GROUPS = ("V1", "V2", "V3", "V4", "Face", "Body", "Scene", "Word")


def parcel_overlaps(
    vertices: np.ndarray,
    roi_arrays: dict[str, np.ndarray],
    roi_name_maps: dict[str, dict[int, str]],
) -> list[dict]:
    overlaps: list[dict] = []
    if len(vertices) == 0:
        return overlaps
    for collection, labels_by_vertex in roi_arrays.items():
        labels = labels_by_vertex[vertices]
        spec = ROI_SPECS[collection]
        for code, count in Counter(labels.tolist()).items():
            code = int(code)
            if code == 0:
                continue
            roi_name = roi_name_maps[collection].get(code, f"unknown_{code}")
            group = spec["group_by_code"].get(roi_name)
            if group is None:
                continue
            overlaps.append({
                "collection": collection,
                "roi": roi_name,
                "group": group,
                "level": spec["level"],
                "overlap": float(count / len(vertices)),
            })
    return sorted(overlaps, key=lambda item: (-item["overlap"], item["roi"]))


def dominant_assignment(overlaps: list[dict], threshold: float) -> tuple[str, str, float]:
    qualifying = [item for item in overlaps if float(item["overlap"]) > threshold]
    if not qualifying:
        return "Unlabeled", "unlabeled", 0.0
    best = max(
        qualifying,
        key=lambda item: (float(item["overlap"]), item["group"], item["roi"]),
    )
    return str(best["group"]), str(best["level"]), float(best["overlap"])

