#!/usr/bin/env python3
"""Map NeuroAdapter's selected Subject 1 Schaefer tokens to NSD functional ROIs.

The mapping follows NeuroAdapter Appendix P: a parcel receives an ROI label when
more than ``min_overlap`` of its fsaverage vertices lie in that ROI.  The script
uses ``selected_parcel_idx`` stored in a checkpoint, so token positions match the
actual model input exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


ROI_SPECS = {
    "prf-visualrois": {
        "group_by_code": {
            "V1v": "V1", "V1d": "V1", "V2v": "V2", "V2d": "V2",
            "V3v": "V3", "V3d": "V3", "hV4": "V4",
        },
        "level": "low_level",
    },
    "floc-faces": {
        "group_by_code": {"OFA": "Face", "FFA-1": "Face", "FFA-2": "Face", "mTL-faces": "Face", "aTL-faces": "Face"},
        "level": "high_level",
    },
    "floc-bodies": {
        "group_by_code": {"EBA": "Body", "FBA-1": "Body", "FBA-2": "Body", "mTL-bodies": "Body"},
        "level": "high_level",
    },
    "floc-places": {
        "group_by_code": {"OPA": "Scene", "PPA": "Scene", "RSC": "Scene"},
        "level": "high_level",
    },
    "floc-words": {
        "group_by_code": {"OWFA": "Word", "VWFA-1": "Word", "VWFA-2": "Word", "mfs-words": "Word", "mTL-words": "Word"},
        "level": "high_level",
    },
}

COLORS = {
    "Unlabeled": "#b9c0c8", "V1": "#1f77b4", "V2": "#4e9bd1", "V3": "#73bfe2", "V4": "#a6d8ef",
    "Face": "#d62728", "Body": "#ff7f0e", "Scene": "#2ca02c", "Word": "#9467bd", "Ambiguous": "#4d4d4d",
}


def torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_mapping(path: Path) -> dict[int, str]:
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return {int(key): str(name) for key, name in dict(value).items()}


def dominant_group(qualifying: list[dict[str, Any]]) -> tuple[str, str]:
    if not qualifying:
        return "Unlabeled", "unlabeled"
    # NSD fLoc collections may overlap. Mirror the released helper's behavior:
    # assign each parcel to the single ROI with the largest qualifying overlap.
    best = max(qualifying, key=lambda item: (item["overlap"], item["group"], item["roi"]))
    return str(best["group"]), str(best["level"])


def plot_distribution(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 3.8), constrained_layout=True)
    for axis, hemi in zip(axes, ("lh", "rh")):
        hemi_rows = [row for row in rows if row["hemisphere"] == hemi]
        for row in hemi_rows:
            pos = int(row["token_position_within_hemi"])
            x, y = pos % 25, 3 - pos // 25
            axis.add_patch(plt.Rectangle((x, y), 0.95, 0.95, color=COLORS[row["dominant_group"]]))
            axis.text(x + 0.475, y + 0.475, str(pos), ha="center", va="center", fontsize=5.5)
        axis.set_xlim(0, 25)
        axis.set_ylim(0, 4)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"{hemi.upper()}: 100 highest-SNR selected parcels (token order)", loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, label=name) for name, color in COLORS.items()]
    fig.legend(handles=handles, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parcel-dir", type=Path, required=True)
    parser.add_argument("--neural-data-dir", type=Path, required=True)
    parser.add_argument("--roi-label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-overlap", type=float, default=0.5)
    args = parser.parse_args()

    if not 0 < args.min_overlap <= 1:
        raise ValueError("--min-overlap must be in (0, 1]")
    checkpoint = torch_load(args.checkpoint)
    selected = checkpoint["selected_parcel_idx"]
    metadata = np.load(args.neural_data_dir / "metadata_sub-01.npy", allow_pickle=True).item()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    roi_arrays: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    roi_name_maps: dict[str, dict[int, str]] = {}
    for collection in ROI_SPECS:
        roi_name_maps[collection] = load_mapping(args.roi_label_dir / f"mapping_{collection}.npy")
        for hemi in ("lh", "rh"):
            array = np.load(args.roi_label_dir / f"{hemi}.{collection}_fsaverage_space.npy")
            if array.shape != (163842,):
                raise ValueError(f"Unexpected {hemi}.{collection} shape: {array.shape}")
            roi_arrays[collection][hemi] = array

    rows: list[dict[str, Any]] = []
    groups: dict[str, list[int]] = defaultdict(list)
    for hemi_index, hemi in enumerate(("lh", "rh")):
        parcels = torch_load(args.parcel_dir / f"{hemi}_labels_s01.pt")[1:]
        snr = np.asarray(metadata[f"{hemi}_ncsnr"]).squeeze()
        for local_position, parcel_index in enumerate(selected[hemi]):
            parcel_index = int(parcel_index)
            vertices = np.asarray(parcels[parcel_index], dtype=np.int64)
            global_position = hemi_index * len(selected[hemi]) + local_position
            qualifying: list[dict[str, Any]] = []
            for collection, spec in ROI_SPECS.items():
                labels = roi_arrays[collection][hemi][vertices]
                for code, count in Counter(labels.tolist()).items():
                    if int(code) == 0:
                        continue
                    roi_name = roi_name_maps[collection].get(int(code), f"unknown_{code}")
                    if roi_name not in spec["group_by_code"]:
                        continue
                    overlap = count / len(vertices)
                    if overlap > args.min_overlap:
                        qualifying.append({
                            "collection": collection,
                            "roi": roi_name,
                            "group": spec["group_by_code"][roi_name],
                            "level": spec["level"],
                            "overlap": round(float(overlap), 6),
                        })
            group, level = dominant_group(qualifying)
            if group != "Unlabeled":
                groups[group].append(global_position)
            if level in {"low_level", "high_level"}:
                groups[level].append(global_position)
            rows.append({
                "global_token_index": global_position,
                "hemisphere": hemi,
                "token_position_within_hemi": local_position,
                "schaefer_parcel_index": parcel_index,
                "num_vertices": int(len(vertices)),
                "mean_ncsnr": round(float(snr[vertices].mean()), 6),
                "dominant_group": group,
                "functional_level": level,
                "qualifying_labels": json.dumps(qualifying, ensure_ascii=False),
            })

    fieldnames = list(rows[0])
    with (args.output_dir / "roi_inventory_subj01.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "mapping_rule": f"ROI overlap strictly greater than {args.min_overlap:.0%}",
        "source": "NSD/Algonauts fsaverage ROI label arrays",
        "num_selected_tokens": len(rows),
        "group_counts": {key: len(value) for key, value in sorted(groups.items())},
        "unlabeled_tokens": sum(row["dominant_group"] == "Unlabeled" for row in rows),
        "ambiguous_tokens": 0,
        "groups": {key: sorted(value) for key, value in sorted(groups.items())},
    }
    (args.output_dir / "roi_groups_subj01.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_distribution(rows, args.output_dir / "roi_parcel_distribution_subj01.png")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
