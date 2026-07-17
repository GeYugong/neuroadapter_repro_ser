#!/usr/bin/env python3
"""Audit the parcel files and top-k selection used by a NeuroAdapter checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def parcel_vector(parcel_path: Path) -> np.ndarray:
    parcels = torch_load(parcel_path)
    labels = np.full(163842, -1, dtype=np.int32)
    for index, vertices in enumerate(parcels):
        labels[np.asarray(vertices, dtype=np.int64)] = index
    if np.any(labels < 0):
        raise ValueError(f"Parcel file does not cover fsaverage: {parcel_path}")
    return labels


def partition_agreement(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    n_labels = int(max(first.max(), second.max()) + 1)
    contingency = np.bincount(
        first.astype(np.int64) * n_labels + second.astype(np.int64),
        minlength=n_labels * n_labels,
    ).reshape(n_labels, n_labels)
    first_sizes = contingency.sum(axis=1)
    second_sizes = contingency.sum(axis=0)
    best_second = contingency.argmax(axis=1)
    intersection = contingency[np.arange(n_labels), best_second]
    union = first_sizes + second_sizes[best_second] - intersection
    jaccard = intersection / union
    # Index 0 is the medial-wall/background label and is not a model token.
    return {
        "best_match_jaccard_mean_excluding_background": float(jaccard[1:].mean()),
        "best_match_jaccard_min_excluding_background": float(jaccard[1:].min()),
        "exact_best_matches_excluding_background": int((jaccard[1:] == 1.0).sum()),
        "same_index_partitions_excluding_background": int(
            sum(np.array_equal(first == index, second == index) for index in range(1, n_labels))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--active-parcel-dir", type=Path, required=True)
    parser.add_argument("--reference-parcel-dir", type=Path, required=True)
    parser.add_argument("--official-ncsnr-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = torch_load(args.checkpoint)
    selected = checkpoint["selected_parcel_idx"]
    report: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "active_parcel_dir": str(args.active_parcel_dir),
        "reference_parcel_dir": str(args.reference_parcel_dir),
        "official_ncsnr_dir": str(args.official_ncsnr_dir),
        "hemispheres": {},
    }
    active_vectors = {}
    reference_vectors = {}
    for hemi in ("lh", "rh"):
        active_path = args.active_parcel_dir / f"{hemi}_labels_s01.pt"
        reference_path = args.reference_parcel_dir / f"{hemi}_labels_s01.pt"
        active_vectors[hemi] = parcel_vector(active_path)
        reference_vectors[hemi] = parcel_vector(reference_path)
        ncsnr = np.asarray(
            nib.load(str(args.official_ncsnr_dir / f"{hemi}.ncsnr.mgh")).get_fdata()
        ).squeeze()
        topk = np.argsort(
            [ncsnr[np.where(active_vectors[hemi] == index)[0]].mean() for index in range(1, 501)]
        )[::-1][:100]
        checkpoint_topk = np.asarray(selected[hemi], dtype=np.int64)
        report["hemispheres"][hemi] = {
            "checkpoint_topk_matches_official_ncsnr": bool(np.array_equal(topk, checkpoint_topk)),
            "topk_set_overlap": int(len(set(topk.tolist()) & set(checkpoint_topk.tolist()))),
            "active_vs_reference": partition_agreement(active_vectors[hemi], reference_vectors[hemi]),
        }
    report["reference_left_right_partition_agreement"] = partition_agreement(
        reference_vectors["lh"], reference_vectors["rh"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
