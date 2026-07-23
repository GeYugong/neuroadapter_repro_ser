#!/usr/bin/env python3
"""Build the complete Subject 1 parcel-to-ROI inventory and coverage report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "src"))

from neuro_roi_causal.mapping import (  # noqa: E402
    FUNCTIONAL_GROUPS,
    ROI_SPECS,
    dominant_assignment,
    parcel_overlaps,
)
from neuro_roi_causal.provenance import git_commit, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--parcel-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--roi-label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-overlap", type=float, default=0.5)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument(
        "--mapping-source-url",
        default="https://algonauts.csail.mit.edu/challenge.html",
    )
    return parser.parse_args()


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_name_map(path: Path) -> dict[int, str]:
    value = np.load(path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    return {int(key): str(name) for key, name in dict(value).items()}


def load_roi_inputs(roi_dir: Path, hemi: str) -> tuple[dict[str, np.ndarray], dict[str, dict[int, str]]]:
    arrays, names = {}, {}
    for collection in ROI_SPECS:
        array_path = roi_dir / f"{hemi}.{collection}_fsaverage_space.npy"
        mapping_path = roi_dir / f"mapping_{collection}.npy"
        arrays[collection] = np.load(array_path)
        names[collection] = load_name_map(mapping_path)
    return arrays, names


def build_rows(args: argparse.Namespace) -> tuple[list[dict], dict]:
    checkpoint = torch_load(args.checkpoint)
    metadata = np.load(args.metadata, allow_pickle=True).item()
    selected = {
        hemi: [int(value) for value in checkpoint["selected_parcel_idx"][hemi]]
        for hemi in ("lh", "rh")
    }
    rows: list[dict] = []
    for hemi in ("lh", "rh"):
        parcels = torch_load(args.parcel_dir / f"{hemi}_labels_s{args.subject:02}.pt")[1:]
        ncsnr = np.asarray(metadata[f"{hemi}_ncsnr"]).squeeze()
        roi_arrays, roi_names = load_roi_inputs(args.roi_label_dir, hemi)
        if any(array.shape != ncsnr.shape for array in roi_arrays.values()):
            shapes = {key: value.shape for key, value in roi_arrays.items()}
            raise ValueError(f"{hemi} ROI/ncsnr shape mismatch: ncsnr={ncsnr.shape}, roi={shapes}")

        mean_ncsnr = np.array([
            float(ncsnr[np.asarray(parcel, dtype=np.int64)].mean()) for parcel in parcels
        ])
        rank_order = np.argsort(mean_ncsnr)[::-1]
        ranks = np.empty(len(parcels), dtype=np.int64)
        ranks[rank_order] = np.arange(1, len(parcels) + 1)
        selected_position = {parcel_index: position for position, parcel_index in enumerate(selected[hemi])}
        hemi_offset = 0 if hemi == "lh" else len(selected["lh"])

        for parcel_index, parcel in enumerate(parcels):
            vertices = np.asarray(parcel, dtype=np.int64)
            overlaps = parcel_overlaps(vertices, roi_arrays, roi_names)
            group, level, dominant_overlap = dominant_assignment(overlaps, args.min_overlap)
            is_selected = parcel_index in selected_position
            rows.append({
                "hemisphere": hemi,
                "schaefer_parcel_index": parcel_index,
                "num_vertices": int(len(vertices)),
                "mean_ncsnr": float(mean_ncsnr[parcel_index]),
                "top_snr_rank_within_hemi": int(ranks[parcel_index]),
                "in_top_snr_200": is_selected,
                "top200_token_index": (
                    hemi_offset + selected_position[parcel_index] if is_selected else ""
                ),
                "dominant_roi": group,
                "functional_level": level,
                "dominant_overlap": dominant_overlap,
                "all_roi_overlaps": json.dumps(overlaps, ensure_ascii=False, separators=(",", ":")),
            })
    return rows, checkpoint


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def coverage_rows(frame: pd.DataFrame) -> list[dict]:
    result = []
    overall_rate = float(frame["in_top_snr_200"].mean())
    for group in (*FUNCTIONAL_GROUPS, "Unlabeled"):
        subset = frame[frame["dominant_roi"] == group]
        selected = subset[subset["in_top_snr_200"]]
        all_count = int(len(subset))
        top_count = int(len(selected))
        result.append({
            "roi_group": group,
            "all_parcels": all_count,
            "top200_parcels": top_count,
            "selection_rate": top_count / all_count if all_count else 0.0,
            "overall_selection_rate": overall_rate,
            "selection_rate_difference": (
                top_count / all_count - overall_rate if all_count else -overall_rate
            ),
            "mean_ncsnr_all": float(subset["mean_ncsnr"].mean()) if all_count else float("nan"),
            "mean_ncsnr_top200": (
                float(selected["mean_ncsnr"].mean()) if top_count else float("nan")
            ),
        })
    return result


def save_figures(frame: pd.DataFrame, coverage: pd.DataFrame, output_dir: Path) -> list[str]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    groups = list(FUNCTIONAL_GROUPS)

    fig, axis = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(groups))
    all_counts = [int((frame["dominant_roi"] == group).sum()) for group in groups]
    top_counts = [
        int(((frame["dominant_roi"] == group) & frame["in_top_snr_200"]).sum())
        for group in groups
    ]
    axis.bar(x - 0.2, all_counts, 0.4, label="All parcels")
    axis.bar(x + 0.2, top_counts, 0.4, label="Top-SNR-200")
    axis.set_xticks(x, groups)
    axis.set_ylabel("Parcel count")
    axis.legend()
    axis.set_title("Functional ROI coverage before and after top-SNR selection")
    path = figures / "roi_counts_all_vs_top200.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(output_dir)))

    fig, axis = plt.subplots(figsize=(10, 4.5))
    selected_coverage = coverage[coverage["roi_group"].isin(groups)]
    colors = ["#c44e52" if value < 0 else "#4c72b0" for value in selected_coverage["selection_rate_difference"]]
    axis.bar(selected_coverage["roi_group"], selected_coverage["selection_rate_difference"], color=colors)
    axis.axhline(0, color="black", linewidth=1)
    axis.set_ylabel("Selection rate minus overall rate")
    axis.set_title("ROI-specific top-SNR retention bias")
    path = figures / "roi_selection_rate_difference.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(output_dir)))

    fig, axis = plt.subplots(figsize=(11, 4.5))
    values = [frame.loc[frame["dominant_roi"] == group, "mean_ncsnr"].dropna().values for group in groups]
    axis.boxplot(values, tick_labels=groups, showfliers=False)
    axis.set_ylabel("Mean parcel ncsnr")
    axis.set_title("ncsnr distributions by functional ROI")
    path = figures / "roi_ncsnr_distributions.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(output_dir)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, (title, subset) in zip(
        axes,
        (("All parcels", frame), ("Top-SNR-200", frame[frame["in_top_snr_200"]])),
    ):
        counts = subset["dominant_roi"].value_counts().reindex([*groups, "Unlabeled"], fill_value=0)
        axis.bar(counts.index, counts.values)
        axis.tick_params(axis="x", rotation=45)
        axis.set_title(title)
    axes[0].set_ylabel("Parcel count")
    path = figures / "functional_composition_all_vs_top200.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(output_dir)))

    fig, axis = plt.subplots(figsize=(11, 4.5))
    hemi = (
        frame[frame["dominant_roi"].isin(groups)]
        .groupby(["dominant_roi", "hemisphere"])
        .size()
        .unstack(fill_value=0)
        .reindex(groups, fill_value=0)
    )
    hemi.plot.bar(ax=axis)
    axis.set_xlabel("")
    axis.set_ylabel("Parcel count")
    axis.set_title("Functional ROI distribution by hemisphere")
    path = figures / "roi_hemisphere_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path.relative_to(output_dir)))
    return paths


def main() -> None:
    args = parse_args()
    if not 0 < args.min_overlap <= 1:
        raise ValueError("--min-overlap must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, checkpoint = build_rows(args)
    frame = pd.DataFrame(rows)
    if len(frame) != 1000:
        raise ValueError(f"Expected 1000 non-medial-wall parcels, found {len(frame)}")
    if int(frame["in_top_snr_200"].sum()) != 200:
        raise ValueError("Checkpoint selection does not identify exactly 200 parcels")

    full_path = args.output_dir / "algonauts_full_parcel_inventory_subj01.csv"
    top_path = args.output_dir / "algonauts_top200_mapping_subj01.csv"
    write_csv(full_path, rows)
    top_rows = sorted(
        (row for row in rows if row["in_top_snr_200"]),
        key=lambda row: int(row["top200_token_index"]),
    )
    write_csv(top_path, top_rows)

    coverage = coverage_rows(frame)
    coverage_path = args.output_dir / "roi_coverage_all_vs_top200.csv"
    write_csv(coverage_path, coverage)
    coverage_frame = pd.DataFrame(coverage)
    figure_paths = save_figures(frame, coverage_frame, args.output_dir)

    key_groups = {"Face", "Word", "V4"}
    bias = {
        row["roi_group"]: {
            "selection_rate": row["selection_rate"],
            "overall_selection_rate": row["overall_selection_rate"],
            "underrepresented": row["selection_rate_difference"] < 0,
        }
        for row in coverage
        if row["roi_group"] in key_groups
    }
    top_counts = Counter(row["dominant_roi"] for row in top_rows)
    metadata = {
        "schema_version": 1,
        "subject": args.subject,
        "mapping_source": "Algonauts Project 2023 Subject 1 fsaverage ROI masks",
        "mapping_source_url": args.mapping_source_url,
        "overlap_rule": f"strictly greater than {args.min_overlap}",
        "num_all_parcels": len(rows),
        "num_top_snr_parcels": len(top_rows),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "top200_counts": {key: int(top_counts.get(key, 0)) for key in (*FUNCTIONAL_GROUPS, "Unlabeled")},
        "selection_bias_summary": bias,
        "figures": figure_paths,
        "official_nsd_sensitivity_analysis": (
            "../roi_ablation/mapping/official_mapping_subj01_nsd_fsaverage_reg"
        ),
        "git_commit": git_commit(REPRO_ROOT),
    }
    (args.output_dir / "mapping_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    input_paths = [
        args.checkpoint,
        args.metadata,
        args.parcel_dir / f"lh_labels_s{args.subject:02}.pt",
        args.parcel_dir / f"rh_labels_s{args.subject:02}.pt",
    ]
    input_paths.extend(sorted(args.roi_label_dir.glob("*.npy")))
    hashes = {
        path.name: {"path_role": f"input:{path.name}", "sha256": sha256_file(path)}
        for path in input_paths
    }
    output_paths = [full_path, top_path, coverage_path, args.output_dir / "mapping_metadata.json"]
    output_paths.extend(args.output_dir / path for path in figure_paths)
    hashes.update({
        path.name: {
            "path_role": f"generated:{path.relative_to(args.output_dir).as_posix()}",
            "sha256": sha256_file(path),
        }
        for path in output_paths
    })
    (args.output_dir / "file_hashes.json").write_text(
        json.dumps(hashes, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
