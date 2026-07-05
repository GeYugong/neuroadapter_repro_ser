#!/usr/bin/env python3
"""Convert Schaefer2018 fsaverage annotation files into NeuroAdapter parcel labels.

Expected inputs:
  /public/home/mty/GeYugong/data/parcels/schaefer/raw/
    lh.Schaefer2018_1000Parcels_7Networks_order.annot
    rh.Schaefer2018_1000Parcels_7Networks_order.annot

Outputs:
  /public/home/mty/GeYugong/data/neuroadapter/parcels/schaefer/
    lh_labels_s01.pt
    rh_labels_s01.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel.freesurfer.io as fsio
import torch


def convert(raw_dir: Path, out_dir: Path, subj: int, parcels: int, networks: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for hemi in ["lh", "rh"]:
        annot_path = raw_dir / f"{hemi}.Schaefer2018_{parcels}Parcels_{networks}Networks_order.annot"
        labels, _ctab, _names = fsio.read_annot(str(annot_path))
        label_tensor = torch.from_numpy(labels.astype("int64", copy=True)).long()

        parcel_indices = []
        for label_id in sorted(set(labels.tolist())):
            parcel_indices.append(torch.where(label_tensor == int(label_id))[0].long())

        torch.save(parcel_indices, out_dir / f"{hemi}_labels_s{subj:02}.pt")
        summary[hemi] = {
            "n_vertices": int(labels.shape[0]),
            "n_parcels_including_medial_wall": len(parcel_indices),
            "n_parcels_excluding_medial_wall": len(parcel_indices) - 1,
            "medial_wall_vertices": int(parcel_indices[0].numel()),
            "min_parcel_size_excluding_medial_wall": min(int(x.numel()) for x in parcel_indices[1:]),
            "max_parcel_size_excluding_medial_wall": max(int(x.numel()) for x in parcel_indices[1:]),
        }

    (out_dir / "source.txt").write_text(
        f"Schaefer2018_{parcels}Parcels_{networks}Networks_order.annot from "
        "ThomasYeoLab/CBIG, FreeSurfer5.3/fsaverage/label. "
        "Index 0 is Background+FreeSurfer_Defined_Medial_Wall.\n"
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("/public/home/mty/GeYugong/data/parcels/schaefer/raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("/public/home/mty/GeYugong/data/neuroadapter/parcels/schaefer"))
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--parcels", type=int, default=1000)
    parser.add_argument("--networks", type=int, default=7)
    args = parser.parse_args()

    summary = convert(args.raw_dir, args.out_dir, args.subj, args.parcels, args.networks)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
