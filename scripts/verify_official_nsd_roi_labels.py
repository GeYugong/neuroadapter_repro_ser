#!/usr/bin/env python3
"""Convert official NSD .mgz ROI labels and compare them to a mirrored copy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


COLLECTIONS = ("prf-visualrois", "floc-faces", "floc-bodies", "floc-places", "floc-words")


def read_ctab(path: Path) -> dict[int, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            result[int(fields[0])] = fields[1]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"official_dir": str(args.official_dir), "mirror_dir": str(args.mirror_dir), "collections": {}}
    for collection in COLLECTIONS:
        ctab = read_ctab(args.official_dir / f"{collection}.mgz.ctab")
        np.save(args.output_dir / f"mapping_{collection}.npy", ctab, allow_pickle=True)
        collection_report = {"ctab": ctab, "hemispheres": {}}
        for hemisphere in ("lh", "rh"):
            official = np.rint(nib.load(str(args.official_dir / f"{hemisphere}.{collection}.mgz")).get_fdata()).astype(np.int16).reshape(-1)
            if official.shape != (163842,):
                raise ValueError(f"Unexpected {hemisphere}.{collection} shape {official.shape}")
            np.save(args.output_dir / f"{hemisphere}.{collection}_fsaverage_space.npy", official)
            mirrored_path = args.mirror_dir / f"{hemisphere}.{collection}_fsaverage_space.npy"
            mirrored = np.load(mirrored_path).reshape(-1) if mirrored_path.exists() else None
            collection_report["hemispheres"][hemisphere] = {
                "official_unique": {str(value): int((official == value).sum()) for value in np.unique(official)},
                "mirror_present": mirrored is not None,
                "same_shape": bool(mirrored is not None and mirrored.shape == official.shape),
                "exact_equal": bool(mirrored is not None and mirrored.shape == official.shape and np.array_equal(mirrored, official)),
                "different_vertices": None if mirrored is None or mirrored.shape != official.shape else int(np.count_nonzero(mirrored != official)),
            }
        report["collections"][collection] = collection_report
    (args.output_dir / "official_label_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
