#!/usr/bin/env python3
"""Resample official NSD native-surface ROI labels onto fsaverage by sphere.reg."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.freesurfer.io import read_geometry
from scipy.spatial import cKDTree


COLLECTIONS = ("prf-visualrois", "floc-faces", "floc-bodies", "floc-places", "floc-words")


def unit(coords: np.ndarray) -> np.ndarray:
    return coords / np.linalg.norm(coords, axis=1, keepdims=True)


def load_target_sphere(fsaverage_dir: Path, hemi: str, side: str) -> tuple[np.ndarray, str]:
    """Use NSD's fsaverage registration sphere when available.

    A surface named ``sphere.gii`` is not necessarily the registration sphere
    used by FreeSurfer's ``mri_surf2surf``.  The official NSD fsaverage
    ``sphere.reg`` is therefore preferred; the GIFTI path remains a backwards
    compatible fallback for earlier local setups.
    """
    registration_sphere = fsaverage_dir / f"{hemi}.sphere.reg"
    if registration_sphere.exists():
        coords, _ = read_geometry(str(registration_sphere))
        return coords, "official NSD fsaverage sphere.reg"

    gifti_sphere = fsaverage_dir / f"sphere_{side}.gii.gz"
    if not gifti_sphere.exists():
        raise FileNotFoundError(
            f"Neither {registration_sphere} nor {gifti_sphere} exists"
        )
    coords = nib.load(str(gifti_sphere)).agg_data("pointset")
    return coords, "GIFTI fsaverage sphere fallback"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--sphere-dir", type=Path, required=True)
    parser.add_argument("--fsaverage-dir", type=Path, required=True)
    parser.add_argument("--mirror-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"collections": {}}
    for collection in COLLECTIONS:
        mapping_path = args.mirror_dir / f"mapping_{collection}.npy"
        if not mapping_path.exists():
            raise FileNotFoundError(f"Missing ROI code table: {mapping_path}")
        shutil.copy2(mapping_path, args.output_dir / mapping_path.name)
    for hemi, side in (("lh", "left"), ("rh", "right")):
        source_coords, _ = read_geometry(str(args.sphere_dir / f"subj01_{hemi}.sphere.reg"))
        target_coords, target_source = load_target_sphere(args.fsaverage_dir, hemi, side)
        if target_coords.shape[0] != 163842:
            raise ValueError(f"Unexpected fsaverage vertex count {target_coords.shape}")
        distances, nearest = cKDTree(unit(source_coords)).query(unit(target_coords), k=1)
        np.save(args.output_dir / f"{hemi}_nearest_native_index.npy", nearest)
        np.save(args.output_dir / f"{hemi}_nearest_sphere_distance.npy", distances)
        report[hemi] = {
            "source_vertices": int(source_coords.shape[0]),
            "target_vertices": int(target_coords.shape[0]),
            "target_sphere_source": target_source,
            "mean_nearest_distance": float(distances.mean()),
            "max_nearest_distance": float(distances.max()),
        }
        for collection in COLLECTIONS:
            native = np.rint(nib.load(str(args.official_dir / f"{hemi}.{collection}.mgz")).get_fdata()).astype(np.int16).reshape(-1)
            resampled = native[nearest]
            np.save(args.output_dir / f"{hemi}.{collection}_fsaverage_space.npy", resampled)
            mirror = np.load(args.mirror_dir / f"{hemi}.{collection}_fsaverage_space.npy").reshape(-1)
            report["collections"].setdefault(collection, {})[hemi] = {
                "exact_equal_to_mirror": bool(np.array_equal(resampled, mirror)),
                "different_vertices": int(np.count_nonzero(resampled != mirror)),
                "resampled_counts": {str(k): int((resampled == k).sum()) for k in np.unique(resampled)},
                "mirror_counts": {str(k): int((mirror == k).sum()) for k in np.unique(mirror)},
            }
    (args.output_dir / "official_fsaverage_resampling_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
