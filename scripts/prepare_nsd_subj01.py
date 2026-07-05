#!/usr/bin/env python3
"""Prepare NSD subject 1 files expected by NeuroAdapter.

Inputs are the official NSD AWS files downloaded under /public/home/mty/GeYugong/data/nsd.
Outputs:
  data/neuroadapter/neural_data/metadata_sub-01.npy
  data/neuroadapter/neural_data/betas_sub-01.h5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
import scipy.io as sio
from tqdm import tqdm


def load_mgh_session(path: Path) -> np.ndarray:
    data = nib.load(str(path)).get_fdata(dtype=np.float32)
    return np.asarray(data[:, 0, 0, :], dtype=np.float32).T  # [750, 163842]


def build_metadata(nsd_root: Path, out_dir: Path, subj: int) -> dict:
    exp_path = nsd_root / "experiments" / "nsd" / "nsd_expdesign.mat"
    design = sio.loadmat(exp_path)

    subjectim = np.asarray(design["subjectim"], dtype=np.int64)[subj - 1]
    masterordering = np.asarray(design["masterordering"], dtype=np.int64).reshape(-1)
    sharedix = np.asarray(design["sharedix"], dtype=np.int64).reshape(-1)

    # NSD MATLAB files use 1-based image IDs. NeuroAdapter indexes imgBrick with 0-based IDs.
    img_presentation_order = subjectim[masterordering - 1] - 1
    all_subject_imgs = subjectim - 1
    test_img_num = sharedix - 1
    train_img_num = np.setdiff1d(all_subject_imgs, test_img_num, assume_unique=False)
    val_img_num = train_img_num[:0]

    beta_dir = nsd_root / "betas" / f"subj{subj:02}" / "fsaverage" / "betas_fithrf_GLMdenoise_RR"
    metadata = {
        "img_presentation_order": img_presentation_order.astype(np.int64),
        "train_img_num": train_img_num.astype(np.int64),
        "test_img_num": test_img_num.astype(np.int64),
        "val_img_num": val_img_num.astype(np.int64),
    }

    for hemi in ["lh", "rh"]:
        ncsnr_path = beta_dir / f"{hemi}.ncsnr.mgh"
        if ncsnr_path.exists():
            ncsnr = nib.load(str(ncsnr_path)).get_fdata(dtype=np.float32)
            metadata[f"{hemi}_ncsnr"] = np.asarray(ncsnr[:, 0, 0], dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"metadata_sub-{subj:02}.npy", metadata, allow_pickle=True)
    return metadata


def build_betas(nsd_root: Path, out_dir: Path, subj: int, overwrite: bool) -> None:
    out_path = out_dir / f"betas_sub-{subj:02}.h5"
    if out_path.exists() and not overwrite:
        print(f"[skip] {out_path} already exists")
        return

    beta_dir = nsd_root / "betas" / f"subj{subj:02}" / "fsaverage" / "betas_fithrf_GLMdenoise_RR"
    tmp_path = out_path.with_suffix(".h5.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    with h5py.File(tmp_path, "w") as h5:
        for hemi in ["lh", "rh"]:
            dset = h5.create_dataset(
                f"{hemi}_betas",
                shape=(30000, 163842),
                dtype="float32",
                chunks=(16, 163842),
                compression="lzf",
            )
            cursor = 0
            for sess in tqdm(range(1, 41), desc=f"{hemi} sessions"):
                session_path = beta_dir / f"{hemi}.betas_session{sess:02}.mgh"
                arr = load_mgh_session(session_path)
                n_trials = arr.shape[0]
                dset[cursor : cursor + n_trials, :] = arr
                cursor += n_trials
            if cursor != 30000:
                raise RuntimeError(f"{hemi}: expected 30000 trials, got {cursor}")
    tmp_path.rename(out_path)
    print(f"[ok] wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsd-root", type=Path, default=Path("/public/home/mty/GeYugong/data/nsd"))
    parser.add_argument("--out-dir", type=Path, default=Path("/public/home/mty/GeYugong/data/neuroadapter/neural_data"))
    parser.add_argument("--subj", type=int, default=1)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata = build_metadata(args.nsd_root, args.out_dir, args.subj)
    print("[metadata] trials", metadata["img_presentation_order"].shape)
    print("[metadata] train/test/val", len(metadata["train_img_num"]), len(metadata["test_img_num"]), len(metadata["val_img_num"]))
    print("[metadata] first image ids", metadata["img_presentation_order"][:10].tolist())

    if not args.metadata_only:
        build_betas(args.nsd_root, args.out_dir, args.subj, args.overwrite)


if __name__ == "__main__":
    main()
