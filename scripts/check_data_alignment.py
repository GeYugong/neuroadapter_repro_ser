#!/usr/bin/env python3
"""Check that saved GT PNGs match the NSD test dataset indices used in summaries."""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from torchvision import transforms

NEUROADAPTER = Path("/public/home/mty/GeYugong/projects/neuroadapter-iclr2026/code/NeuroAdapter")
if str(NEUROADAPTER) not in sys.path:
    sys.path.insert(0, str(NEUROADAPTER))

from brain_adapter.dataset import nsd_topk_parcel_dataset  # noqa: E402


def tensor_to_image_array(tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().clamp(0, 1)
    image = transforms.ToPILImage()(tensor).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=10)
    args = parser.parse_args()

    import torch

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    dataset_args = SimpleNamespace(
        subj=1,
        hemi=None,
        backbone_arch="dinov2_q",
        data_dir="/public/home/mty/GeYugong/data/neuroadapter/neural_data",
        imgs_dir="/public/home/mty/GeYugong/data/nsd/stimuli",
        parcel_dir="/public/home/mty/GeYugong/data/neuroadapter/parcels/schaefer",
        tokenizer=None,
        gen_size=512,
    )
    with contextlib.redirect_stdout(sys.stderr):
        dataset = nsd_topk_parcel_dataset(
            dataset_args,
            split="test",
            transform=None,
            topk=summary["topk"],
            selected_parcel_idx=ckpt["selected_parcel_idx"],
        )

    checked = []
    mismatches = []
    for record in summary["records"][: args.max_samples]:
        idx = int(record["dataset_idx"])
        dataset_item = dataset[idx]
        expected = tensor_to_image_array(dataset_item["img_encoder"])
        saved = np.asarray(Image.open(record["gt"]).convert("RGB"), dtype=np.uint8)
        same_shape = expected.shape == saved.shape
        max_abs_diff = int(np.max(np.abs(expected.astype(np.int16) - saved.astype(np.int16)))) if same_shape else None
        ok = same_shape and max_abs_diff == 0
        checked.append({"dataset_idx": idx, "same_shape": same_shape, "max_abs_diff": max_abs_diff, "ok": ok})
        if not ok:
            mismatches.append(checked[-1])

    result = {
        "summary": str(args.summary),
        "checkpoint": str(args.checkpoint),
        "checked_count": len(checked),
        "mismatch_count": len(mismatches),
        "checked": checked,
    }
    print(json.dumps(result, indent=2))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
