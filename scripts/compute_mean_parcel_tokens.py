#!/usr/bin/env python3
"""Compute the Subject 1 training-set mean of ParcelMapper output tokens."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "src"))

from brain_adapter.dataset import nsd_topk_parcel_dataset
from brain_adapter.model import GuidanceGenerator
from neuro_roi_causal.mean_cache import (
    atomic_torch_save,
    build_cache,
    file_sha256,
    load_validated_cache,
)
from neuro_roi_causal.model_wrapper import map_fmri_to_parcel_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def make_train_dataset(checkpoint: dict, project_root: Path):
    args = SimpleNamespace(
        subj=1,
        hemi=None,
        backbone_arch="dinov2_q",
        data_dir=str(project_root / "data" / "neuroadapter" / "neural_data"),
        imgs_dir=str(project_root / "data" / "nsd" / "stimuli"),
        parcel_dir=str(project_root / "data" / "neuroadapter" / "parcels" / "schaefer"),
        tokenizer=None,
        gen_size=512,
    )
    return nsd_topk_parcel_dataset(
        args,
        split="train",
        transform=None,
        topk=int(checkpoint["num_parcels"]) // 2,
        selected_parcel_idx=checkpoint["selected_parcel_idx"],
    )


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_sha = file_sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if args.output.exists():
        cache = load_validated_cache(
            args.output,
            checkpoint_sha256=checkpoint_sha,
            selected_parcel_idx=checkpoint["selected_parcel_idx"],
            subject=1,
        )
        print(f"Reusing validated cache: {args.output}")
        print(f"num_train_samples={cache['num_train_samples']}")
        return

    config = checkpoint["config"]
    guidance = GuidanceGenerator(
        num_parcels=int(checkpoint["num_parcels"]),
        max_voxels=int(checkpoint["max_voxels"]),
        num_decoder_queries=int(config["num_decoder_queries"]),
        output_dim=int(config["condition_dim"]),
        sub_approach=config["sub_approach"],
    )
    guidance.load_state_dict(checkpoint["guidance_generator"], strict=True)
    device = torch.device(args.device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    guidance.to(device=device, dtype=dtype).eval()
    dataset = make_train_dataset(checkpoint, args.project_root.resolve())

    total = None
    count = 0
    brain_batch: list[torch.Tensor] = []
    base = dataset.base_dataset
    with torch.no_grad():
        for index in tqdm(range(len(dataset)), desc="training parcel means"):
            data_indices = base.img_to_runs[index]
            lh = torch.from_numpy(base.betas[0][data_indices]).mean(dim=0)
            rh = torch.from_numpy(base.betas[1][data_indices]).mean(dim=0)
            brain_batch.append(
                torch.cat(
                    [
                        dataset.extract_and_pad(lh, hemi="lh"),
                        dataset.extract_and_pad(rh, hemi="rh"),
                    ],
                    dim=0,
                )
            )
            if len(brain_batch) < args.batch_size and index + 1 < len(dataset):
                continue
            brain = torch.stack(brain_batch).to(device=device, dtype=dtype)
            tokens = map_fmri_to_parcel_tokens(
                guidance, brain, int(checkpoint["num_parcels"])
            )
            summed = tokens.float().sum(dim=0).cpu()
            total = summed if total is None else total + summed
            count += int(tokens.shape[0])
            brain_batch.clear()
    if total is None or count != len(dataset):
        raise RuntimeError(f"Mean accumulation incomplete: {count}/{len(dataset)}")

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPRO_ROOT, text=True
    ).strip()
    cache = build_cache(
        total / count,
        subject=1,
        num_train_samples=count,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha,
        checkpoint_step=int(checkpoint["step"]),
        selected_parcel_idx=checkpoint["selected_parcel_idx"],
        sub_approach=config["sub_approach"],
        git_commit=commit,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    atomic_torch_save(cache, args.output)
    load_validated_cache(
        args.output,
        checkpoint_sha256=checkpoint_sha,
        selected_parcel_idx=checkpoint["selected_parcel_idx"],
        subject=1,
    )
    print(f"Wrote validated cache: {args.output}")
    print(f"shape={cache['shape']} num_train_samples={count}")


if __name__ == "__main__":
    main()
