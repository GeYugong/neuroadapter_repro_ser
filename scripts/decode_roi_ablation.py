#!/usr/bin/env python3
"""Run deterministic inference-time ROI token ablations for a NeuroAdapter checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "src"))

from brain_adapter.dataset import nsd_topk_parcel_dataset
from brain_adapter.model import GuidanceGenerator
from neuro_roi_causal.model_wrapper import forward_with_parcel_intervention, map_fmri_to_parcel_tokens

from decode_limited import BrainIPAdapter, load_diffusion_models, run_diffusion, setup_ip_adapter_modules


PROJECT_ROOT = Path("/public/home/mty/GeYugong/projects/neuroadapter-iclr2026")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--roi-groups", type=Path, required=True)
    parser.add_argument("--mask-group", required=True, help="no_mask or a key in roi_groups_subj01.json")
    parser.add_argument("--mask-mode", choices=("zero", "mean"), default="zero")
    parser.add_argument("--mean-token-path", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--denoising-steps", type=int, default=50)
    parser.add_argument("--noise-factor", type=float, default=4.0)
    parser.add_argument("--mean-batch-size", type=int, default=8)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def make_dataset(tokenizer, checkpoint: dict, split: str, topk: int):
    dataset_args = SimpleNamespace(
        subj=1,
        hemi=None,
        backbone_arch="dinov2_q",
        data_dir="/public/home/mty/GeYugong/data/neuroadapter/neural_data",
        imgs_dir="/public/home/mty/GeYugong/data/nsd/stimuli",
        parcel_dir="/public/home/mty/GeYugong/data/neuroadapter/parcels/schaefer",
        tokenizer=tokenizer,
        gen_size=512,
    )
    return nsd_topk_parcel_dataset(
        dataset_args,
        split=split,
        transform=None,
        topk=topk,
        selected_parcel_idx=checkpoint["selected_parcel_idx"],
    )


def compute_mean_parcel_tokens(
    guidance_generator: GuidanceGenerator,
    train_dataset,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    output_path: Path,
) -> torch.Tensor:
    """Compute E_train[ParcelMapper(fMRI)] once and cache it by checkpoint."""
    if output_path.exists():
        cached = torch.load(output_path, map_location="cpu")
        if "mean_parcel_tokens" not in cached:
            raise ValueError(
                f"{output_path} is a legacy condition-token mean cache; "
                "parcel-level mean masking requires regeneration"
            )
        return cached["mean_parcel_tokens"].to(device=device, dtype=dtype)

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total = None
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="mean condition tokens"):
            brain_data = torch.cat(
                [batch["brain_lh_f"].to(device=device, dtype=dtype), batch["brain_rh_f"].to(device=device, dtype=dtype)],
                dim=1,
            )
            parcel_tokens = map_fmri_to_parcel_tokens(
                guidance_generator, brain_data, guidance_generator.parcel_mapper.num_parcels
            )
            token_sum = parcel_tokens.float().sum(dim=0)
            total = token_sum if total is None else total + token_sum
            count += parcel_tokens.shape[0]
    if total is None or count == 0:
        raise RuntimeError("Training dataset produced no condition tokens")
    mean = total / count
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mean_parcel_tokens": mean.cpu(),
        "num_train_samples": count,
        "intervention_stage": "after_parcel_mapper_before_token_mapper",
    }, output_path)
    return mean.to(device=device, dtype=dtype)


def to_pil(tensor: torch.Tensor) -> Image.Image:
    return transforms.ToPILImage()(tensor.detach().cpu().clamp(0, 1))


def save_pair(gt: Image.Image, pred: Image.Image, path: Path) -> Image.Image:
    gt = gt.resize((256, 256))
    pred = pred.resize((256, 256))
    canvas = Image.new("RGB", (512, 256), "white")
    canvas.paste(gt, (0, 0))
    canvas.paste(pred, (256, 0))
    canvas.save(path)
    return canvas


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.start_idx < 0:
        raise ValueError("num-samples must be positive and start-idx non-negative")
    group_data = json.loads(args.roi_groups.read_text(encoding="utf-8"))
    if args.mask_group == "no_mask":
        mask_indices: list[int] = []
    else:
        try:
            mask_indices = [int(index) for index in group_data["groups"][args.mask_group]]
        except KeyError as error:
            raise ValueError(f"Unknown mask group: {args.mask_group}") from error
    if args.mask_group == "no_mask" and args.mask_mode != "zero":
        raise ValueError("no_mask must use --mask-mode zero")

    output_dir = PROJECT_ROOT / "outputs" / "roi_ablation" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    started_at = datetime.now().isoformat(timespec="seconds")
    started = time.perf_counter()

    scheduler, tokenizer, text_encoder, vae, unet = load_diffusion_models(config["pretrained_model_name_or_path"])
    module_args = SimpleNamespace(condition_dim=config["condition_dim"])
    image_proj, adapter_modules = setup_ip_adapter_modules(module_args, unet, num_tokens=checkpoint["num_parcels"])
    brain_adapter = BrainIPAdapter(unet, image_proj, adapter_modules)
    brain_adapter.image_proj_model.load_state_dict(checkpoint["image_proj"], strict=True)
    brain_adapter.adapter_modules.load_state_dict(checkpoint["ip_adapter"], strict=True)
    guidance = GuidanceGenerator(
        num_parcels=checkpoint["num_parcels"], max_voxels=checkpoint["max_voxels"],
        num_decoder_queries=config["num_decoder_queries"], output_dim=config["condition_dim"],
        sub_approach=config["sub_approach"],
    )
    guidance.load_state_dict(checkpoint["guidance_generator"], strict=True)
    for module in (text_encoder, vae, unet, brain_adapter, guidance):
        module.to(device=device, dtype=dtype).eval()

    test_dataset = make_dataset(tokenizer, checkpoint, "test", topk=checkpoint["num_parcels"] // 2)
    if args.start_idx + args.num_samples > len(test_dataset):
        raise ValueError("Requested samples exceed test dataset length")
    mean_tokens = None
    if args.mask_mode == "mean":
        if args.mean_token_path is None:
            raise ValueError("--mean-token-path is required for mean masking")
        train_dataset = make_dataset(tokenizer, checkpoint, "train", topk=checkpoint["num_parcels"] // 2)
        mean_tokens = compute_mean_parcel_tokens(
            guidance, train_dataset, device, dtype, args.mean_batch_size, args.mean_token_path
        )

    models = {
        "tokenizer": tokenizer, "text_encoder": text_encoder, "vae": vae,
        "noise_scheduler": scheduler, "brain_adapter": brain_adapter,
        "device": device, "weight_dtype": dtype,
    }
    indices = list(range(args.start_idx, args.start_idx + args.num_samples))
    loader = DataLoader(Subset(test_dataset, indices), batch_size=1, shuffle=False, num_workers=0)
    rows, grid_rows, intervention_audits = [], [], []
    with torch.no_grad():
        for dataset_index, batch in zip(indices, tqdm(loader, desc=args.run_name)):
            brain_data = torch.cat(
                [batch["brain_lh_f"].to(device=device, dtype=dtype), batch["brain_rh_f"].to(device=device, dtype=dtype)], dim=1
            )
            intervention_mode = "none" if args.mask_group == "no_mask" else args.mask_mode
            condition_tokens, _, audit = forward_with_parcel_intervention(
                guidance,
                brain_data,
                expected_num_parcels=int(checkpoint["num_parcels"]),
                indices=mask_indices,
                mode=intervention_mode,
                mean_parcel_tokens=mean_tokens,
            )
            intervention_audits.append({"dataset_idx": dataset_index, **audit.to_dict()})
            generator = torch.Generator(device=device).manual_seed(args.seed + dataset_index)
            init_image = torch.zeros_like(batch["img_ipadapter"].to(device=device, dtype=dtype))
            generated = run_diffusion(
                condition_tokens, init_image, models, num_predictions=1,
                noise_factor=args.noise_factor, denoising_steps=args.denoising_steps, generator=generator,
            )
            gt = to_pil(batch["img_encoder"][0])
            pred = Image.fromarray(generated[0])
            gt_path = output_dir / f"sample_{dataset_index:04d}_gt.png"
            pred_path = output_dir / f"sample_{dataset_index:04d}_pred.png"
            pair_path = output_dir / f"sample_{dataset_index:04d}_gt_pred.png"
            gt.save(gt_path)
            pred.save(pred_path)
            grid_rows.append(save_pair(gt, pred, pair_path))
            rows.append({"dataset_idx": dataset_index, "gt": str(gt_path), "pred": str(pred_path), "comparison": str(pair_path)})

    grid = Image.new("RGB", (512, 256 * len(grid_rows)), "white")
    for row_index, row in enumerate(grid_rows):
        grid.paste(row, (0, 256 * row_index))
    grid_path = output_dir / "grid_gt_pred.png"
    grid.save(grid_path)
    summary = {
        "run_name": args.run_name, "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"), "elapsed_sec": time.perf_counter() - started,
        "checkpoint": str(args.checkpoint), "checkpoint_step": int(checkpoint["step"]),
        "sub_approach": config["sub_approach"], "mask_group": args.mask_group,
        "mask_mode": args.mask_mode, "masked_token_indices": mask_indices,
        "intervention_stage": "after_parcel_mapper_before_token_mapper",
        "intervention_audits": intervention_audits,
        "num_masked_tokens": len(mask_indices), "seed": args.seed,
        "seed_strategy": "seed + dataset_idx", "num_samples": args.num_samples,
        "start_idx": args.start_idx, "denoising_steps": args.denoising_steps,
        "noise_factor": args.noise_factor, "device": str(device), "records": rows, "grid": str(grid_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
