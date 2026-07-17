#!/usr/bin/env python3
"""Decode matched ROI-ablation conditions in GPU batches.

Every condition for a test image starts from the same VAE latent and diffusion
noise.  The only intended difference is the selected parcel-token intervention.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from brain_adapter.dataset import nsd_topk_parcel_dataset
from brain_adapter.model import GuidanceGenerator
from decode_limited import BrainIPAdapter, load_diffusion_models, setup_ip_adapter_modules


PROJECT_ROOT = Path("/public/home/mty/GeYugong/projects/neuroadapter-iclr2026")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--condition-spec", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--denoising-steps", type=int, default=50)
    parser.add_argument("--noise-factor", type=float, default=4.0)
    parser.add_argument("--condition-batch-size", type=int, default=8)
    parser.add_argument("--mean-token-path", type=Path)
    parser.add_argument("--mean-batch-size", type=int, default=8)
    return parser.parse_args()


def make_dataset(tokenizer, checkpoint: dict, split: str):
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
        topk=checkpoint["num_parcels"] // 2,
        selected_parcel_idx=checkpoint["selected_parcel_idx"],
    )


def compute_mean_tokens(guidance, dataset, device, dtype, batch_size, path: Path) -> torch.Tensor:
    if path.exists():
        return torch.load(path, map_location="cpu")["mean_condition_tokens"].to(device=device, dtype=dtype)
    total, count = None, 0
    for batch in tqdm(DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0), desc="mean tokens"):
        brain = torch.cat([
            batch["brain_lh_f"].to(device=device, dtype=dtype),
            batch["brain_rh_f"].to(device=device, dtype=dtype),
        ], dim=1)
        with torch.no_grad():
            tokens, _ = guidance(brain)
        summed = tokens.float().sum(0)
        total = summed if total is None else total + summed
        count += tokens.shape[0]
    if total is None:
        raise RuntimeError("No training examples available to calculate mean tokens")
    mean = total / count
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean_condition_tokens": mean.cpu(), "num_train_samples": count}, path)
    return mean.to(device=device, dtype=dtype)


def run_diffusion_conditions(condition_tokens, base_image, models, denoising_steps, noise_factor, generator):
    """Generate B conditions together with exactly matched latent/noise draws."""
    tokenizer = models["tokenizer"]
    text_encoder = models["text_encoder"]
    vae = models["vae"]
    scheduler = models["noise_scheduler"]
    brain_adapter = models["brain_adapter"]
    device = models["device"]
    dtype = models["dtype"]
    batch_size = condition_tokens.shape[0]

    input_ids = tokenizer("", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
    text_states = text_encoder(input_ids)[0].to(device=device, dtype=dtype).repeat(batch_size, 1, 1)
    try:
        base_latents = vae.encode(base_image).latent_dist.sample(generator=generator) * vae.config.scaling_factor
    except TypeError:
        base_latents = vae.encode(base_image).latent_dist.sample() * vae.config.scaling_factor
    latents = base_latents.repeat(batch_size, 1, 1, 1)
    noise = torch.randn(base_latents.shape, device=device, dtype=dtype, generator=generator)
    scheduler.set_timesteps(denoising_steps)
    latents = scheduler.add_noise(latents, noise.repeat(batch_size, 1, 1, 1), scheduler.timesteps[:1])
    for timestep in scheduler.timesteps:
        uncond, cond = brain_adapter(latents, timestep, text_states, condition_tokens)
        latents = scheduler.step(uncond + noise_factor * (cond - uncond), timestep, latents).prev_sample
    images = vae.decode(latents / vae.config.scaling_factor).sample
    images = (images / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
    return (images * 255).round().astype("uint8")


def make_pair(gt: Image.Image, pred: Image.Image, path: Path) -> Image.Image:
    canvas = Image.new("RGB", (512, 256), "white")
    canvas.paste(gt.resize((256, 256)), (0, 0))
    canvas.paste(pred.resize((256, 256)), (256, 0))
    canvas.save(path)
    return canvas


def main() -> None:
    args = parse_args()
    spec = json.loads(args.condition_spec.read_text(encoding="utf-8"))
    conditions = spec["conditions"]
    names = [item["name"] for item in conditions]
    if len(names) != len(set(names)):
        raise ValueError("Condition names must be unique")
    if args.condition_batch_size <= 0 or args.num_samples <= 0:
        raise ValueError("Batch size and sample count must be positive")
    requires_mean = any(item.get("mask_mode", "zero") == "mean" for item in conditions)
    if requires_mean and args.mean_token_path is None:
        raise ValueError("Mean-mask conditions require --mean-token-path")

    root = PROJECT_ROOT / "outputs" / "roi_ablation" / args.run_name
    root.mkdir(parents=True, exist_ok=False)
    condition_dirs = {name: root / name for name in names}
    for path in condition_dirs.values():
        path.mkdir()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    scheduler, tokenizer, text_encoder, vae, unet = load_diffusion_models(config["pretrained_model_name_or_path"])
    image_proj, adapter_modules = setup_ip_adapter_modules(SimpleNamespace(condition_dim=config["condition_dim"]), unet, num_tokens=checkpoint["num_parcels"])
    brain_adapter = BrainIPAdapter(unet, image_proj, adapter_modules)
    brain_adapter.image_proj_model.load_state_dict(checkpoint["image_proj"], strict=True)
    brain_adapter.adapter_modules.load_state_dict(checkpoint["ip_adapter"], strict=True)
    guidance = GuidanceGenerator(num_parcels=checkpoint["num_parcels"], max_voxels=checkpoint["max_voxels"], num_decoder_queries=config["num_decoder_queries"], output_dim=config["condition_dim"], sub_approach=config["sub_approach"])
    guidance.load_state_dict(checkpoint["guidance_generator"], strict=True)
    for module in (text_encoder, vae, unet, brain_adapter, guidance):
        module.to(device=device, dtype=dtype).eval()
    models = {"tokenizer": tokenizer, "text_encoder": text_encoder, "vae": vae, "noise_scheduler": scheduler, "brain_adapter": brain_adapter, "device": device, "dtype": dtype}
    test_dataset = make_dataset(tokenizer, checkpoint, "test")
    indices = list(range(args.start_idx, args.start_idx + args.num_samples))
    if indices[-1] >= len(test_dataset):
        raise ValueError("Requested test range exceeds dataset")
    mean_tokens = None
    if requires_mean:
        mean_tokens = compute_mean_tokens(guidance, make_dataset(tokenizer, checkpoint, "train"), device, dtype, args.mean_batch_size, args.mean_token_path)

    records = {name: [] for name in names}
    grids = {name: [] for name in names}
    started_at = datetime.now().isoformat(timespec="seconds")
    started = time.perf_counter()
    with torch.no_grad():
        loader = DataLoader(Subset(test_dataset, indices), batch_size=1, shuffle=False, num_workers=0)
        for dataset_idx, batch in zip(indices, tqdm(loader, desc=args.run_name)):
            brain = torch.cat([batch["brain_lh_f"].to(device=device, dtype=dtype), batch["brain_rh_f"].to(device=device, dtype=dtype)], dim=1)
            base_tokens, _ = guidance(brain)
            gt = transforms.ToPILImage()(batch["img_encoder"][0].detach().cpu().clamp(0, 1))
            gt_path = root / f"sample_{dataset_idx:04d}_gt.png"
            gt.save(gt_path)
            base_image = torch.zeros_like(batch["img_ipadapter"].to(device=device, dtype=dtype))
            for start in range(0, len(conditions), args.condition_batch_size):
                chunk = conditions[start:start + args.condition_batch_size]
                token_batch = []
                for condition in chunk:
                    token_copy = base_tokens.clone()
                    indices_to_mask = [int(value) for value in condition.get("masked_token_indices", [])]
                    if condition.get("mask_mode", "zero") == "zero":
                        token_copy[:, indices_to_mask, :] = 0
                    else:
                        token_copy[:, indices_to_mask, :] = mean_tokens[indices_to_mask].unsqueeze(0)
                    token_batch.append(token_copy)
                # Reinitialize per chunk so every condition sees the exact same
                # VAE posterior draw and diffusion noise, including across chunks.
                generator = torch.Generator(device=device).manual_seed(args.seed + dataset_idx)
                generated = run_diffusion_conditions(torch.cat(token_batch, dim=0), base_image, models, args.denoising_steps, args.noise_factor, generator)
                for condition, image in zip(chunk, generated):
                    name = condition["name"]
                    pred = Image.fromarray(image)
                    pred_path = condition_dirs[name] / f"sample_{dataset_idx:04d}_pred.png"
                    pair_path = condition_dirs[name] / f"sample_{dataset_idx:04d}_gt_pred.png"
                    pred.save(pred_path)
                    grids[name].append(make_pair(gt, pred, pair_path))
                    records[name].append({"dataset_idx": dataset_idx, "gt": str(gt_path), "pred": str(pred_path), "comparison": str(pair_path)})
    finished = datetime.now().isoformat(timespec="seconds")
    for condition in conditions:
        name = condition["name"]
        grid = Image.new("RGB", (512, 256 * len(grids[name])), "white")
        for row, image in enumerate(grids[name]):
            grid.paste(image, (0, 256 * row))
        grid_path = condition_dirs[name] / "grid_gt_pred.png"
        grid.save(grid_path)
        summary = {"run_name": args.run_name, "condition": condition, "checkpoint": str(args.checkpoint), "checkpoint_step": int(checkpoint["step"]), "sub_approach": config["sub_approach"], "seed": args.seed, "seed_strategy": "shared seed + dataset_idx; matched across all conditions", "num_samples": args.num_samples, "start_idx": args.start_idx, "denoising_steps": args.denoising_steps, "noise_factor": args.noise_factor, "condition_batch_size": args.condition_batch_size, "records": records[name], "grid": str(grid_path)}
        (condition_dirs[name] / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    root_summary = {"run_name": args.run_name, "started_at": started_at, "finished_at": finished, "elapsed_sec": time.perf_counter() - started, "condition_spec": str(args.condition_spec), "conditions": conditions, "num_samples": args.num_samples, "denoising_steps": args.denoising_steps, "condition_batch_size": args.condition_batch_size, "seed": args.seed}
    (root / "run_summary.json").write_text(json.dumps(root_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(root_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
