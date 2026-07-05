#!/usr/bin/env python3
"""Small-scale NeuroAdapter decoding with brain-encoder candidate selection."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPRO_ROOT = Path(__file__).resolve().parents[1]
NEUROADAPTER = Path("/public/home/mty/GeYugong/code/NeuroAdapter")
WBE_ROOT = Path("/public/home/mty/GeYugong/tools/whole_brain_encoder")
for p in [REPRO_ROOT / "scripts", NEUROADAPTER, WBE_ROOT]:
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from decode_limited import BrainIPAdapter, load_diffusion_models, run_diffusion, setup_ip_adapter_modules, tensor_to_pil
from brain_adapter.dataset import nsd_topk_parcel_dataset
from brain_adapter.model import GuidanceGenerator


def ensure_brain_encoder_compat_data() -> Path:
    src_dir = Path("/public/home/mty/GeYugong/data/neuroadapter/neural_data")
    compat = Path("/public/home/mty/GeYugong/data/brain_encoder_compat/neural_data")
    compat.mkdir(parents=True, exist_ok=True)
    meta = np.load(src_dir / "metadata_sub-01.npy", allow_pickle=True).item()
    for hemi in ["lh", "rh"]:
        meta.setdefault(f"{hemi}_anterior_vertices", np.array([], dtype=np.int64))
        meta.setdefault(f"{hemi}_posterior_vertices", np.arange(163842, dtype=np.int64))
        meta.setdefault(f"{hemi}_rois", {})
    np.save(compat / "metadata_sub-01.npy", meta, allow_pickle=True)
    beta_link = compat / "betas_sub-01.h5"
    if not beta_link.exists():
        beta_link.symlink_to(src_dir / "betas_sub-01.h5")
    return compat


def patch_whole_brain_encoder_args(compat_data_dir: Path):
    import utils.args as args_mod
    original = args_mod.get_args_parser

    def patched_get_args_parser():
        parser = original()
        for action in parser._actions:
            if action.dest == "data_dir":
                action.default = str(compat_data_dir)
            elif action.dest == "imgs_dir":
                action.default = "/public/home/mty/GeYugong/data/nsd/stimuli"
            elif action.dest == "output_path":
                action.default = str(WBE_ROOT / "checkpoints")
            elif action.dest == "parcel_dir":
                action.default = "/public/home/mty/GeYugong/data/neuroadapter/parcels/schaefer"
        return parser

    args_mod.get_args_parser = patched_get_args_parser
    import brain_encoder_wrapper as bew
    bew.get_args_parser = patched_get_args_parser
    return bew


def build_local_brain_encoder_wrapper(num_gpus: int = 1):
    compat = ensure_brain_encoder_compat_data()
    bew = patch_whole_brain_encoder_args(compat)
    from datasets.nsd import nsd_dataset_avg
    from models.brain_encoder import brain_encoder

    class LocalBrainEncoderWrapper(bew.BrainEncoderWrapper):
        def load_model_path(self, model_path, images, device="cpu"):
            checkpoint = torch.load(model_path / "checkpoint_nonavg.pth", map_location="cpu", weights_only=False)
            pretrained_dict = checkpoint["model"]
            args = copy.deepcopy(checkpoint["args"])
            args.data_dir = str(compat)
            args.imgs_dir = "/public/home/mty/GeYugong/data/nsd/stimuli"
            args.parcel_dir = "/public/home/mty/GeYugong/data/neuroadapter/parcels/schaefer"
            args.device = device
            dataset = nsd_dataset_avg(args)
            model = brain_encoder(args, dataset)
            pretrained_dict = {k.replace("_orig_mod.", ""): v for k, v in pretrained_dict.items()}
            model_dict = model.state_dict()
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            model = model.to(device, non_blocking=True)
            model.eval()
            return model, args, dataset

    return LocalBrainEncoderWrapper(
        subj=1,
        backbone_arch="dinov2_q",
        encoder_arch="transformer",
        enc_output_layer=[1],
        runs=[1],
        num_gpus=num_gpus,
        parcel_strategy="schaefer",
    )


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def score_candidates(pred_fmri, batch, dataset) -> list[dict]:
    scores = []
    for cand_i in range(len(pred_fmri["lh"])):
        candidate_scores = {"candidate_index": cand_i}
        all_true, all_pred = [], []
        for hemi, batch_key in [("lh", "brain_lh_f"), ("rh", "brain_rh_f")]:
            true_parcels = batch[batch_key][0].detach().cpu().numpy()
            pred_full = pred_fmri[hemi][cand_i]
            hemi_true, hemi_pred = [], []
            for parcel_pos, parcel_idx in enumerate(dataset.selected_parcel_idx[hemi]):
                voxel_idx = dataset.parcels[hemi][int(parcel_idx)].detach().cpu().numpy()
                n = min(len(voxel_idx), true_parcels.shape[1])
                hemi_true.append(true_parcels[parcel_pos, :n])
                hemi_pred.append(pred_full[voxel_idx[:n]])
            hemi_true_flat = np.concatenate(hemi_true)
            hemi_pred_flat = np.concatenate(hemi_pred)
            candidate_scores[hemi] = pearson_corr(hemi_pred_flat, hemi_true_flat)
            all_true.append(hemi_true_flat)
            all_pred.append(hemi_pred_flat)
        candidate_scores["mean"] = pearson_corr(np.concatenate(all_pred), np.concatenate(all_true))
        scores.append(candidate_scores)
    return scores


def make_selection_grid(gt: Image.Image, candidates: list[Image.Image], scores: list[dict], best_idx: int) -> Image.Image:
    tile, label_h = 256, 34
    canvas = Image.new("RGB", (tile * (len(candidates) + 1), tile + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    canvas.paste(gt.resize((tile, tile)), (0, label_h))
    draw.text((8, 8), "GT", fill=(0, 0, 0))
    for i, image in enumerate(candidates):
        x = tile * (i + 1)
        canvas.paste(image.resize((tile, tile)), (x, label_h))
        if i == best_idx:
            draw.rectangle([x, label_h, x + tile - 1, label_h + tile - 1], outline=(220, 40, 40), width=5)
        prefix = "BEST " if i == best_idx else ""
        draw.text((x + 8, 8), f"{prefix}cand {i} r={scores[i]['mean']:.4f}", fill=(0, 0, 0))
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--denoising-steps", type=int, default=20)
    parser.add_argument("--noise-factor", type=float, default=4.0)
    parser.add_argument("--num-predictions", type=int, default=4)
    parser.add_argument("--mixed-precision", default="fp16", choices=["no", "fp16"])
    parser.add_argument("--brain-encoder-gpus", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"{datetime.now():%Y%m%d-%H%M%S}-be-select"
    out_dir = Path("/public/home/mty/GeYugong/outputs/neuroadapter_decode") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    start_time = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" and device.type == "cuda" else torch.float32

    print(f"[run] {run_name}")
    print(f"[out] {out_dir}")
    print(f"[checkpoint] {args.checkpoint}")
    print("[brain_encoder] loading enc_1/run_1 minimal wrapper")
    brain_encoder = build_local_brain_encoder_wrapper(num_gpus=args.brain_encoder_gpus)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_config = ckpt["config"]
    noise_scheduler, tokenizer, text_encoder, vae, unet = load_diffusion_models(train_config["pretrained_model_name_or_path"])
    image_proj_model, adapter_modules = setup_ip_adapter_modules(SimpleNamespace(condition_dim=train_config["condition_dim"]), unet, num_tokens=args.topk * 2)
    brain_adapter = BrainIPAdapter(unet, image_proj_model, adapter_modules)
    brain_adapter.image_proj_model.load_state_dict(ckpt["image_proj"], strict=True)
    brain_adapter.adapter_modules.load_state_dict(ckpt["ip_adapter"], strict=True)

    guidance_generator = GuidanceGenerator(
        num_parcels=ckpt["num_parcels"],
        max_voxels=ckpt["max_voxels"],
        num_decoder_queries=train_config["num_decoder_queries"],
        output_dim=train_config["condition_dim"],
        sub_approach=train_config["sub_approach"],
    )
    guidance_generator.load_state_dict(ckpt["guidance_generator"], strict=True)

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
    test_dataset = nsd_topk_parcel_dataset(dataset_args, split="test", transform=None, topk=args.topk, selected_parcel_idx=ckpt["selected_parcel_idx"])
    indices = list(range(args.start_idx, args.start_idx + args.num_samples))
    dataloader = DataLoader(Subset(test_dataset, indices), batch_size=1, shuffle=False, num_workers=0)

    vae.to(device, dtype=weight_dtype).eval()
    text_encoder.to(device, dtype=weight_dtype).eval()
    unet.to(device, dtype=weight_dtype).eval()
    brain_adapter.to(device, dtype=weight_dtype).eval()
    guidance_generator.to(device, dtype=weight_dtype).eval()
    models_dict = {"tokenizer": tokenizer, "text_encoder": text_encoder, "vae": vae, "noise_scheduler": noise_scheduler, "brain_adapter": brain_adapter, "device": device, "weight_dtype": weight_dtype}

    records, grids = [], []
    with torch.no_grad():
        for local_i, batch in enumerate(tqdm(dataloader, desc="decode/select samples")):
            dataset_idx = indices[local_i]
            lh = batch["brain_lh_f"].to(device, dtype=weight_dtype)
            rh = batch["brain_rh_f"].to(device, dtype=weight_dtype)
            brain_embeds, _ = guidance_generator(torch.cat([lh, rh], dim=1))
            img_init = torch.zeros_like(batch["img_ipadapter"].to(device, dtype=weight_dtype))
            pred_images = run_diffusion(brain_embeds, img_init, models_dict, num_predictions=args.num_predictions, noise_factor=args.noise_factor, denoising_steps=args.denoising_steps)
            pred_fmri = brain_encoder.forward(pred_images, use_dataloader=True)
            scores = score_candidates(pred_fmri, batch, test_dataset)
            best_idx = int(np.argmax([s["mean"] if np.isfinite(s["mean"]) else -np.inf for s in scores]))

            sample_dir = out_dir / f"sample_{dataset_idx:04d}"
            sample_dir.mkdir(exist_ok=True)
            gt = tensor_to_pil(batch["img_encoder"][0])
            gt_path = sample_dir / "gt.png"
            gt.save(gt_path)
            candidates, candidate_paths = [], []
            for cand_i, arr in enumerate(pred_images):
                img = Image.fromarray(arr)
                path = sample_dir / f"candidate_{cand_i:02d}.png"
                img.save(path)
                candidates.append(img)
                candidate_paths.append(str(path))
            grid = make_selection_grid(gt, candidates, scores, best_idx)
            grid_path = sample_dir / "selection_grid.png"
            grid.save(grid_path)
            grids.append(grid)
            records.append({"dataset_idx": dataset_idx, "gt": str(gt_path), "candidate_paths": candidate_paths, "scores": scores, "best_candidate_index": best_idx, "best_candidate_path": candidate_paths[best_idx], "selection_grid": str(grid_path)})

    combined_grid_path = out_dir / "grid_brain_encoder_selection.png"
    if grids:
        combined = Image.new("RGB", (max(g.width for g in grids), sum(g.height for g in grids)), "white")
        y = 0
        for g in grids:
            combined.paste(g, (0, y))
            y += g.height
        combined.save(combined_grid_path)

    summary = {"run_name": run_name, "started_at": started, "finished_at": datetime.now().isoformat(timespec="seconds"), "elapsed_sec": time.perf_counter() - start_time, "checkpoint": str(args.checkpoint), "checkpoint_step": ckpt["step"], "topk": args.topk, "num_samples": args.num_samples, "start_idx": args.start_idx, "denoising_steps": args.denoising_steps, "noise_factor": args.noise_factor, "num_predictions": args.num_predictions, "brain_encoder": "whole_brain_encoder dinov2_q enc_1 run_1 lh/rh", "device": str(device), "dtype": str(weight_dtype), "records": records, "grid": str(combined_grid_path)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
