#!/usr/bin/env python3
"""Decode a small number of NSD test samples from a limited NeuroAdapter checkpoint."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from brain_adapter.dataset import nsd_topk_parcel_dataset
from brain_adapter.model import GuidanceGenerator

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from brain_adapter.ip_adapter.ip_adapter import ImageProjModel
from brain_adapter.ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from brain_adapter.ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from brain_adapter.ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor


class BrainIPAdapter(torch.nn.Module):
    def __init__(self, unet, image_proj_model, adapter_modules):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, brain_embeds):
        brain_embeds_cond = brain_embeds
        brain_embeds_uncond = torch.zeros_like(brain_embeds)
        ip_tokens_cond = self.image_proj_model(brain_embeds_cond)
        ip_tokens_uncond = self.image_proj_model(brain_embeds_uncond)
        encoder_hidden_states_cond = torch.cat([encoder_hidden_states, ip_tokens_cond], dim=1)
        encoder_hidden_states_uncond = torch.cat([encoder_hidden_states, ip_tokens_uncond], dim=1)
        combined_encoder_hidden_states = torch.cat([encoder_hidden_states_uncond, encoder_hidden_states_cond], dim=0)
        combined_noisy_latents = torch.cat([noisy_latents, noisy_latents], dim=0)
        combined_timesteps = torch.cat([timesteps, timesteps], dim=0) if isinstance(timesteps, torch.Tensor) and timesteps.dim() > 0 else timesteps
        combined_noise_pred = self.unet(combined_noisy_latents, combined_timesteps, combined_encoder_hidden_states).sample
        return combined_noise_pred.chunk(2)


def load_diffusion_models(model_path="runwayml/stable-diffusion-v1-5"):
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
    return noise_scheduler, tokenizer, text_encoder, vae, unet


def setup_ip_adapter_modules(args, unet, num_tokens=200):
    image_proj_model = ImageProjModel(
        cross_attention_dim=unet.config.cross_attention_dim,
        clip_embeddings_dim=args.condition_dim,
        clip_extra_context_tokens=num_tokens,
    )
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                scale=1.0,
                num_tokens=num_tokens,
            )
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    return image_proj_model, torch.nn.ModuleList(unet.attn_processors.values())


def run_diffusion(brain_embeds, img_ip, models_dict, num_predictions=1, noise_factor=4.0, denoising_steps=20, generator=None):
    tokenizer = models_dict["tokenizer"]
    text_encoder = models_dict["text_encoder"]
    vae = models_dict["vae"]
    noise_scheduler = models_dict["noise_scheduler"]
    brain_adapter = models_dict["brain_adapter"]
    device = models_dict["device"]
    weight_dtype = models_dict["weight_dtype"]

    text_input_ids = tokenizer("", max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
    encoder_hidden_states = text_encoder(text_input_ids)[0].to(device, dtype=weight_dtype).repeat(num_predictions, 1, 1)
    noise_scheduler.set_timesteps(denoising_steps)
    timesteps = noise_scheduler.timesteps

    try:
        init_latents = vae.encode(img_ip).latent_dist.sample(generator=generator) * vae.config.scaling_factor
    except TypeError:
        init_latents = vae.encode(img_ip).latent_dist.sample() * vae.config.scaling_factor
    latents = []
    for _ in range(num_predictions):
        noise = torch.randn(init_latents.shape, device=device, dtype=weight_dtype, generator=generator)
        latents.append(noise_scheduler.add_noise(init_latents, noise, timesteps[:1]))
    latents = torch.cat(latents, dim=0)

    for t in timesteps:
        noise_pred_uncond, noise_pred_cond = brain_adapter(latents, t, encoder_hidden_states, brain_embeds.repeat(num_predictions, 1, 1))
        noise_pred = noise_pred_uncond + noise_factor * (noise_pred_cond - noise_pred_uncond)
        latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / vae.config.scaling_factor
    pred_images = vae.decode(latents).sample
    pred_images = (pred_images / 2 + 0.5).clamp(0, 1)
    pred_images = pred_images.cpu().permute(0, 2, 3, 1).numpy()
    return (pred_images * 255).round().astype("uint8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--denoising-steps", type=int, default=20)
    parser.add_argument("--noise-factor", type=float, default=4.0)
    parser.add_argument("--num-predictions", type=int, default=1)
    parser.add_argument("--mixed-precision", default="fp16", choices=["no", "fp16"])
    return parser.parse_args()


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    img_tensor = img_tensor.detach().cpu().clamp(0, 1)
    return transforms.ToPILImage()(img_tensor)


def make_grid_row(gt: Image.Image, pred: Image.Image) -> Image.Image:
    gt = gt.resize((256, 256))
    pred = pred.resize((256, 256))
    canvas = Image.new("RGB", (512, 256), "white")
    canvas.paste(gt, (0, 0))
    canvas.paste(pred, (256, 0))
    return canvas


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"{datetime.now():%Y%m%d-%H%M%S}-decode{args.num_samples}"
    out_dir = Path("/public/home/mty/GeYugong/outputs/neuroadapter_decode") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" and device.type == "cuda" else torch.float32
    started = datetime.now().isoformat(timespec="seconds")
    start_time = time.perf_counter()

    print(f"[run] {run_name}")
    print(f"[out] {out_dir}")
    print(f"[checkpoint] {args.checkpoint}")
    print(f"[device] {device}, dtype={weight_dtype}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    train_config = ckpt["config"]

    noise_scheduler, tokenizer, text_encoder, vae, unet = load_diffusion_models(train_config["pretrained_model_name_or_path"])
    decode_args = SimpleNamespace(condition_dim=train_config["condition_dim"])
    image_proj_model, adapter_modules = setup_ip_adapter_modules(decode_args, unet, num_tokens=args.topk * 2)

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
    test_dataset = nsd_topk_parcel_dataset(dataset_args, split="test", transform=None, topk=args.topk)
    indices = list(range(args.start_idx, args.start_idx + args.num_samples))
    dataloader = DataLoader(Subset(test_dataset, indices), batch_size=1, shuffle=False, num_workers=0)

    vae.to(device, dtype=weight_dtype).eval()
    text_encoder.to(device, dtype=weight_dtype).eval()
    unet.to(device, dtype=weight_dtype).eval()
    brain_adapter.to(device, dtype=weight_dtype).eval()
    guidance_generator.to(device, dtype=weight_dtype).eval()

    models_dict = {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "noise_scheduler": noise_scheduler,
        "brain_adapter": brain_adapter,
        "device": device,
        "weight_dtype": weight_dtype,
    }

    rows = []
    records = []
    with torch.no_grad():
        for local_i, batch in enumerate(tqdm(dataloader, desc="decode samples")):
            dataset_idx = indices[local_i]
            lh = batch["brain_lh_f"].to(device, dtype=weight_dtype)
            rh = batch["brain_rh_f"].to(device, dtype=weight_dtype)
            brain_data = torch.cat([lh, rh], dim=1)
            brain_embeds, _ = guidance_generator(brain_data)
            img_init = torch.zeros_like(batch["img_ipadapter"].to(device, dtype=weight_dtype))

            pred_images = run_diffusion(
                brain_embeds,
                img_init,
                models_dict,
                num_predictions=args.num_predictions,
                noise_factor=args.noise_factor,
                denoising_steps=args.denoising_steps,
            )
            pred = Image.fromarray(pred_images[0])
            gt = tensor_to_pil(batch["img_encoder"][0])

            gt_path = out_dir / f"sample_{dataset_idx:04d}_gt.png"
            pred_path = out_dir / f"sample_{dataset_idx:04d}_pred.png"
            row_path = out_dir / f"sample_{dataset_idx:04d}_gt_pred.png"
            gt.save(gt_path)
            pred.save(pred_path)
            row = make_grid_row(gt, pred)
            row.save(row_path)
            rows.append(row)
            records.append({"dataset_idx": dataset_idx, "gt": str(gt_path), "pred": str(pred_path), "comparison": str(row_path)})

    if rows:
        grid = Image.new("RGB", (512, 256 * len(rows)), "white")
        for i, row in enumerate(rows):
            grid.paste(row, (0, 256 * i))
        grid.save(out_dir / "grid_gt_pred.png")

    summary = {
        "run_name": run_name,
        "started_at": started,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": time.perf_counter() - start_time,
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": ckpt["step"],
        "topk": args.topk,
        "num_samples": args.num_samples,
        "start_idx": args.start_idx,
        "denoising_steps": args.denoising_steps,
        "noise_factor": args.noise_factor,
        "num_predictions": args.num_predictions,
        "device": str(device),
        "dtype": str(weight_dtype),
        "records": records,
        "grid": str(out_dir / "grid_gt_pred.png"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
