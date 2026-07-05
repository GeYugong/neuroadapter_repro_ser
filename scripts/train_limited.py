#!/usr/bin/env python3
"""Run a limited NeuroAdapter training job on prepared NSD subject 1 data."""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from accelerate import Accelerator

import brain_adapter.dataset as neuro_dataset
from brain_adapter.dataset import nsd_topk_parcel_dataset

# The released train_brain_adapter.py imports this symbol, but the current dataset.py
# does not define it. Single-subject training does not use it.
if not hasattr(neuro_dataset, "nsd_groupwise_topk_parcel_dataset"):
    neuro_dataset.nsd_groupwise_topk_parcel_dataset = None

from train_brain_adapter import (  # noqa: E402
    load_pretrained_models,
    setup_ip_adapter,
    setup_models_and_optimizer,
    setup_weight_dtype,
    process_training_batch,
)


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-steps", type=int, default=50, help="Number of additional optimizer steps to run.")
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Optional checkpoint-step-*.pt to resume model weights from.")
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--mixed-precision", default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--cuda-device", default=None, help="Informational only; set CUDA_VISIBLE_DEVICES outside if needed.")
    return parser.parse_args()


def save_checkpoint(out_dir: Path, step: int, args_obj, neuro_adapter, guidance_generator, train_dataset, losses) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ip_adapter = unwrap(neuro_adapter)
    guidance = unwrap(guidance_generator)
    ckpt_path = out_dir / f"checkpoint-step-{step:04d}.pt"
    torch.save(
        {
            "step": step,
            "config": vars(args_obj),
            "image_proj": ip_adapter.image_proj_model.state_dict(),
            "ip_adapter": ip_adapter.adapter_modules.state_dict(),
            "guidance_generator": guidance.state_dict(),
            "selected_parcel_idx": {
                hemi: train_dataset.selected_parcel_idx[hemi].tolist()
                for hemi in ["lh", "rh"]
            },
            "num_parcels": train_dataset.num_parcels,
            "max_voxels": train_dataset.max_voxels,
            "losses": losses,
        },
        ckpt_path,
    )
    return ckpt_path


def main() -> None:
    cli = parse_args()
    run_name = cli.run_name or f"{datetime.now():%Y%m%d-%H%M%S}-topk{cli.topk}-steps{cli.max_steps}"
    output_root = Path("/public/home/mty/GeYugong/outputs/neuroadapter") / run_name
    output_root.mkdir(parents=True, exist_ok=True)

    train_args = SimpleNamespace(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        pretrained_ip_adapter_path=None,
        subj=1,
        subject_id=1,
        topk=cli.topk,
        train_batch_size=cli.batch_size,
        dataloader_num_workers=cli.num_workers,
        learning_rate=cli.lr,
        weight_decay=cli.weight_decay,
        mixed_precision=cli.mixed_precision,
        gradient_accumulation_steps=1,
        clip_max_norm=1.0,
        condition_dim=768,
        num_decoder_queries=50,
        sub_approach="linear_projection",
        backbone_arch="dinov2_q",
        data_dir="/public/home/mty/GeYugong/data/neuroadapter/neural_data",
        imgs_dir="/public/home/mty/GeYugong/data/nsd/stimuli",
        parcel_dir="/public/home/mty/GeYugong/data/neuroadapter/parcels/schaefer",
        hemi=None,
        gen_size=512,
        init_checkpoint=str(cli.init_checkpoint) if cli.init_checkpoint else None,
        initial_step=0,
    )

    accelerator = Accelerator(mixed_precision=train_args.mixed_precision)
    torch.manual_seed(42)

    started_at = datetime.now().isoformat(timespec="seconds")
    print(f"[run] {run_name}")
    print(f"[output] {output_root}")
    print(f"[started_at] {started_at}")

    noise_scheduler, tokenizer, text_encoder, vae, unet = load_pretrained_models(train_args)
    dataset_args = SimpleNamespace(
        subj=train_args.subj,
        hemi=train_args.hemi,
        backbone_arch=train_args.backbone_arch,
        data_dir=train_args.data_dir,
        imgs_dir=train_args.imgs_dir,
        parcel_dir=train_args.parcel_dir,
        tokenizer=tokenizer,
        gen_size=train_args.gen_size,
    )
    train_dataset = nsd_topk_parcel_dataset(dataset_args, split="train", transform=None, topk=train_args.topk)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=train_args.train_batch_size,
        num_workers=train_args.dataloader_num_workers,
    )

    image_proj_model, adapter_modules, _num_fmri_tokens = setup_ip_adapter(unet, train_args)
    neuro_adapter, guidance_generator, optimizer, _, _ = setup_models_and_optimizer(
        train_args, accelerator, train_dataset, image_proj_model, adapter_modules, unet
    )

    if cli.init_checkpoint is not None:
        init_ckpt = torch.load(cli.init_checkpoint, map_location="cpu")
        ckpt_num_parcels = int(init_ckpt["num_parcels"])
        ckpt_max_voxels = int(init_ckpt["max_voxels"])
        if ckpt_num_parcels != train_dataset.num_parcels:
            raise ValueError(f"Checkpoint num_parcels {ckpt_num_parcels} != dataset {train_dataset.num_parcels}")
        if ckpt_max_voxels != train_dataset.max_voxels:
            raise ValueError(f"Checkpoint max_voxels {ckpt_max_voxels} != dataset {train_dataset.max_voxels}")
        unwrap(neuro_adapter).image_proj_model.load_state_dict(init_ckpt["image_proj"], strict=True)
        unwrap(neuro_adapter).adapter_modules.load_state_dict(init_ckpt["ip_adapter"], strict=True)
        unwrap(guidance_generator).load_state_dict(init_ckpt["guidance_generator"], strict=True)
        train_args.initial_step = int(init_ckpt["step"])
        print(f"[resume] loaded {cli.init_checkpoint} at step {train_args.initial_step}")

    weight_dtype = setup_weight_dtype(accelerator)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    text_encoder.eval()
    unet.train()

    config = {
        "run_name": run_name,
        "started_at": started_at,
        "output_root": str(output_root),
        "train_args": vars(train_args),
        "dataset_len": len(train_dataset),
        "num_parcels": train_dataset.num_parcels,
        "max_voxels": train_dataset.max_voxels,
        "device": str(accelerator.device),
        "torch_version": torch.__version__,
    }
    (output_root / "config.json").write_text(json.dumps(config, indent=2))

    csv_path = output_root / "losses.csv"
    losses = []
    start = time.perf_counter()
    data_iter = iter(train_dataloader)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss", "elapsed_sec"])
        writer.writeheader()
        progress = tqdm(range(1, cli.max_steps + 1), desc="limited train")
        for local_step in progress:
            step = train_args.initial_step + local_step
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            optimizer.zero_grad()
            loss = process_training_batch(
                batch,
                vae,
                noise_scheduler,
                text_encoder,
                guidance_generator,
                neuro_adapter,
                weight_dtype,
                accelerator,
                train_dataset,
            )
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(
                list(unwrap(neuro_adapter).image_proj_model.parameters())
                + list(unwrap(neuro_adapter).adapter_modules.parameters())
                + list(unwrap(guidance_generator).parameters()),
                train_args.clip_max_norm,
            )
            optimizer.step()

            loss_value = float(loss.detach().cpu())
            elapsed = time.perf_counter() - start
            row = {"step": step, "loss": loss_value, "elapsed_sec": elapsed}
            writer.writerow(row)
            f.flush()
            losses.append(row)
            progress.set_postfix(loss=f"{loss_value:.5f}")

            if cli.save_every > 0 and local_step % cli.save_every == 0:
                ckpt_path = save_checkpoint(output_root, step, train_args, neuro_adapter, guidance_generator, train_dataset, losses)
                print(f"[checkpoint] {ckpt_path}")

    final_step = train_args.initial_step + cli.max_steps
    final_ckpt = save_checkpoint(output_root, final_step, train_args, neuro_adapter, guidance_generator, train_dataset, losses)
    finished_at = datetime.now().isoformat(timespec="seconds")
    summary = {
        **config,
        "finished_at": finished_at,
        "elapsed_sec": time.perf_counter() - start,
        "initial_step": train_args.initial_step,
        "additional_steps": cli.max_steps,
        "final_step": final_step,
        "final_checkpoint": str(final_ckpt),
        "first_loss": losses[0]["loss"] if losses else None,
        "last_loss": losses[-1]["loss"] if losses else None,
        "min_loss": min((x["loss"] for x in losses), default=None),
        "max_loss": max((x["loss"] for x in losses), default=None),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
