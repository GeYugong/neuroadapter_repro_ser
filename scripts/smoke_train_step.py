#!/usr/bin/env python3
"""Run one real-data NeuroAdapter training step.

This is a smoke test, not a reproduction run. It verifies:
  NSD subject 1 data -> dataloader -> Stable Diffusion/NeuroAdapter forward -> loss -> backward.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from accelerate import Accelerator
from transformers import CLIPTokenizer

import brain_adapter.dataset as neuro_dataset
from brain_adapter.dataset import nsd_topk_parcel_dataset

# The released train script imports this name, but the current dataset.py does not define it.
# The smoke test only uses the single-subject dataset, so a placeholder is enough.
if not hasattr(neuro_dataset, "nsd_groupwise_topk_parcel_dataset"):
    neuro_dataset.nsd_groupwise_topk_parcel_dataset = None

from train_brain_adapter import (
    load_pretrained_models,
    setup_ip_adapter,
    setup_models_and_optimizer,
    setup_weight_dtype,
    process_training_batch,
)


def main() -> None:
    repo_root = Path("/public/home/mty/GeYugong/code/NeuroAdapter")
    args = SimpleNamespace(
        pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
        pretrained_ip_adapter_path=None,
        subj=1,
        subject_id=1,
        topk=4,
        train_batch_size=1,
        dataloader_num_workers=0,
        learning_rate=1e-4,
        weight_decay=0.01,
        mixed_precision="no",
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
    )

    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    torch.manual_seed(42)

    noise_scheduler, tokenizer, text_encoder, vae, unet = load_pretrained_models(args)
    dataset_args = SimpleNamespace(
        subj=args.subj,
        hemi=args.hemi,
        backbone_arch=args.backbone_arch,
        data_dir=args.data_dir,
        imgs_dir=args.imgs_dir,
        parcel_dir=args.parcel_dir,
        tokenizer=tokenizer,
        gen_size=args.gen_size,
    )
    train_dataset = nsd_topk_parcel_dataset(dataset_args, split="train", transform=None, topk=args.topk)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    image_proj_model, adapter_modules, _ = setup_ip_adapter(unet, args)
    neuro_adapter, guidance_generator, optimizer, _, _ = setup_models_and_optimizer(
        args, accelerator, train_dataset, image_proj_model, adapter_modules, unet
    )

    weight_dtype = setup_weight_dtype(accelerator)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.eval()
    text_encoder.eval()
    unet.train()
    guidance_generator.train()

    train_dataloader = accelerator.prepare(train_dataloader)
    batch = next(iter(train_dataloader))
    if batch["text_input_ids"].ndim == 3 and batch["text_input_ids"].shape[1] == 1:
        batch["text_input_ids"] = batch["text_input_ids"].squeeze(1)

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
    optimizer.step()
    optimizer.zero_grad()

    print("smoke_train_step ok")
    print("loss", float(loss.detach().cpu()))
    print("num_parcels", train_dataset.num_parcels)
    print("max_voxels", train_dataset.max_voxels)
    print("img_ipadapter", tuple(batch["img_ipadapter"].shape))
    print("brain_lh_f", tuple(batch["brain_lh_f"].shape))
    print("brain_rh_f", tuple(batch["brain_rh_f"].shape))
    print("device", str(accelerator.device))


if __name__ == "__main__":
    main()
