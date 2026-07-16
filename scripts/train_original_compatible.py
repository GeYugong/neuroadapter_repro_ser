#!/usr/bin/env python3
"""Run the released NeuroAdapter trainer against the local NSD layout.

The released trainer hard-codes the authors' dataset paths and imports a
groupwise dataset symbol that is absent from the released ``dataset.py``.
This wrapper leaves the original training functions intact, supplies local
paths for single-subject training, and is intended to be launched by
``accelerate``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


CODE_ROOT = Path(
    os.environ.get(
        "NEUROADAPTER_CODE_ROOT",
        "/public/home/mty/GeYugong/projects/neuroadapter-iclr2026/code/NeuroAdapter",
    )
)
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import brain_adapter.dataset as neuro_dataset  # noqa: E402
from brain_adapter.dataset import nsd_topk_parcel_dataset  # noqa: E402

# The original script imports this unconditionally, although it is only used
# for multi-subject training and is not included in the released dataset.py.
if not hasattr(neuro_dataset, "nsd_groupwise_topk_parcel_dataset"):
    neuro_dataset.nsd_groupwise_topk_parcel_dataset = None

import train_brain_adapter as released_train  # noqa: E402


def setup_local_dataset(args, tokenizer):
    if args.training_subjects is not None:
        raise ValueError("The compatibility wrapper currently supports one subject only.")

    dataset_args = SimpleNamespace(
        subj=args.subject_id,
        backbone_arch="dinov2_q",
        data_dir=args.neural_data_dir,
        imgs_dir=args.stimuli_dir,
        parcel_dir=args.parcel_dir,
        hemi=None,
        tokenizer=tokenizer,
        gen_size=args.gen_img_resolution,
        num_decoder_queries=args.num_decoder_queries,
    )
    train_dataset = nsd_topk_parcel_dataset(
        dataset_args,
        split="train",
        transform=None,
        topk=args.topk,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    return train_dataset, train_dataloader


def training_loop_with_optional_resume(
    args,
    accelerator,
    neuro_adapter,
    guidance_generator,
    train_dataloader,
    optimizer,
    lr_scheduler,
    vae,
    text_encoder,
    noise_scheduler,
    weight_dtype,
    train_dataset,
):
    if args.resume_from_checkpoint is not None:
        accelerator.print(f"Restoring full Accelerate state from {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint)
        accelerator.wait_for_everyone()

    return released_training_loop(
        args,
        accelerator,
        neuro_adapter,
        guidance_generator,
        train_dataloader,
        optimizer,
        lr_scheduler,
        vae,
        text_encoder,
        noise_scheduler,
        weight_dtype,
        train_dataset,
    )


released_training_loop = released_train.training_loop


def main() -> None:
    parser = released_train.create_argument_parser()
    parser.add_argument(
        "--neural-data-dir",
        default="/public/home/mty/GeYugong/data/neuroadapter/neural_data",
    )
    parser.add_argument(
        "--stimuli-dir",
        default="/public/home/mty/GeYugong/data/nsd/stimuli",
    )
    parser.add_argument(
        "--parcel-dir",
        default="/public/home/mty/GeYugong/data/neuroadapter/parcels/schaefer",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Accelerate checkpoint directory created by an earlier compatible run.",
    )
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1 and local_rank != args.local_rank:
        args.local_rank = local_rank

    if args.learning_rate > 0:
        if args.time is None:
            from datetime import datetime

            args.time = datetime.now().strftime("%m_%d_%Y-%H_%M")
        args.output_dir = os.path.join(args.output_dir, args.time)
        os.makedirs(args.output_dir, exist_ok=True)

    released_train.setup_dataset_and_dataloader = setup_local_dataset
    released_train.training_loop = training_loop_with_optional_resume
    released_train.main(args)


if __name__ == "__main__":
    main()
