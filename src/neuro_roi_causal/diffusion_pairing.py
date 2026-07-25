"""Auditable helpers for sharing diffusion state across causal conditions."""

from __future__ import annotations

import hashlib

import torch


def tensor_sha256(tensor: torch.Tensor) -> str:
    contiguous = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def shared_state_audit(base_latents: torch.Tensor, noise: torch.Tensor, seed: int) -> dict:
    if base_latents.shape != noise.shape:
        raise ValueError("Initial latent and noise shapes must match")
    return {
        "sample_seed": int(seed),
        "initial_latent_sha256": tensor_sha256(base_latents),
        "diffusion_noise_sha256": tensor_sha256(noise),
        "shape": list(base_latents.shape),
        "dtype": str(base_latents.dtype),
    }
