"""Validated interventions on parcel-wise brain tokens."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class InterventionAudit:
    mode: str
    masked_indices: list[int]
    num_parcels: int
    changed_indices: list[int]
    masked_norm_before: list[float]
    masked_norm_after: list[float]
    max_abs_delta_non_target: float

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_indices(indices: Iterable[int], num_parcels: int) -> list[int]:
    normalized = sorted({int(index) for index in indices})
    if normalized and (normalized[0] < 0 or normalized[-1] >= num_parcels):
        raise IndexError(
            f"Parcel mask indices {normalized[0]}..{normalized[-1]} "
            f"are outside [0, {num_parcels})"
        )
    return normalized


def apply_parcel_intervention(
    parcel_tokens: torch.Tensor,
    indices: Iterable[int],
    mode: str,
    *,
    mean_parcel_tokens: torch.Tensor | None = None,
    expected_num_parcels: int | None = None,
) -> tuple[torch.Tensor, InterventionAudit]:
    """Apply an intervention while proving that non-target parcels are unchanged."""
    if parcel_tokens.ndim != 3:
        raise ValueError(f"Expected parcel tokens [B, P, D], got {tuple(parcel_tokens.shape)}")
    num_parcels = int(parcel_tokens.shape[1])
    if expected_num_parcels is not None and num_parcels != int(expected_num_parcels):
        raise ValueError(
            f"Parcel token count {num_parcels} does not match checkpoint "
            f"num_parcels={expected_num_parcels}"
        )
    if mode not in {"none", "zero", "mean"}:
        raise ValueError(f"Unsupported intervention mode: {mode}")

    normalized = normalize_indices(indices, num_parcels)
    if mode == "none" and normalized:
        raise ValueError("none intervention must not contain masked indices")
    if mode != "none" and not normalized:
        raise ValueError(f"{mode} intervention requires at least one parcel index")

    if not normalized:
        audit = InterventionAudit(
            mode="none",
            masked_indices=[],
            num_parcels=num_parcels,
            changed_indices=[],
            masked_norm_before=[],
            masked_norm_after=[],
            max_abs_delta_non_target=0.0,
        )
        return parcel_tokens, audit

    result = parcel_tokens.clone()
    before = parcel_tokens[:, normalized, :].float().norm(dim=-1).mean(dim=0)
    if mode == "zero":
        result[:, normalized, :] = 0
    else:
        if mean_parcel_tokens is None:
            raise ValueError("mean intervention requires mean_parcel_tokens")
        if mean_parcel_tokens.ndim == 2:
            mean_parcel_tokens = mean_parcel_tokens.unsqueeze(0)
        if mean_parcel_tokens.ndim != 3:
            raise ValueError("mean_parcel_tokens must have shape [P, D] or [1, P, D]")
        if mean_parcel_tokens.shape[1:] != parcel_tokens.shape[1:]:
            raise ValueError(
                "Mean parcel token shape does not match parcel tokens: "
                f"{tuple(mean_parcel_tokens.shape)} vs {tuple(parcel_tokens.shape)}"
            )
        result[:, normalized, :] = mean_parcel_tokens[:, normalized, :].to(
            device=result.device, dtype=result.dtype
        )

    delta = (result - parcel_tokens).abs()
    changed = torch.nonzero(delta.amax(dim=(0, 2)) > 0, as_tuple=False).flatten().tolist()
    non_target = torch.ones(num_parcels, dtype=torch.bool, device=delta.device)
    non_target[normalized] = False
    non_target_delta = delta[:, non_target, :]
    max_non_target = float(non_target_delta.max().item()) if non_target_delta.numel() else 0.0
    if changed != normalized or max_non_target != 0.0:
        raise RuntimeError(
            f"Intervention changed unexpected parcels: expected={normalized}, actual={changed}, "
            f"max_non_target_delta={max_non_target}"
        )
    after = result[:, normalized, :].float().norm(dim=-1).mean(dim=0)
    audit = InterventionAudit(
        mode=mode,
        masked_indices=normalized,
        num_parcels=num_parcels,
        changed_indices=changed,
        masked_norm_before=before.cpu().tolist(),
        masked_norm_after=after.cpu().tolist(),
        max_abs_delta_non_target=max_non_target,
    )
    return result, audit

