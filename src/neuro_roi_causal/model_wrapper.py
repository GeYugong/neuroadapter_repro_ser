"""Read-only integration with NeuroAdapter's GuidanceGenerator."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from .interventions import InterventionAudit, apply_parcel_intervention


def parcel_tokens_to_condition(guidance_generator, parcel_tokens: torch.Tensor) -> torch.Tensor:
    if guidance_generator.sub_approach == "transformer_decoder":
        if not hasattr(guidance_generator, "token_mapper"):
            raise AttributeError("transformer_decoder guidance is missing token_mapper")
        return guidance_generator.token_mapper(parcel_tokens)
    return parcel_tokens


def map_fmri_to_parcel_tokens(
    guidance_generator,
    fmri_data: torch.Tensor,
    expected_num_parcels: int,
) -> torch.Tensor:
    parcel_tokens = guidance_generator.parcel_mapper(fmri_data)
    if parcel_tokens.ndim != 3 or parcel_tokens.shape[1] != expected_num_parcels:
        raise ValueError(
            f"ParcelMapper returned {tuple(parcel_tokens.shape)}; "
            f"expected [B, {expected_num_parcels}, D]"
        )
    return parcel_tokens


def forward_with_parcel_intervention(
    guidance_generator,
    fmri_data: torch.Tensor,
    *,
    expected_num_parcels: int,
    indices: Iterable[int] = (),
    mode: str = "none",
    mean_parcel_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, InterventionAudit]:
    parcel_tokens = map_fmri_to_parcel_tokens(
        guidance_generator, fmri_data, expected_num_parcels
    )
    intervened, audit = apply_parcel_intervention(
        parcel_tokens,
        indices,
        mode,
        mean_parcel_tokens=mean_parcel_tokens,
        expected_num_parcels=expected_num_parcels,
    )
    condition_tokens = parcel_tokens_to_condition(guidance_generator, intervened)
    return condition_tokens, intervened, audit

