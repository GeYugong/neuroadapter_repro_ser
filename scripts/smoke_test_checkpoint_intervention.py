#!/usr/bin/env python3
"""Smoke-test parcel intervention against a real NeuroAdapter checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "src"))

from neuro_roi_causal.model_wrapper import forward_with_parcel_intervention  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.upstream_root))
    from brain_adapter.model import GuidanceGenerator

    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    num_parcels = int(checkpoint["num_parcels"])
    max_voxels = int(checkpoint["max_voxels"])
    guidance = GuidanceGenerator(
        num_parcels=num_parcels,
        max_voxels=max_voxels,
        num_decoder_queries=int(config["num_decoder_queries"]),
        output_dim=int(config["condition_dim"]),
        sub_approach=str(config["sub_approach"]),
    )
    guidance.load_state_dict(checkpoint["guidance_generator"], strict=True)
    guidance.eval()

    fmri = torch.zeros(1, num_parcels, max_voxels)
    with torch.inference_mode():
        reference_condition, reference_parcels = guidance(fmri)
        no_mask_condition, no_mask_parcels, no_mask_audit = (
            forward_with_parcel_intervention(
                guidance,
                fmri,
                expected_num_parcels=num_parcels,
                mode="none",
            )
        )
        zero_condition, zero_parcels, zero_audit = forward_with_parcel_intervention(
            guidance,
            fmri,
            expected_num_parcels=num_parcels,
            indices=[0],
            mode="zero",
        )

    assert torch.equal(no_mask_parcels, reference_parcels)
    assert torch.equal(no_mask_condition, reference_condition)
    assert no_mask_audit.changed_indices == []
    assert torch.count_nonzero(zero_parcels[:, 0, :]) == 0
    assert torch.equal(zero_parcels[:, 1:, :], reference_parcels[:, 1:, :])
    assert zero_audit.changed_indices == [0]
    assert zero_audit.max_abs_delta_non_target == 0.0

    result = {
        "checkpoint_step": int(checkpoint["step"]),
        "sub_approach": str(config["sub_approach"]),
        "fmri_shape": list(fmri.shape),
        "parcel_token_shape": list(reference_parcels.shape),
        "condition_token_shape": list(reference_condition.shape),
        "zero_condition_shape": list(zero_condition.shape),
        "no_mask_matches_reference": True,
        "zero_changed_indices": zero_audit.changed_indices,
        "zero_max_abs_delta_non_target": zero_audit.max_abs_delta_non_target,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
