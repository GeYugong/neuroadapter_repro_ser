from pathlib import Path

import pytest
import torch

from neuro_roi_causal.mean_cache import build_cache, load_validated_cache, validate_cache


def make_cache(tokens=None):
    return build_cache(
        torch.ones(200, 768) if tokens is None else tokens,
        subject=1,
        num_train_samples=9000,
        checkpoint_path=Path("/checkpoint.pt"),
        checkpoint_sha256="abc",
        checkpoint_step=100000,
        selected_parcel_idx={"lh": list(range(100)), "rh": list(range(100))},
        sub_approach="linear_projection",
        git_commit="deadbeef",
        created_at="2026-07-26T00:00:00+00:00",
    )


def test_mean_cache_shape_and_finite_values():
    cache = make_cache()
    validate_cache(
        cache,
        checkpoint_sha256="abc",
        selected_parcel_idx={"lh": list(range(100)), "rh": list(range(100))},
    )
    assert cache["shape"] == [200, 768]
    assert not cache["contains_nan"]
    assert not cache["contains_inf"]


def test_legacy_cache_is_rejected(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"mean_parcel_tokens": torch.ones(200, 768)}, path)
    with pytest.raises(ValueError, match="Legacy"):
        load_validated_cache(path)


def test_checkpoint_and_selected_indices_mismatch_are_rejected():
    cache = make_cache()
    with pytest.raises(ValueError, match="checkpoint"):
        validate_cache(cache, checkpoint_sha256="different")
    with pytest.raises(ValueError, match="indices"):
        validate_cache(
            cache,
            selected_parcel_idx={"lh": list(range(99)) + [101], "rh": list(range(100))},
        )


def test_nan_inf_and_all_zero_are_rejected():
    for tokens, message in (
        (torch.zeros(200, 768), "all zero"),
        (torch.full((200, 768), float("nan")), "NaN"),
        (torch.full((200, 768), float("inf")), "Inf"),
    ):
        cache = make_cache(tokens)
        with pytest.raises(ValueError, match=message):
            validate_cache(cache)
