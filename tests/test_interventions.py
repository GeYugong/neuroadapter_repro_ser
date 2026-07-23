import pytest
import torch

from neuro_roi_causal.interventions import apply_parcel_intervention


def test_no_mask_is_identical_and_returns_original_tensor():
    tokens = torch.randn(2, 4, 3)
    result, audit = apply_parcel_intervention(tokens, [], "none", expected_num_parcels=4)
    assert result is tokens
    assert torch.equal(result, tokens)
    assert audit.changed_indices == []


def test_zero_changes_only_target_parcels():
    tokens = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3) + 1
    result, audit = apply_parcel_intervention(
        tokens, [1, 3], "zero", expected_num_parcels=4
    )
    assert torch.count_nonzero(result[:, [1, 3], :]) == 0
    assert torch.equal(result[:, [0, 2], :], tokens[:, [0, 2], :])
    assert audit.changed_indices == [1, 3]
    assert audit.max_abs_delta_non_target == 0


def test_mean_changes_only_target_parcels():
    tokens = torch.randn(2, 4, 3)
    mean = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    result, audit = apply_parcel_intervention(
        tokens, [2], "mean", mean_parcel_tokens=mean, expected_num_parcels=4
    )
    assert torch.equal(result[:, 2, :], mean[2].expand(2, -1))
    assert torch.equal(result[:, [0, 1, 3], :], tokens[:, [0, 1, 3], :])
    assert audit.changed_indices == [2]


def test_dimension_and_index_guards():
    tokens = torch.randn(1, 4, 3)
    with pytest.raises(ValueError, match="checkpoint"):
        apply_parcel_intervention(tokens, [1], "zero", expected_num_parcels=5)
    with pytest.raises(IndexError):
        apply_parcel_intervention(tokens, [4], "zero", expected_num_parcels=4)

