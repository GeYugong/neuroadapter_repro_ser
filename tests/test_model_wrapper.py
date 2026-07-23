import torch
from torch import nn

from neuro_roi_causal.model_wrapper import forward_with_parcel_intervention


class FakeGuidance(nn.Module):
    def __init__(self, transformer_decoder: bool):
        super().__init__()
        self.parcel_mapper = nn.Linear(2, 3, bias=False)
        self.sub_approach = "transformer_decoder" if transformer_decoder else "parcel"
        if transformer_decoder:
            self.token_mapper = nn.Linear(3, 2, bias=False)


def test_transformer_decoder_masks_before_token_mapper():
    guidance = FakeGuidance(transformer_decoder=True)
    brain = torch.randn(1, 4, 2)
    condition, parcels, audit = forward_with_parcel_intervention(
        guidance, brain, expected_num_parcels=4, indices=[1], mode="zero"
    )
    assert parcels.shape == (1, 4, 3)
    assert condition.shape == (1, 4, 2)
    assert torch.count_nonzero(parcels[:, 1, :]) == 0
    assert audit.changed_indices == [1]


def test_no_token_mapper_returns_intervened_parcels():
    guidance = FakeGuidance(transformer_decoder=False)
    brain = torch.randn(1, 4, 2)
    condition, parcels, _ = forward_with_parcel_intervention(
        guidance, brain, expected_num_parcels=4, indices=[2], mode="zero"
    )
    assert condition is parcels
    assert condition.shape == (1, 4, 3)

