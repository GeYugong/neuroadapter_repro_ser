import pytest

from neuro_roi_causal.decoding import sample_seed, validate_paired_indices


def test_seed_is_stable_per_sample():
    assert sample_seed(12345, 17) == 12362
    assert sample_seed(12345, 17) == sample_seed(12345, 17)


def test_paired_indices_must_match():
    assert validate_paired_indices([[1, 2], [1, 2]]) == [1, 2]
    with pytest.raises(ValueError, match="identical"):
        validate_paired_indices([[1, 2], [2, 1]])

