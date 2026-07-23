"""Deterministic pairing helpers shared by causal decoding scripts."""

from __future__ import annotations

from collections.abc import Iterable


def sample_seed(base_seed: int, dataset_index: int) -> int:
    if base_seed < 0 or dataset_index < 0:
        raise ValueError("Seeds and dataset indices must be non-negative")
    return int(base_seed) + int(dataset_index)


def validate_paired_indices(condition_indices: Iterable[Iterable[int]]) -> list[int]:
    groups = [list(map(int, indices)) for indices in condition_indices]
    if not groups:
        raise ValueError("At least one condition is required")
    reference = groups[0]
    if any(indices != reference for indices in groups[1:]):
        raise ValueError("All paired conditions must use identical dataset indices")
    return reference

