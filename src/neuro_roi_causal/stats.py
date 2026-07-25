"""Deterministic E2 causal-effect statistics."""

from __future__ import annotations

import numpy as np


HIGHER_IS_BETTER = {"pixel_corr", "ssim", "clip", "dino"}


def bootstrap_ci(values: np.ndarray, seed: int, draws: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))]
    return [float(value) for value in np.quantile(samples.mean(axis=1), [0.025, 0.975])]


def sign_flip_pvalue(values: np.ndarray, seed: int, draws: int = 20000) -> float:
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (draws + 1))


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return [float(value) for value in output]


def causal_loss(metric: str, baseline: float, masked: float) -> float:
    return baseline - masked if metric in HIGHER_IS_BETTER else masked - baseline
