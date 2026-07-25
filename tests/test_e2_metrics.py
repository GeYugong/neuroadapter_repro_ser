import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_e2_pilot.py"
SPEC = importlib.util.spec_from_file_location("evaluate_e2_pilot", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_causal_loss_uses_metric_direction():
    assert MODULE.causal_loss("pixel_corr", 0.5, 0.4) == pytest.approx(0.1)
    assert MODULE.causal_loss("ssim", 0.5, 0.4) == pytest.approx(0.1)
    assert MODULE.causal_loss("clip", 0.5, 0.4) == pytest.approx(0.1)
    assert MODULE.causal_loss("dino", 0.5, 0.4) == pytest.approx(0.1)
    assert MODULE.causal_loss("lpips", 0.4, 0.5) == pytest.approx(0.1)


def test_benjamini_hochberg_is_monotonic_in_rank():
    qvalues = MODULE.benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert qvalues == pytest.approx([0.04, 0.0533333333, 0.0533333333, 0.20])
