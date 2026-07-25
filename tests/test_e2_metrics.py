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
