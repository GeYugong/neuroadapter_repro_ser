from pathlib import Path

import torch
from PIL import Image

from neuro_roi_causal.metrics import ContentAddressedCache


def write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (4, 4), color).save(path)


def test_same_image_hash_computes_embedding_once(tmp_path):
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    write_image(first, (1, 2, 3))
    second.write_bytes(first.read_bytes())
    cache = ContentAddressedCache()
    calls = []

    def compute(path):
        calls.append(path)
        return torch.tensor([1.0, 2.0])

    assert torch.equal(cache.embedding("clip", first, compute), cache.embedding("clip", second, compute))
    assert len(calls) == 1


def test_same_gt_pred_hash_pair_computes_lpips_once(tmp_path):
    gt, pred, copy = tmp_path / "gt.png", tmp_path / "pred.png", tmp_path / "copy.png"
    write_image(gt, (1, 2, 3))
    write_image(pred, (4, 5, 6))
    copy.write_bytes(pred.read_bytes())
    cache = ContentAddressedCache()
    calls = []

    def compute(left, right):
        calls.append((left, right))
        return 0.25

    assert cache.pair_value("lpips", gt, pred, compute) == 0.25
    assert cache.pair_value("lpips", gt, copy, compute) == 0.25
    assert len(calls) == 1
