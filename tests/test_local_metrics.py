from pathlib import Path

import pytest
from PIL import Image

from neuro_roi_causal.local_metrics import (
    apply_region,
    crop_pair_by_box,
    person_mask,
    region_pixel_consistency,
)


def test_person_polygon_mask_and_regions(tmp_path: Path):
    record = {
        "width": 10,
        "height": 10,
        "annotations": [{"segmentation": [[2, 2, 7, 2, 7, 7, 2, 7]], "bbox": [2, 2, 5, 5]}],
    }
    mask, method = person_mask(record, (10, 10))
    assert method == "coco_polygon"
    assert mask.getpixel((4, 4)) == 255
    assert mask.getpixel((0, 0)) == 0
    image = Image.new("RGB", (10, 10), "red")
    person = apply_region(image, mask, keep_mask=True)
    background = apply_region(image, mask, keep_mask=False)
    assert person.getpixel((4, 4)) == (255, 0, 0)
    assert person.getpixel((0, 0)) == (127, 127, 127)
    assert background.getpixel((4, 4)) == (127, 127, 127)


def test_face_crop_uses_same_ground_truth_box():
    gt = Image.new("RGB", (20, 20), "red")
    pred = Image.new("RGB", (40, 40), "blue")
    gt_crop, pred_crop = crop_pair_by_box(gt, pred, (2, 3, 5, 6))
    assert gt_crop.size == (5, 6)
    assert pred_crop.size == (10, 12)


def test_region_consistency_is_independent_and_masked():
    gt = Image.new("RGB", (10, 10), "black")
    pred = Image.new("RGB", (10, 10), "white")
    for x in range(2, 8):
        for y in range(2, 8):
            gt.putpixel((x, y), (x * 20, y * 20, 50))
            pred.putpixel((x, y), (x * 20, y * 20, 50))
    mask = Image.new("L", (10, 10), 0)
    for x in range(2, 8):
        for y in range(2, 8):
            mask.putpixel((x, y), 255)
    assert region_pixel_consistency(gt, pred, mask, size=10) == pytest.approx(1.0)
