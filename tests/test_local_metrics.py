from pathlib import Path

from PIL import Image

from neuro_roi_causal.local_metrics import apply_region, crop_pair_by_box, person_mask


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
    pred = Image.new("RGB", (20, 20), "blue")
    gt_crop, pred_crop = crop_pair_by_box(gt, pred, (2, 3, 5, 6))
    assert gt_crop.size == pred_crop.size == (5, 6)
