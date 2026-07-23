import json

import pytest

from neuro_roi_causal.coco_annotations import load_person_geometry


def test_load_person_geometry_uses_segmentation_area(tmp_path):
    payload = {
        "images": [
            {"id": 10, "width": 100, "height": 50},
            {"id": 11, "width": 20, "height": 20},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
        "annotations": [
            {"image_id": 10, "category_id": 1, "area": 500},
            {"image_id": 10, "category_id": 1, "area": 250},
            {"image_id": 10, "category_id": 2, "area": 1000},
        ],
    }
    path = tmp_path / "instances_train2017.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    geometry, files = load_person_geometry(tmp_path, splits=("train2017",))

    assert files == [path]
    assert geometry[("train2017", 10)]["person_area_ratio"] == pytest.approx(0.15)
    assert geometry[("train2017", 10)]["person_count"] == 2
    assert geometry[("train2017", 11)] == {
        "person_area_ratio": 0.0,
        "person_count": 0,
    }


def test_missing_coco_annotations_fail_explicitly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download the official COCO"):
        load_person_geometry(tmp_path, splits=("train2017",))
