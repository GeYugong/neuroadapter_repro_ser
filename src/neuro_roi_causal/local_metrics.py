"""Explicit local-region assets for E3 evaluation without network downloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def load_coco_person_annotations(
    annotations_root: Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for split in ("train2017", "val2017"):
        path = annotations_root / f"instances_{split}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Required COCO annotations are missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        person_ids = {
            int(item["id"]) for item in payload["categories"] if item["name"] == "person"
        }
        images = {int(item["id"]): item for item in payload["images"]}
        annotations: dict[int, list[dict[str, Any]]] = {}
        for item in payload["annotations"]:
            if int(item["category_id"]) in person_ids:
                annotations.setdefault(int(item["image_id"]), []).append(item)
        for image_id, image in images.items():
            result[(split, image_id)] = {
                "width": int(image["width"]),
                "height": int(image["height"]),
                "annotations": annotations.get(image_id, []),
            }
    return result


def person_mask(record: dict[str, Any], output_size: tuple[int, int]) -> tuple[Image.Image, str]:
    width, height = int(record["width"]), int(record["height"])
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    method = "coco_polygon"
    for annotation in record["annotations"]:
        segmentation = annotation.get("segmentation")
        if isinstance(segmentation, list):
            for polygon in segmentation:
                if len(polygon) >= 6:
                    draw.polygon(
                        [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)],
                        fill=255,
                    )
        else:
            method = "coco_bbox_fallback_for_rle"
            x, y, box_width, box_height = map(float, annotation["bbox"])
            draw.rectangle((x, y, x + box_width, y + box_height), fill=255)
    return mask.resize(output_size, Image.Resampling.NEAREST), method


def detect_largest_face(image: Image.Image, cascade_path: Path) -> tuple[int, int, int, int] | None:
    if not cascade_path.is_file():
        raise FileNotFoundError(f"Required Haar cascade is missing: {cascade_path}")
    import cv2

    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
    )
    if len(faces) == 0:
        return None
    x, y, width, height = max(faces, key=lambda item: int(item[2]) * int(item[3]))
    return int(x), int(y), int(width), int(height)


def crop_pair_by_box(
    gt: Image.Image, pred: Image.Image, box: tuple[int, int, int, int]
) -> tuple[Image.Image, Image.Image]:
    x, y, width, height = box
    bounds = (x, y, x + width, y + height)
    return gt.crop(bounds), pred.crop(bounds)


def apply_region(
    image: Image.Image, mask: Image.Image, *, keep_mask: bool
) -> Image.Image:
    rgb = image.convert("RGB")
    selected = mask if keep_mask else Image.eval(mask, lambda value: 255 - value)
    neutral = Image.new("RGB", rgb.size, (127, 127, 127))
    return Image.composite(rgb, neutral, selected)

