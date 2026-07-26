"""Explicit local-region assets for E3 evaluation without network downloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


NEAREST = getattr(Image, "Resampling", Image).NEAREST
BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
HAAR_SCALE_FACTOR = 1.1
HAAR_MIN_NEIGHBORS = 5
HAAR_MIN_SIZE = (24, 24)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return mask.resize(output_size, NEAREST), method


def face_detector_backend(
    cascade_path: Path,
    *,
    require_opencv: bool = False,
) -> str:
    if not cascade_path.is_file():
        raise FileNotFoundError(f"Required Haar cascade is missing: {cascade_path}")
    import cv2

    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "cvtColor"):
        return "opencv_haar_e1"
    if require_opencv:
        raise RuntimeError(
            "pilot/formal evaluation requires OpenCV Haar, but cv2 does not "
            "provide CascadeClassifier and cvtColor"
        )
    from skimage import data

    fallback = Path(data.lbp_frontal_face_cascade_filename())
    if not fallback.is_file():
        raise RuntimeError(
            "OpenCV Haar API is unavailable and the bundled scikit-image LBP "
            "fallback is missing; refusing to download a detector"
        )
    return "skimage_bundled_lbp_smoke_fallback"


def detect_largest_face(
    image: Image.Image,
    cascade_path: Path,
    *,
    require_opencv: bool = False,
) -> tuple[int, int, int, int] | None:
    backend = face_detector_backend(
        cascade_path,
        require_opencv=require_opencv,
    )
    rgb = np.asarray(image.convert("RGB"))
    if backend == "opencv_haar_e1":
        import cv2

        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=HAAR_MIN_SIZE,
        )
        if len(faces) == 0:
            return None
        x, y, width, height = max(
            faces, key=lambda item: int(item[2]) * int(item[3])
        )
        return int(x), int(y), int(width), int(height)

    from skimage import data
    from skimage.feature import Cascade

    detector = Cascade(data.lbp_frontal_face_cascade_filename())
    detections = detector.detect_multi_scale(
        img=rgb,
        scale_factor=1.2,
        step_ratio=1,
        min_size=(24, 24),
        max_size=image.size,
    )
    if not detections:
        return None
    selected = max(
        detections, key=lambda item: int(item["width"]) * int(item["height"])
    )
    return (
        int(selected["c"]),
        int(selected["r"]),
        int(selected["width"]),
        int(selected["height"]),
    )


def crop_pair_by_box(
    gt: Image.Image, pred: Image.Image, box: tuple[int, int, int, int]
) -> tuple[Image.Image, Image.Image]:
    x, y, width, height = box
    gt_bounds = (x, y, x + width, y + height)
    scale_x = pred.width / gt.width
    scale_y = pred.height / gt.height
    pred_bounds = (
        round(x * scale_x),
        round(y * scale_y),
        round((x + width) * scale_x),
        round((y + height) * scale_y),
    )
    return gt.crop(gt_bounds), pred.crop(pred_bounds)


def apply_region(
    image: Image.Image, mask: Image.Image, *, keep_mask: bool
) -> Image.Image:
    rgb = image.convert("RGB")
    if mask.size != rgb.size:
        mask = mask.resize(rgb.size, NEAREST)
    selected = mask if keep_mask else Image.eval(mask, lambda value: 255 - value)
    neutral = Image.new("RGB", rgb.size, (127, 127, 127))
    return Image.composite(rgb, neutral, selected)


def region_pixel_consistency(
    gt: Image.Image,
    pred: Image.Image,
    mask: Image.Image,
    *,
    size: int = 256,
) -> float:
    """Return RGB correlation inside a ground-truth region mask."""
    if size <= 0:
        raise ValueError("Region consistency size must be positive")
    gt_array = np.asarray(
        gt.convert("RGB").resize((size, size), BILINEAR),
        dtype=np.float64,
    )
    pred_array = np.asarray(
        pred.convert("RGB").resize((size, size), BILINEAR),
        dtype=np.float64,
    )
    mask_array = np.asarray(mask.resize((size, size), NEAREST), dtype=np.uint8) > 0
    if not mask_array.any():
        raise ValueError("Person region mask is empty")
    x = gt_array[mask_array].reshape(-1)
    y = pred_array[mask_array].reshape(-1)
    x -= x.mean()
    y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return 0.0 if denominator == 0 else float(np.dot(x, y) / denominator)
