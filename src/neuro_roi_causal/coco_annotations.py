"""Read the COCO person geometry needed for stimulus selection."""

from __future__ import annotations

import json
from pathlib import Path


def load_person_geometry(
    annotations_root: Path,
    splits: tuple[str, ...] = ("train2017", "val2017"),
) -> tuple[dict[tuple[str, int], dict[str, float | int]], list[Path]]:
    """Return person segmentation coverage keyed by ``(split, image_id)``."""
    geometry: dict[tuple[str, int], dict[str, float | int]] = {}
    source_files: list[Path] = []
    for split in splits:
        path = annotations_root / f"instances_{split}.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"COCO instance annotations not found: {path}. "
                "Download the official COCO 2017 train/val annotations explicitly."
            )
        source_files.append(path)
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        categories = {
            int(category["id"]): str(category["name"])
            for category in payload["categories"]
        }
        person_ids = {
            category_id for category_id, name in categories.items() if name == "person"
        }
        if len(person_ids) != 1:
            raise ValueError(f"Expected one COCO person category in {path}, got {person_ids}")
        person_id = next(iter(person_ids))

        image_area = {
            int(image["id"]): float(image["width"]) * float(image["height"])
            for image in payload["images"]
        }
        person_area: dict[int, float] = {}
        person_count: dict[int, int] = {}
        for annotation in payload["annotations"]:
            if int(annotation["category_id"]) != person_id:
                continue
            image_id = int(annotation["image_id"])
            person_area[image_id] = person_area.get(image_id, 0.0) + float(annotation["area"])
            person_count[image_id] = person_count.get(image_id, 0) + 1

        for image_id, area in image_area.items():
            geometry[(split, image_id)] = {
                "person_area_ratio": min(person_area.get(image_id, 0.0) / area, 1.0),
                "person_count": person_count.get(image_id, 0),
            }
    return geometry, source_files
