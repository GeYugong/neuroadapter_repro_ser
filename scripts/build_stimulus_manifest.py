#!/usr/bin/env python3
"""Build category-specific NSD Subject 1 stimulus manifests without downloads."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "src"))

from neuro_roi_causal.coco_annotations import load_person_geometry  # noqa: E402
from neuro_roi_causal.provenance import git_commit, sha256_file  # noqa: E402
from neuro_roi_causal.stimulus_categories import assign_category  # noqa: E402


CLIP_PROMPTS = {
    "Face": [
        "a close-up photograph of a human face",
        "a portrait with a clearly visible face",
        "a photograph focused on a person's face",
    ],
    "Body": [
        "a photograph of a full human body",
        "a person whose body is clearly visible",
        "a photograph focused on a standing or moving person",
    ],
    "Scene": [
        "a wide photograph of an indoor or outdoor scene",
        "a landscape, room, street, beach, or natural scene",
        "a photograph focused on the surrounding environment",
    ],
    "Word": [
        "an image containing prominent readable text",
        "a sign, menu, label, or document with words",
        "a photograph focused on written words",
    ],
    "Other": [
        "a photograph focused on an object",
        "a photograph that is not mainly a face, body, scene, or text",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--stimuli-hdf5", type=Path, required=True)
    parser.add_argument("--stim-info-csv", type=Path, required=True)
    parser.add_argument("--coco-annotations-root", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--face-cascade", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--audit-per-category", type=int, default=8)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(min(value, 20.0), -20.0)))


def detect_face_geometry(image: np.ndarray, cv2, face_detector) -> dict:
    rgb = np.asarray(image, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces, _, face_weights = face_detector.detectMultiScale3(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(24, 24),
        outputRejectLevels=True,
    )
    height, width = rgb.shape[:2]
    image_area = float(height * width)
    face_areas = [float(w * h) / image_area for (_, _, w, h) in faces]
    face_area = max(face_areas, default=0.0)
    face_confidence = max((sigmoid(float(value)) for value in face_weights), default=0.0)

    return {
        "face_confidence": face_confidence,
        "face_area_ratio": face_area,
        "face_count": len(faces),
    }


def load_clip(checkpoint: Path, device: str):
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"CLIP checkpoint not found: {checkpoint}. "
            "The manifest builder never downloads weights automatically."
        )
    import clip

    model, preprocess = clip.load(str(checkpoint), device=device, jit=False)
    model.eval()
    labels = list(CLIP_PROMPTS)
    class_embeddings = []
    with torch.no_grad():
        for label in labels:
            tokens = clip.tokenize(CLIP_PROMPTS[label]).to(device)
            embedding = model.encode_text(tokens).float()
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            mean_embedding = embedding.mean(dim=0)
            class_embeddings.append(mean_embedding / mean_embedding.norm())
    return model, preprocess, labels, torch.stack(class_embeddings)


def clip_scores(model, preprocess, labels, text_features, images: list[Image.Image], device: str) -> list[dict]:
    tensor = torch.stack([preprocess(image) for image in images]).to(device)
    with torch.no_grad():
        features = model.encode_image(tensor).float()
        features = features / features.norm(dim=-1, keepdim=True)
        probabilities = (100.0 * features @ text_features.T).softmax(dim=-1).cpu().numpy()
    return [
        {f"clip_{label.lower()}": float(score) for label, score in zip(labels, row)}
        for row in probabilities
    ]


def image_from_hdf5(dataset, nsd_image_id: int) -> Image.Image:
    value = np.asarray(dataset[nsd_image_id])
    if value.ndim != 3:
        raise ValueError(f"Unexpected NSD image shape: {value.shape}")
    if value.shape[0] in {1, 3} and value.shape[-1] not in {1, 3}:
        value = np.moveaxis(value, 0, -1)
    return Image.fromarray(value.astype(np.uint8)).convert("RGB")


def select_rows(rows: list[dict], category: str, target: int, confirmatory: bool) -> list[dict]:
    if confirmatory:
        candidates = [row for row in rows if row["confirmatory_category"] == category]
    else:
        candidates = [
            row
            for row in rows
            if row[f"{category.lower()}_flag"] and not row["confirmatory_category"]
        ]
    return sorted(
        candidates,
        key=lambda row: (-float(row[f"{category.lower()}_selection_score"]), int(row["dataset_idx"])),
    )[:target]


def audit_grid(
    rows_by_category: dict[str, list[dict]],
    stimuli,
    output_path: Path,
    per_category: int,
) -> None:
    categories = ["Face", "Body", "Scene", "Word"]
    cell = 180
    label_height = 34
    canvas = Image.new("RGB", (cell * per_category, (cell + label_height) * len(categories)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, category in enumerate(categories):
        category_rows = rows_by_category[category]
        if len(category_rows) <= per_category:
            selected = category_rows
        else:
            audit_indices = np.linspace(
                0, len(category_rows) - 1, num=per_category, dtype=int
            )
            selected = [category_rows[int(index)] for index in audit_indices]
        for column, row in enumerate(selected):
            image = image_from_hdf5(stimuli, int(row["nsd_image_id"])).resize((cell, cell))
            x = column * cell
            y = row_index * (cell + label_height)
            canvas.paste(image, (x, y))
            draw.text(
                (x + 3, y + cell + 2),
                f"{category} idx={row['dataset_idx']} score={float(row[f'{category.lower()}_selection_score']):.3f}",
                fill="black",
            )
        y = row_index * (cell + label_height)
        draw.text((3, y + 3), category, fill="red")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.audit_per_category <= 0:
        raise ValueError("Batch and audit sizes must be positive")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    thresholds = {key: float(value) for key, value in config["thresholds"].items()}
    target = int(config["target_per_category"])
    minimum = int(config["minimum_confirmatory_count"])
    minimum_word = int(config["minimum_word_count"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import cv2

    if not args.face_cascade.is_file():
        raise FileNotFoundError(
            f"OpenCV face cascade not found: {args.face_cascade}. "
            "Provide a local file explicitly; this script never downloads detector weights."
        )
    face_detector = cv2.CascadeClassifier(str(args.face_cascade))
    if face_detector.empty():
        raise ValueError(f"Failed to load OpenCV face cascade: {args.face_cascade}")
    person_geometry, coco_annotation_files = load_person_geometry(
        args.coco_annotations_root
    )
    model, preprocess, clip_labels, text_features = load_clip(args.clip_checkpoint, args.device)

    metadata = np.load(args.metadata, allow_pickle=True).item()
    test_ids = [int(value) for value in np.asarray(metadata["test_img_num"]).reshape(-1)]
    stim_info = pd.read_csv(args.stim_info_csv).set_index("nsdId", drop=False)
    rows: list[dict] = []
    pending_images: list[Image.Image] = []
    pending_base: list[dict] = []

    with h5py.File(args.stimuli_hdf5, "r") as handle:
        if "imgBrick" not in handle:
            raise KeyError(f"imgBrick not found; keys={list(handle.keys())}")
        stimuli = handle["imgBrick"]
        for dataset_idx, nsd_image_id in enumerate(test_ids):
            image = image_from_hdf5(stimuli, nsd_image_id)
            geometry = detect_face_geometry(np.asarray(image), cv2, face_detector)
            info = stim_info.loc[nsd_image_id]
            coco_split = str(info["cocoSplit"])
            coco_image_id = int(info["cocoId"])
            coco_key = (coco_split, coco_image_id)
            if coco_key not in person_geometry:
                raise KeyError(f"COCO image missing from instance annotations: {coco_key}")
            pending_images.append(image)
            pending_base.append({
                "dataset_idx": dataset_idx,
                "nsd_image_id": nsd_image_id,
                "coco_image_id": coco_image_id,
                "coco_split": coco_split,
                **geometry,
                **person_geometry[coco_key],
            })
            if len(pending_images) < args.batch_size and dataset_idx + 1 < len(test_ids):
                continue
            scores = clip_scores(
                model, preprocess, clip_labels, text_features, pending_images, args.device
            )
            for base, score in zip(pending_base, scores):
                decision = assign_category(
                    face_confidence=float(base["face_confidence"]),
                    face_area_ratio=float(base["face_area_ratio"]),
                    person_count=int(base["person_count"]),
                    person_area_ratio=float(base["person_area_ratio"]),
                    clip_face=score["clip_face"],
                    clip_body=score["clip_body"],
                    clip_scene=score["clip_scene"],
                    clip_word=score["clip_word"],
                    thresholds=thresholds,
                )
                flags = decision["category_flags"]
                selection_scores = decision["category_scores"]
                rows.append({
                    **base,
                    **score,
                    "ocr_available": False,
                    "ocr_character_count": 0,
                    "ocr_area_ratio": 0.0,
                    "face_flag": flags["Face"],
                    "body_flag": flags["Body"],
                    "scene_flag": flags["Scene"],
                    "word_flag": flags["Word"],
                    "confirmatory_category": decision["confirmatory_category"],
                    "exclusion_reason": decision["exclusion_reason"],
                    "face_selection_score": selection_scores["Face"],
                    "body_selection_score": selection_scores["Body"],
                    "scene_selection_score": selection_scores["Scene"],
                    "word_selection_score": selection_scores["Word"],
                })
            pending_images.clear()
            pending_base.clear()

        fieldnames = list(rows[0])
        write_csv(args.output_dir / "all_test_images.csv", rows, fieldnames)
        selected_by_category = {
            category: select_rows(
                rows, category, target, confirmatory=category != "Word"
            )
            for category in ("Face", "Body", "Scene", "Word")
        }
        for category in ("Face", "Body", "Scene", "Word"):
            candidate_rows = [row for row in rows if row[f"{category.lower()}_flag"]]
            write_csv(
                args.output_dir / f"{category.lower()}_candidates.csv",
                candidate_rows,
                fieldnames,
            )

        confirmatory_rows = []
        for category in ("Face", "Body", "Scene"):
            for row in selected_by_category[category]:
                copied = dict(row)
                copied["analysis_role"] = "confirmatory"
                confirmatory_rows.append(copied)
        exploratory_rows = []
        for row in selected_by_category["Word"]:
            copied = dict(row)
            copied["analysis_role"] = "exploratory_word_no_ocr"
            exploratory_rows.append(copied)
        manifest_fields = [*fieldnames, "analysis_role"]
        write_csv(
            args.output_dir / "confirmatory_manifest.csv",
            confirmatory_rows,
            manifest_fields,
        )
        write_csv(
            args.output_dir / "exploratory_manifest.csv",
            exploratory_rows,
            manifest_fields,
        )
        audit_grid(
            selected_by_category,
            stimuli,
            args.output_dir / "figures" / "category_audit_grid.png",
            args.audit_per_category,
        )

    candidate_counts = {
        category: int(sum(bool(row[f"{category.lower()}_flag"]) for row in rows))
        for category in ("Face", "Body", "Scene", "Word")
    }
    selected_counts = {
        category: len(selected_by_category[category])
        for category in ("Face", "Body", "Scene", "Word")
    }
    shortages = {
        category: count < minimum
        for category, count in selected_counts.items()
        if category != "Word"
    }
    output_paths = [
        args.output_dir / "all_test_images.csv",
        args.output_dir / "face_candidates.csv",
        args.output_dir / "body_candidates.csv",
        args.output_dir / "scene_candidates.csv",
        args.output_dir / "word_candidates.csv",
        args.output_dir / "confirmatory_manifest.csv",
        args.output_dir / "exploratory_manifest.csv",
        args.output_dir / "figures" / "category_audit_grid.png",
    ]
    manifest_metadata = {
        "schema_version": 1,
        "subject": int(config["subject"]),
        "num_test_images": len(rows),
        "category_method": {
            "semantic": "OpenAI CLIP RN50 zero-shot prompt ensembles",
            "face_geometry": "OpenCV Haar frontal-face cascade, gated by COCO person presence",
            "person_geometry": "COCO 2017 instance segmentation annotations",
            "ocr": "unavailable; Word is exploratory and CLIP-only",
        },
        "software": {
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "clip_checkpoint": args.clip_checkpoint.name,
        "clip_model": str(config["clip_model"]),
        "clip_checkpoint_source": (
            "https://openaipublic.azureedge.net/clip/models/"
            "afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/"
            "RN50.pt"
        ),
        "clip_checkpoint_sha256": sha256_file(args.clip_checkpoint),
        "face_cascade": args.face_cascade.name,
        "face_cascade_source": (
            "https://raw.githubusercontent.com/opencv/opencv/4.12.0/"
            "data/haarcascades/haarcascade_frontalface_default.xml"
        ),
        "face_cascade_sha256": sha256_file(args.face_cascade),
        "coco_annotation_source": (
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        ),
        "coco_annotation_files": {
            path.name: sha256_file(path) for path in coco_annotation_files
        },
        "inference": {
            "device": args.device,
            "batch_size": args.batch_size,
            "clip_logit_scale": 100.0,
            "face_scale_factor": 1.1,
            "face_min_neighbors": 5,
            "face_min_size": [24, 24],
        },
        "audit_sampling": "evenly spaced score ranks across each selected category",
        "clip_prompts": CLIP_PROMPTS,
        "thresholds": thresholds,
        "candidate_counts": candidate_counts,
        "selected_counts": selected_counts,
        "confirmatory_minimum": minimum,
        "confirmatory_shortages": shortages,
        "word_minimum": minimum_word,
        "word_confirmatory": False,
        "word_status": "exploratory_no_ocr_evidence",
        "git_commit": git_commit(REPRO_ROOT),
        "selection_uses_reconstruction_outputs": False,
        "output_hashes": {
            path.relative_to(args.output_dir).as_posix(): sha256_file(path)
            for path in output_paths
        },
    }
    (args.output_dir / "manifest_metadata.json").write_text(
        json.dumps(manifest_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest_metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
