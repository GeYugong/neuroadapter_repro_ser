#!/usr/bin/env python3
"""Evaluate E3 outputs with global and category-specific local metrics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "src"))
sys.path.insert(0, str(REPRO_ROOT / "scripts"))

from evaluate_e2 import load_models, load_records
from neuro_roi_causal.e2 import read_csv
from neuro_roi_causal.e3 import interaction_rows, joint_result_rows
from neuro_roi_causal.local_metrics import (
    HAAR_MIN_NEIGHBORS,
    HAAR_MIN_SIZE,
    HAAR_SCALE_FACTOR,
    apply_region,
    crop_pair_by_box,
    detect_largest_face,
    face_detector_backend,
    file_sha256,
    load_coco_person_annotations,
    person_mask,
    region_pixel_consistency,
)
from neuro_roi_causal.metrics import (
    ContentAddressedCache,
    load_rgb,
    pixel_corr,
    representation_metrics,
    ssim_gray,
)
from neuro_roi_causal.stats import causal_loss


GLOBAL_METRICS = ("pixel_corr", "ssim", "lpips", "clip", "dino")
LOCAL_METRICS = {
    "Face": ("face_lpips", "face_dino", "face_detection_success"),
    "Body": ("person_lpips", "person_dino", "person_region_consistency"),
    "Scene": ("background_dino", "background_clip", "scene_class_consistency"),
}
SCENE_LABELS = (
    "indoor room",
    "street",
    "beach",
    "mountain",
    "forest",
    "field",
    "waterfront",
    "urban skyline",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coco-annotations", type=Path, required=True)
    parser.add_argument("--haar-cascade", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--dinov2-repo", type=Path, required=True)
    parser.add_argument("--lpips-package-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument(
        "--mode",
        choices=("smoke", "pilot", "formal"),
        default="smoke",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Deprecated alias for --mode smoke.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_pair(
    output_dir: Path,
    stem: str,
    gt: Image.Image,
    pred: Image.Image,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_path = output_dir / f"{stem}_gt.png"
    pred_path = output_dir / f"{stem}_pred.png"
    gt.convert("RGB").save(gt_path)
    pred.convert("RGB").save(pred_path)
    return gt_path, pred_path


def scene_label_consistency(
    gt: Path, pred: Path, models: dict, cache: ContentAddressedCache
) -> float:
    import clip

    device = models["device"]
    text = clip.tokenize([f"a photograph of a {label}" for label in SCENE_LABELS]).to(
        device
    )
    with torch.inference_mode():
        text_features = models["clip"].encode_text(text).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    def label(path: Path) -> int:
        feature = cache.embedding(
            "clip",
            path,
            lambda item: models["clip"].encode_image(
                models["clip_preprocess"](Image.open(item).convert("RGB"))
                .unsqueeze(0)
                .to(device)
            )[0].float(),
        ).to(device)
        feature /= feature.norm()
        return int((feature @ text_features.T).argmax().item())

    return float(label(gt) == label(pred))


def visual_grid(
    category_root: Path,
    records: dict[str, dict],
    metadata: dict[str, dict],
    output: Path,
) -> None:
    targets = [
        name for name, info in metadata.items() if info["control_type"] == "target_roi"
    ]
    names = ["no_mask", *targets]
    display_names = [
        "GT",
        "no mask",
        *[f"mask {metadata[name]['masked_roi']}" for name in targets],
    ]
    dataset_idx = next(iter(records["no_mask"]))
    cell, header = 200, 40
    canvas = Image.new("RGB", (cell * (len(names) + 1), header + cell), "white")
    draw = ImageDraw.Draw(canvas)
    for column, name in enumerate(display_names):
        draw.text((column * cell + 4, 10), name, fill="black")
    gt = Image.open(records["no_mask"][dataset_idx]["gt"]).convert("RGB")
    canvas.paste(gt.resize((cell, cell)), (0, header))
    for column, name in enumerate(names, start=1):
        pred = Image.open(records[name][dataset_idx]["pred"]).convert("RGB")
        canvas.paste(pred.resize((cell, cell)), (column * cell, header))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def effect_plot(rows: list[dict], output: Path, value_column: str) -> None:
    valid = [
        row
        for row in rows
        if row.get(value_column) not in (None, "")
        and np.isfinite(float(row[value_column]))
    ]
    labels = [
        " / ".join(
            str(row[key])
            for key in ("image_category", "masked_roi", "metric")
            if key in row
        )
        for row in valid
    ]
    values = [float(row[value_column]) for row in valid]
    height = max(4.0, 0.22 * len(valid))
    figure, axis = plt.subplots(figsize=(10, height))
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.scatter(values, range(len(values)), color="#1f77b4", s=22)
    axis.set_yticks(range(len(values)), labels)
    axis.set_xlabel("Engineering-smoke effect estimate (no CI)")
    axis.set_title("Descriptive smoke effects; not formal inference")
    axis.invert_yaxis()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def aggregate_excess(
    rows: list[dict],
    metadata_by_category: dict[str, dict[str, dict]],
) -> list[dict]:
    output = []
    for category in ("Face", "Body", "Scene"):
        category_rows = [row for row in rows if row["image_category"] == category]
        keys = sorted(
            {
                (int(row["seed"]), int(row["dataset_idx"]))
                for row in category_rows
                if row["condition"] == "no_mask"
            }
        )
        by_condition = {
            name: {
                (int(row["seed"]), int(row["dataset_idx"])): row
                for row in category_rows
                if row["condition"] == name
            }
            for name in metadata_by_category[category]
        }
        baseline = by_condition["no_mask"]
        for target, info in metadata_by_category[category].items():
            if info.get("control_type") != "target_roi":
                continue
            controls = [
                name
                for name, candidate in metadata_by_category[category].items()
                if candidate.get("control_type") == "pure_matched_random"
                and candidate.get("masked_roi") == info["masked_roi"]
            ]
            metrics = [*GLOBAL_METRICS, *LOCAL_METRICS[category]]
            for metric in metrics:
                by_image: dict[int, list[float]] = defaultdict(list)
                for key in keys:
                    if any(
                        by_condition[name][key].get(metric) in (None, "")
                        for name in [target, *controls, "no_mask"]
                    ):
                        continue
                    target_loss = causal_loss(
                        metric,
                        float(baseline[key][metric]),
                        float(by_condition[target][key][metric]),
                    )
                    random_loss = np.mean(
                        [
                            causal_loss(
                                metric,
                                float(baseline[key][metric]),
                                float(by_condition[name][key][metric]),
                            )
                            for name in controls
                        ]
                    )
                    by_image[key[1]].append(float(target_loss - random_loss))
                for dataset_idx, seed_values in by_image.items():
                    output.append(
                        {
                            "image_category": category,
                            "masked_roi": info["masked_roi"],
                            "metric": metric,
                            "dataset_idx": dataset_idx,
                            "excess_causal_loss": float(np.mean(seed_values)),
                            "num_seeds": len(seed_values),
                        }
                    )
    return output


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    mode = "smoke" if args.smoke else args.mode
    for required in (
        args.manifest,
        args.coco_annotations / "instances_train2017.json",
        args.coco_annotations / "instances_val2017.json",
        args.haar_cascade,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"Required local evaluation asset missing: {required}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    derived_root = args.output_dir / "derived_local_regions"
    manifest = {
        int(row["dataset_idx"]): row for row in read_csv(args.manifest)
    }
    coco = load_coco_person_annotations(args.coco_annotations)
    models = load_models(args)
    require_opencv = mode in {"pilot", "formal"}
    cascade_sha256 = file_sha256(args.haar_cascade)
    detector_backend = face_detector_backend(
        args.haar_cascade,
        require_opencv=require_opencv,
    )
    detector_plan = plan.get("face_detector")
    if require_opencv:
        if detector_plan is None:
            raise ValueError("pilot/formal plan must freeze face_detector settings")
        expected_detector = {
            "backend": "opencv_haar_e1",
            "cascade_filename": args.haar_cascade.name,
            "cascade_sha256": cascade_sha256,
            "scale_factor": HAAR_SCALE_FACTOR,
            "min_neighbors": HAAR_MIN_NEIGHBORS,
            "min_size": list(HAAR_MIN_SIZE),
        }
        mismatched = {
            key: {"plan": detector_plan.get(key), "actual": value}
            for key, value in expected_detector.items()
            if detector_plan.get(key) != value
        }
        if mismatched:
            raise ValueError(
                f"Face detector differs from the frozen pilot plan: {mismatched}"
            )
        if detector_plan.get(f"{mode}_fallback_allowed") is not False:
            raise ValueError(f"{mode} plan must explicitly forbid detector fallback")
    cache = ContentAddressedCache()
    rows: list[dict] = []
    representation_jobs: list[tuple[int, str, Path, Path]] = []
    metadata_by_category: dict[str, dict[str, dict]] = {}
    seed_roots = sorted(path for path in args.run_root.glob("seed_*") if path.is_dir())
    observed_seeds = [int(path.name.removeprefix("seed_")) for path in seed_roots]
    if mode == "smoke" and observed_seeds != [int(plan["seeds"][0])]:
        raise ValueError("Engineering smoke must contain only the first frozen seed")
    if mode == "pilot" and observed_seeds != [
        int(seed)
        for seed in plan["seeds"][: int(plan["execution"]["pilot_seeds"])]
    ]:
        raise ValueError("Pilot seed set does not match the frozen execution config")
    if mode == "formal" and observed_seeds != [int(seed) for seed in plan["seeds"]]:
        raise ValueError("Formal evaluation requires all frozen seeds")
    for category in ("Face", "Body", "Scene"):
        visual_records = None
        for seed_root in seed_roots:
            seed = int(seed_root.name.removeprefix("seed_"))
            records, metadata = load_records(seed_root / category)
            observed_indices = sorted(records["no_mask"])
            frozen_indices = plan["categories"][category]["dataset_indices"]
            if mode == "smoke" and observed_indices != sorted(frozen_indices[:1]):
                raise ValueError(f"{category} smoke indices differ from the plan")
            if mode == "pilot" and observed_indices != sorted(
                frozen_indices[: int(plan["execution"]["pilot_images_per_category"])]
            ):
                raise ValueError(f"{category} pilot indices differ from the plan")
            if mode == "formal" and observed_indices != sorted(frozen_indices):
                raise ValueError(f"{category} formal indices differ from the plan")
            metadata_by_category[category] = metadata
            visual_records = records
            for condition, condition_records in records.items():
                for dataset_idx, record in sorted(condition_records.items()):
                    gt_path, pred_path = Path(record["gt"]), Path(record["pred"])
                    gt_image = Image.open(gt_path).convert("RGB")
                    pred_image = Image.open(pred_path).convert("RGB")
                    gt_array = load_rgb(gt_path, 425)
                    pred_array = load_rgb(pred_path, 425)
                    row = {
                        "experiment": plan["name"],
                        "image_category": category,
                        "condition": condition,
                        "masked_roi": metadata[condition]["masked_roi"],
                        "control_type": metadata[condition]["control_type"],
                        "seed": seed,
                        "dataset_idx": dataset_idx,
                        "pixel_corr": pixel_corr(gt_array, pred_array),
                        "ssim": ssim_gray(gt_array, pred_array),
                    }
                    row_index = len(rows)
                    rows.append(row)
                    representation_jobs.append((row_index, "global", gt_path, pred_path))
                    manifest_row = manifest[dataset_idx]
                    stem = f"{category}_{seed}_{dataset_idx}_{condition}"
                    if category == "Face":
                        gt_box = detect_largest_face(
                            gt_image,
                            args.haar_cascade,
                            require_opencv=require_opencv,
                        )
                        pred_box = detect_largest_face(
                            pred_image,
                            args.haar_cascade,
                            require_opencv=require_opencv,
                        )
                        if gt_box is None:
                            raise RuntimeError(
                                "Ground-truth Face region is empty for "
                                f"dataset_idx={dataset_idx}, condition={condition}"
                            )
                        row["face_detection_success"] = float(pred_box is not None)
                        row["face_detector_backend"] = detector_backend
                        gt_local, pred_local = crop_pair_by_box(
                            gt_image, pred_image, gt_box
                        )
                        if not gt_local.width or not gt_local.height:
                            raise RuntimeError("Detected Face crop is empty")
                        row["local_region_pixels"] = (
                            gt_local.width * gt_local.height
                        )
                        paths = save_pair(derived_root, stem, gt_local, pred_local)
                        representation_jobs.append((row_index, "face", *paths))
                    else:
                        key = (
                            str(manifest_row["coco_split"]),
                            int(manifest_row["coco_image_id"]),
                        )
                        mask, method = person_mask(coco[key], gt_image.size)
                        row["person_mask_method"] = method
                        keep = category == "Body"
                        mask_array = np.asarray(mask, dtype=np.uint8) > 0
                        selected_pixels = (
                            int(mask_array.sum())
                            if keep
                            else int((~mask_array).sum())
                        )
                        if selected_pixels == 0:
                            raise RuntimeError(
                                f"{category} local region is empty for "
                                f"dataset_idx={dataset_idx}"
                            )
                        row["local_region_pixels"] = selected_pixels
                        if keep:
                            row["person_region_consistency"] = (
                                region_pixel_consistency(
                                    gt_image,
                                    pred_image,
                                    mask,
                                )
                            )
                        gt_local = apply_region(gt_image, mask, keep_mask=keep)
                        pred_local = apply_region(pred_image, mask, keep_mask=keep)
                        paths = save_pair(derived_root, stem, gt_local, pred_local)
                        representation_jobs.append(
                            (row_index, "person" if keep else "background", *paths)
                        )
    pairs = [(gt, pred) for _, _, gt, pred in representation_jobs]
    values = representation_metrics(pairs, models, args.batch_size, cache)
    for (row_index, kind, gt_path, pred_path), result in zip(
        representation_jobs, values
    ):
        row = rows[row_index]
        if kind == "global":
            row.update({key: result[key] for key in ("lpips", "clip", "dino")})
        elif kind == "face":
            row["face_lpips"] = result["lpips"]
            row["face_dino"] = result["dino"]
        elif kind == "person":
            row["person_lpips"] = result["lpips"]
            row["person_dino"] = result["dino"]
        else:
            row["background_dino"] = result["dino"]
            row["background_clip"] = result["clip"]
            row["scene_class_consistency"] = scene_label_consistency(
                gt_path, pred_path, models, cache
            )
    required_metrics = {
        category: [*GLOBAL_METRICS, *LOCAL_METRICS[category]]
        for category in ("Face", "Body", "Scene")
    }
    invalid_metrics = []
    for row in rows:
        for metric in required_metrics[row["image_category"]]:
            value = row.get(metric)
            if value in (None, "") or not np.isfinite(float(value)):
                invalid_metrics.append(
                    {
                        "category": row["image_category"],
                        "dataset_idx": row["dataset_idx"],
                        "condition": row["condition"],
                        "metric": metric,
                        "value": value,
                    }
                )
    if invalid_metrics:
        raise RuntimeError(
            f"Missing or non-finite evaluation metrics: {invalid_metrics[:10]}"
        )
    write_csv(args.output_dir / "per_sample_metrics.csv", rows)
    local_rows = [
        {
            key: value
            for key, value in row.items()
            if key
            in {
                "experiment",
                "image_category",
                "condition",
                "masked_roi",
                "control_type",
                "seed",
                "dataset_idx",
                "person_mask_method",
                "local_region_pixels",
                *LOCAL_METRICS[row["image_category"]],
            }
        }
        for row in rows
    ]
    write_csv(args.output_dir / "local_metric_results.csv", local_rows)
    excess = aggregate_excess(rows, metadata_by_category)
    write_csv(args.output_dir / "per_image_excess_effects.csv", excess)
    if plan["name"] == "E3_interaction":
        results = interaction_rows(
            excess,
            metrics=GLOBAL_METRICS,
            draws=args.bootstrap_draws,
            formal=mode == "formal",
            analysis_status=(
                "formal"
                if mode == "formal"
                else f"engineering_{mode}"
            ),
        )
        write_csv(args.output_dir / "interaction_results.csv", results)
        effect_plot(
            results,
            args.output_dir / "figures" / "effect_forest_plot.png",
            "matched_minus_nonmatched",
        )
    else:
        results = joint_result_rows(
            excess,
            draws=args.bootstrap_draws,
            formal=mode == "formal",
            analysis_status=(
                "formal"
                if mode == "formal"
                else f"engineering_{mode}"
            ),
        )
        write_csv(args.output_dir / "joint_mask_results.csv", results)
        effect_plot(
            [row for row in results if row["metric"] == "dino"],
            args.output_dir / "figures" / "effect_forest_plot.png",
            "target_minus_pure_random",
        )
    for category in ("Face", "Body", "Scene"):
        visual_grid(
            seed_roots[0] / category,
            load_records(seed_roots[0] / category)[0],
            metadata_by_category[category],
            args.output_dir / "figures" / f"{category.lower()}_comparison_grid.png",
        )
    summary = {
        "experiment": plan["name"],
        "scope": mode,
        "repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPRO_ROOT,
            text=True,
        ).strip(),
        "python_executable": sys.executable,
        "formal_inference_performed": mode == "formal",
        "num_rows": len(rows),
        "num_excess_rows": len(excess),
        "metric_models": models["metadata"],
        "face_detector_backend": detector_backend,
        "opencv_version": __import__("cv2").__version__,
        "face_detector_formal_compatibility": (
            detector_backend == "opencv_haar_e1"
        ),
        "face_detector_cascade": str(args.haar_cascade.resolve()),
        "face_detector_cascade_sha256": cascade_sha256,
        "face_detector_parameters": {
            "scaleFactor": HAAR_SCALE_FACTOR,
            "minNeighbors": HAAR_MIN_NEIGHBORS,
            "minSize": list(HAAR_MIN_SIZE),
        },
        "invalid_or_missing_metric_count": len(invalid_metrics),
        "minimum_local_region_pixels": min(
            int(row["local_region_pixels"])
            for row in rows
            if "local_region_pixels" in row
        ),
        "content_addressed_cache": {
            "embedding_computations": cache.embedding_computations,
            "pair_computations": cache.pair_computations,
        },
    }
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
