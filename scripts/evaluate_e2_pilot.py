#!/usr/bin/env python3
"""Evaluate E2 outputs with paired pixel and representation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity
from torchvision import transforms


METRICS = ("pixel_corr", "ssim", "lpips", "clip", "dino")
HIGHER_IS_BETTER = {"pixel_corr", "ssim", "clip", "dino"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--dinov2-repo", type=Path, required=True)
    parser.add_argument("--lpips-package-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    return parser.parse_args()


def load_records(category_root: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    conditions = {}
    metadata = {}
    for summary_path in sorted(category_root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        name = summary["condition"]["name"]
        conditions[name] = {
            int(row["dataset_idx"]): row for row in summary["records"]
        }
        metadata[name] = summary["condition"]
    if "no_mask" not in conditions:
        raise ValueError(f"No no_mask baseline under {category_root}")
    expected = set(conditions["no_mask"])
    if any(set(records) != expected for records in conditions.values()):
        raise ValueError(f"Condition dataset indices differ under {category_root}")
    return conditions, metadata


def load_rgb(path: str | Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(
        (size, size), Image.Resampling.BILINEAR
    )
    return np.asarray(image, dtype=np.float32) / 255.0


def pixel_corr(gt: np.ndarray, pred: np.ndarray) -> float:
    x = gt.reshape(-1).astype(np.float64)
    y = pred.reshape(-1).astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return 0.0 if denominator == 0 else float(np.dot(x, y) / denominator)


def ssim_gray(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(
        structural_similarity(
            rgb2gray(gt),
            rgb2gray(pred),
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=1.0,
        )
    )


def bootstrap_ci(values: np.ndarray, seed: int, draws: int) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))]
    return [float(value) for value in np.quantile(samples.mean(axis=1), [0.025, 0.975])]


def sign_flip_pvalue(values: np.ndarray, seed: int, draws: int = 20000) -> float:
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(values)))
    null = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (draws + 1))


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return [float(value) for value in output]


def causal_loss(metric: str, baseline: float, masked: float) -> float:
    if metric in HIGHER_IS_BETTER:
        return baseline - masked
    return masked - baseline


def load_models(args: argparse.Namespace):
    if args.lpips_package_root is not None:
        sys.path.insert(0, str(args.lpips_package_root))
    import clip
    import lpips

    device = torch.device(args.device)
    clip_model, clip_preprocess = clip.load(
        str(args.clip_checkpoint), device=device, jit=False
    )
    clip_model.eval()
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    dino_model = torch.hub.load(
        str(args.dinov2_repo),
        "dinov2_vitb14",
        source="local",
        pretrained=True,
    ).to(device).eval()
    dino_preprocess = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    lpips_preprocess = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return {
        "device": device,
        "clip": clip_model,
        "clip_preprocess": clip_preprocess,
        "lpips": lpips_model,
        "lpips_preprocess": lpips_preprocess,
        "dino": dino_model,
        "dino_preprocess": dino_preprocess,
    }


def representation_metrics(
    pairs: list[tuple[Path, Path]],
    models: dict,
    batch_size: int,
) -> list[dict[str, float]]:
    output = []
    device = models["device"]
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            gt_images = [Image.open(gt).convert("RGB") for gt, _ in batch]
            pred_images = [Image.open(pred).convert("RGB") for _, pred in batch]

            gt_lpips = torch.stack(
                [models["lpips_preprocess"](image) for image in gt_images]
            ).to(device)
            pred_lpips = torch.stack(
                [models["lpips_preprocess"](image) for image in pred_images]
            ).to(device)
            lpips_values = (
                models["lpips"](gt_lpips, pred_lpips).flatten().float().cpu().numpy()
            )

            gt_clip = torch.stack(
                [models["clip_preprocess"](image) for image in gt_images]
            ).to(device)
            pred_clip = torch.stack(
                [models["clip_preprocess"](image) for image in pred_images]
            ).to(device)
            gt_clip_features = models["clip"].encode_image(gt_clip).float()
            pred_clip_features = models["clip"].encode_image(pred_clip).float()
            clip_values = torch.nn.functional.cosine_similarity(
                gt_clip_features, pred_clip_features
            ).cpu().numpy()

            gt_dino = torch.stack(
                [models["dino_preprocess"](image) for image in gt_images]
            ).to(device)
            pred_dino = torch.stack(
                [models["dino_preprocess"](image) for image in pred_images]
            ).to(device)
            gt_dino_features = models["dino"](gt_dino).float()
            pred_dino_features = models["dino"](pred_dino).float()
            dino_values = torch.nn.functional.cosine_similarity(
                gt_dino_features, pred_dino_features
            ).cpu().numpy()

            output.extend(
                {
                    "lpips": float(lpips_value),
                    "clip": float(clip_value),
                    "dino": float(dino_value),
                }
                for lpips_value, clip_value, dino_value in zip(
                    lpips_values, clip_values, dino_values
                )
            )
    return output


def make_visual_grid(
    category_root: Path,
    records: dict[str, dict],
    metadata: dict[str, dict],
    output: Path,
) -> None:
    target = next(
        name
        for name, info in metadata.items()
        if info.get("control_type") == "target_roi" and info["design"] == "full"
    )
    random = next(
        name
        for name, info in metadata.items()
        if info.get("control_type") == "matched_random" and info["design"] == "full"
    )
    unrelated = next(
        name
        for name, info in metadata.items()
        if info.get("control_type") == "unrelated_roi"
    )
    names = ["no_mask", target, random, unrelated]
    indices = sorted(records["no_mask"])
    cell, header = 160, 32
    grid = Image.new(
        "RGB",
        (cell * (len(names) + 1), header + cell * len(indices)),
        "white",
    )
    draw = ImageDraw.Draw(grid)
    for column, name in enumerate(["GT", *names]):
        draw.text((column * cell + 4, 8), name, fill="black")
    for row, index in enumerate(indices):
        gt = Image.open(records["no_mask"][index]["gt"]).convert("RGB")
        grid.paste(gt.resize((cell, cell)), (0, header + row * cell))
        for column, name in enumerate(names, start=1):
            pred = Image.open(records[name][index]["pred"]).convert("RGB")
            grid.paste(
                pred.resize((cell, cell)),
                (column * cell, header + row * cell),
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)


def summarize_category(
    category: str,
    rows: list[dict],
    metadata: dict[str, dict],
    draws: int,
) -> dict:
    by_condition = defaultdict(dict)
    for row in rows:
        by_condition[row["condition"]][row["dataset_idx"]] = row
    baseline = by_condition["no_mask"]
    condition_summary = {}
    losses = {}
    for condition, values_by_index in by_condition.items():
        values = list(values_by_index.values())
        condition_summary[condition] = {
            metric: float(np.mean([row[metric] for row in values]))
            for metric in METRICS
        }
        if condition in {"no_mask", "no_mask_repeat"}:
            continue
        losses[condition] = {
            metric: np.asarray(
                [
                    causal_loss(
                        metric,
                        baseline[index][metric],
                        values_by_index[index][metric],
                    )
                    for index in sorted(baseline)
                ]
            )
            for metric in METRICS
        }
        condition_summary[condition]["causal_loss_mean"] = {
            metric: float(values.mean()) for metric, values in losses[condition].items()
        }

    comparisons = {}
    target_conditions = [
        name
        for name, info in metadata.items()
        if info.get("control_type") == "target_roi"
    ]
    for target in target_conditions:
        design = metadata[target]["design"]
        random_names = [
            name
            for name, info in metadata.items()
            if info.get("control_type") == "matched_random"
            and info["design"] == design
        ]
        comparison = {"target": target, "random_conditions": random_names, "metrics": {}}
        for metric_index, metric in enumerate(METRICS):
            random_mean_per_image = np.mean(
                np.stack([losses[name][metric] for name in random_names]),
                axis=0,
            )
            excess = losses[target][metric] - random_mean_per_image
            comparison["metrics"][metric] = {
                "target_causal_loss_mean": float(losses[target][metric].mean()),
                "random_causal_loss_mean": float(random_mean_per_image.mean()),
                "target_minus_random_mean": float(excess.mean()),
                "ci95": bootstrap_ci(
                    excess,
                    5000 + metric_index + len(category) + len(design),
                    draws,
                ),
                "sign_flip_p": sign_flip_pvalue(
                    excess,
                    6000 + metric_index + len(category) + len(design),
                ),
            }
        qvalues = benjamini_hochberg(
            [
                comparison["metrics"][metric]["sign_flip_p"]
                for metric in METRICS
            ]
        )
        for metric, qvalue in zip(METRICS, qvalues):
            comparison["metrics"][metric]["bh_q_within_design"] = qvalue
        comparisons[design] = comparison
    return {
        "category": category,
        "num_samples": len(baseline),
        "conditions": condition_summary,
        "target_vs_matched_random": comparisons,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    models = load_models(args)
    output_root = args.run_root / "evaluation"
    output_root.mkdir(exist_ok=True)
    all_summary = {
        "run_root": str(args.run_root),
        "metric_direction": {
            "pixel_corr": "higher_is_better",
            "ssim": "higher_is_better",
            "lpips": "lower_is_better",
            "clip": "higher_is_better",
            "dino": "higher_is_better",
        },
        "causal_loss": "positive means masking degraded reconstruction",
        "categories": {},
    }
    for category in ("Face", "Body", "Scene"):
        category_root = args.run_root / category
        records, metadata = load_records(category_root)
        rows = []
        pairs = []
        for condition, condition_records in records.items():
            for dataset_idx, record in sorted(condition_records.items()):
                gt_path, pred_path = Path(record["gt"]), Path(record["pred"])
                gt, pred = load_rgb(gt_path, 425), load_rgb(pred_path, 425)
                rows.append(
                    {
                        "category": category,
                        "condition": condition,
                        "dataset_idx": dataset_idx,
                        "pixel_corr": pixel_corr(gt, pred),
                        "ssim": ssim_gray(gt, pred),
                    }
                )
                pairs.append((gt_path, pred_path))
        neural = representation_metrics(pairs, models, args.batch_size)
        for row, values in zip(rows, neural):
            row.update(values)
        with (output_root / f"{category.lower()}_per_sample_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["category", "condition", "dataset_idx", *METRICS])
            writer.writeheader()
            writer.writerows(rows)
        all_summary["categories"][category] = summarize_category(
            category, rows, metadata, args.bootstrap_draws
        )
        make_visual_grid(
            category_root,
            records,
            metadata,
            output_root / f"{category.lower()}_comparison_grid.png",
        )
    (output_root / "e2_metrics_summary.json").write_text(
        json.dumps(all_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(all_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
