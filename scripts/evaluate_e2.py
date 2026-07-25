#!/usr/bin/env python3
"""Unified, content-addressed evaluator for E2 zero and mean experiments."""

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
from torchvision import transforms

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "src"))

from neuro_roi_causal.mean_cache import file_sha256
from neuro_roi_causal.metrics import (
    ContentAddressedCache,
    load_rgb,
    pixel_corr,
    representation_metrics,
    ssim_gray,
)
from neuro_roi_causal.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    causal_loss,
    sign_flip_pvalue,
)


METRICS = ("pixel_corr", "ssim", "lpips", "clip", "dino")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--clip-checkpoint", type=Path, required=True)
    parser.add_argument("--dinov2-repo", type=Path, required=True)
    parser.add_argument("--lpips-package-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to RUN_ROOT/evaluation; use a new directory for rechecks.",
    )
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


def load_models(args: argparse.Namespace):
    if args.lpips_package_root is not None:
        sys.path.insert(0, str(args.lpips_package_root))
    import clip
    import lpips

    for required in (args.clip_checkpoint, args.dinov2_repo):
        if not required.exists():
            raise FileNotFoundError(f"Required local model asset is missing: {required}")
    device = torch.device(args.device)
    original_download = torch.hub.download_url_to_file

    def reject_download(*_args, **_kwargs):
        raise RuntimeError("Evaluator is forbidden from downloading model weights")

    torch.hub.download_url_to_file = reject_download
    try:
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
    finally:
        torch.hub.download_url_to_file = original_download
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
        "metadata": {
            "clip_checkpoint": str(args.clip_checkpoint.resolve()),
            "clip_checkpoint_sha256": file_sha256(args.clip_checkpoint),
            "dinov2_repo": str(args.dinov2_repo.resolve()),
            "lpips_package_root": (
                str(args.lpips_package_root.resolve())
                if args.lpips_package_root is not None
                else None
            ),
            "clip_preprocess": repr(clip_preprocess),
            "dino_preprocess": repr(dino_preprocess),
            "lpips_preprocess": repr(lpips_preprocess),
            "torch_version": torch.__version__,
        },
    }


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
        key = (int(row["seed"]), int(row["dataset_idx"]))
        by_condition[row["condition"]][key] = row
    baseline = by_condition["no_mask"]
    observation_keys = sorted(baseline)

    def aggregate_seeds(values: np.ndarray) -> np.ndarray:
        grouped = defaultdict(list)
        for (_, dataset_idx), value in zip(observation_keys, values):
            grouped[dataset_idx].append(float(value))
        return np.asarray(
            [np.mean(grouped[index]) for index in sorted(grouped)],
            dtype=np.float64,
        )

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
                        baseline[key][metric],
                        values_by_index[key][metric],
                    )
                    for key in observation_keys
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
            target_by_image = aggregate_seeds(losses[target][metric])
            random_by_image = aggregate_seeds(random_mean_per_image)
            excess = target_by_image - random_by_image
            comparison["metrics"][metric] = {
                "target_causal_loss_mean": float(target_by_image.mean()),
                "random_causal_loss_mean": float(random_by_image.mean()),
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
        "num_samples": len({dataset_idx for _, dataset_idx in baseline}),
        "num_seeds": len({seed for seed, _ in baseline}),
        "statistical_unit": "image after averaging causal loss across seeds",
        "conditions": condition_summary,
        "target_vs_matched_random": comparisons,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    models = load_models(args)
    metric_cache = ContentAddressedCache()
    output_root = args.output_dir or args.run_root / "evaluation"
    output_root.mkdir(parents=True, exist_ok=True)
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
        "model_metadata": models["metadata"],
    }
    if (args.run_root / "Face").is_dir():
        seed_roots = [args.run_root]
    else:
        seed_roots = sorted(
            path
            for path in args.run_root.glob("seed_*")
            if path.is_dir()
        )
    if not seed_roots:
        raise ValueError(f"No category or seed directories under {args.run_root}")

    for category in ("Face", "Body", "Scene"):
        rows = []
        pairs = []
        visual_records = None
        metadata = None
        for seed_root in seed_roots:
            seed_text = seed_root.name.removeprefix("seed_")
            seed = int(seed_text) if seed_text.isdigit() else 0
            records, current_metadata = load_records(seed_root / category)
            if metadata is None:
                metadata = current_metadata
                visual_records = records
            elif current_metadata != metadata:
                raise ValueError(f"Condition metadata differs across seeds for {category}")
            for condition, condition_records in records.items():
                for dataset_idx, record in sorted(condition_records.items()):
                    gt_path, pred_path = Path(record["gt"]), Path(record["pred"])
                    gt, pred = load_rgb(gt_path, 425), load_rgb(pred_path, 425)
                    rows.append(
                        {
                            "category": category,
                            "condition": condition,
                            "seed": seed,
                            "dataset_idx": dataset_idx,
                            "pixel_corr": pixel_corr(gt, pred),
                            "ssim": ssim_gray(gt, pred),
                        }
                    )
                    pairs.append((gt_path, pred_path))
        neural = representation_metrics(pairs, models, args.batch_size, metric_cache)
        for row, values in zip(rows, neural):
            row.update(values)
        with (output_root / f"{category.lower()}_per_sample_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=["category", "condition", "seed", "dataset_idx", *METRICS])
            writer.writeheader()
            writer.writerows(rows)
        all_summary["categories"][category] = summarize_category(
            category, rows, metadata, args.bootstrap_draws
        )
        by_key = {
            (row["condition"], int(row["seed"]), int(row["dataset_idx"])): row
            for row in rows
        }
        for seed_root in seed_roots:
            seed_text = seed_root.name.removeprefix("seed_")
            seed = int(seed_text) if seed_text.isdigit() else 0
            current_records, _ = load_records(seed_root / category)
            for dataset_idx in current_records["no_mask"]:
                baseline_sha = metric_cache.sha(
                    current_records["no_mask"][dataset_idx]["pred"]
                )
                repeat_sha = metric_cache.sha(
                    current_records["no_mask_repeat"][dataset_idx]["pred"]
                )
                if baseline_sha != repeat_sha:
                    raise RuntimeError(
                        f"Image determinism failed for {category}/{seed}/{dataset_idx}"
                    )
                baseline = by_key[("no_mask", seed, dataset_idx)]
                repeat = by_key[("no_mask_repeat", seed, dataset_idx)]
                if any(baseline[metric] != repeat[metric] for metric in METRICS):
                    raise RuntimeError(
                        f"Metric determinism failed for {category}/{seed}/{dataset_idx}"
                    )
        make_visual_grid(
            seed_roots[0] / category,
            visual_records,
            metadata,
            output_root / f"{category.lower()}_comparison_grid.png",
        )
    primary_refs = []
    for category in ("Face", "Body", "Scene"):
        metrics = all_summary["categories"][category][
            "target_vs_matched_random"
        ]["full"]["metrics"]
        for metric in METRICS:
            primary_refs.append((category, metric, metrics[metric]))
    primary_qvalues = benjamini_hochberg(
        [item["sign_flip_p"] for _, _, item in primary_refs]
    )
    for (_, _, item), qvalue in zip(primary_refs, primary_qvalues):
        item["bh_q_primary_15_tests"] = qvalue
    all_summary["primary_multiplicity"] = {
        "family": "3 categories x 5 metrics, full-group target vs matched random",
        "method": "Benjamini-Hochberg",
        "num_tests": 15,
    }
    all_summary["content_addressed_cache"] = {
        "embedding_computations": metric_cache.embedding_computations,
        "pair_computations": metric_cache.pair_computations,
    }
    primary_rows = []
    for category, metric, item in primary_refs:
        primary_rows.append(
            {
                "category": category,
                "metric": metric,
                **item,
            }
        )
    with (output_root / "primary_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=sorted({key for row in primary_rows for key in row})
        )
        writer.writeheader()
        writer.writerows(primary_rows)
    (output_root / "e2_metrics_summary.json").write_text(
        json.dumps(all_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(all_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
