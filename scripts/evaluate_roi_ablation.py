#!/usr/bin/env python3
"""Paired lightweight evaluation and visualization for ROI-ablation PNG outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity


def load_rgb(path: str | Path, size=(425, 425)) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def pixel_corr(gt: np.ndarray, pred: np.ndarray) -> float:
    x, y = gt.reshape(-1).astype(np.float64), pred.reshape(-1).astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return 0.0 if denominator == 0 else float(np.dot(x, y) / denominator)


def ssim_gray(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(structural_similarity(rgb2gray(gt), rgb2gray(pred), gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1.0))


def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 10000) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    sampled = values[generator.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return tuple(float(value) for value in np.quantile(sampled, [0.025, 0.975]))


def sign_flip_pvalue(values: np.ndarray, seed: int, draws: int = 20000) -> float:
    observed = abs(float(values.mean()))
    generator = np.random.default_rng(seed)
    signs = generator.choice(np.asarray([-1.0, 1.0]), size=(draws, len(values)))
    null = abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (draws + 1))


def load_records(summary_path: Path) -> dict[int, dict]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return {int(row["dataset_idx"]): row for row in data["records"]}


def visual_grid(root: Path, ordered_names: list[str], records: dict[str, dict[int, dict]], output: Path, samples: int = 8) -> None:
    indices = sorted(records["no_mask"])[:samples]
    cell = 160
    header = 28
    grid = Image.new("RGB", (cell * (len(ordered_names) + 1), header + cell * len(indices)), "white")
    draw = ImageDraw.Draw(grid)
    draw.text((4, 6), "GT", fill="black")
    for col, name in enumerate(ordered_names, start=1):
        draw.text((cell * col + 4, 6), name, fill="black")
    for row, index in enumerate(indices):
        gt = Image.open(records["no_mask"][index]["gt"]).convert("RGB").resize((cell, cell))
        grid.paste(gt, (0, header + row * cell))
        for col, name in enumerate(ordered_names, start=1):
            pred = Image.open(records[name][index]["pred"]).convert("RGB").resize((cell, cell))
            grid.paste(pred, (col * cell, header + row * cell))
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline", default="no_mask")
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    condition_dirs = sorted(path for path in args.run_root.iterdir() if path.is_dir() and (path / "summary.json").exists())
    names = [path.name for path in condition_dirs]
    if args.baseline not in names:
        raise ValueError(f"Baseline {args.baseline!r} not found")
    records = {name: load_records(args.run_root / name / "summary.json") for name in names}
    expected = set(records[args.baseline])
    if any(set(value) != expected for value in records.values()):
        raise RuntimeError("Conditions do not contain the same dataset indices")

    per_sample: list[dict] = []
    for name in names:
        for index in sorted(expected):
            item = records[name][index]
            gt, pred = load_rgb(item["gt"]), load_rgb(item["pred"])
            per_sample.append({"condition": name, "dataset_idx": index, "pixel_corr": pixel_corr(gt, pred), "ssim": ssim_gray(gt, pred)})
    output = args.run_root / "evaluation"
    output.mkdir(exist_ok=True)
    with (output / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["condition", "dataset_idx", "pixel_corr", "ssim"])
        writer.writeheader(); writer.writerows(per_sample)
    baseline = {row["dataset_idx"]: row for row in per_sample if row["condition"] == args.baseline}
    summary = {"run_root": str(args.run_root), "baseline": args.baseline, "num_samples": len(expected), "conditions": {}}
    for condition in names:
        values = [row for row in per_sample if row["condition"] == condition]
        pixel = np.asarray([row["pixel_corr"] for row in values])
        ssim = np.asarray([row["ssim"] for row in values])
        info = {"pixel_corr_mean": float(pixel.mean()), "pixel_corr_std": float(pixel.std(ddof=1)), "ssim_mean": float(ssim.mean()), "ssim_std": float(ssim.std(ddof=1))}
        if condition != args.baseline:
            pixel_delta = np.asarray([row["pixel_corr"] - baseline[row["dataset_idx"]]["pixel_corr"] for row in values])
            ssim_delta = np.asarray([row["ssim"] - baseline[row["dataset_idx"]]["ssim"] for row in values])
            info["paired_delta_vs_baseline"] = {
                "pixel_corr_mean": float(pixel_delta.mean()), "pixel_corr_ci95": bootstrap_ci(pixel_delta, 1000 + len(condition), args.bootstrap_draws), "pixel_corr_signflip_p": sign_flip_pvalue(pixel_delta, 2000 + len(condition)),
                "ssim_mean": float(ssim_delta.mean()), "ssim_ci95": bootstrap_ci(ssim_delta, 3000 + len(condition), args.bootstrap_draws), "ssim_signflip_p": sign_flip_pvalue(ssim_delta, 4000 + len(condition)),
            }
        summary["conditions"][condition] = info
    (output / "lightweight_metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    visual_grid(args.run_root, names, records, output / "condition_comparison_first8.png")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
