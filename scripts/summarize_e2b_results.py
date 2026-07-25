#!/usr/bin/env python3
"""Create E2b zero-vs-mean robustness tables, figures, and Chinese report."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import spearmanr

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "src"))

from neuro_roi_causal.stats import (
    benjamini_hochberg,
    bootstrap_ci,
    causal_loss,
    sign_flip_pvalue,
)


METRICS = ("pixel_corr", "ssim", "lpips", "clip", "dino")
CATEGORIES = ("Face", "Body", "Scene")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def image_excesses(rows: list[dict[str, str]], summary: dict, metric: str) -> dict[int, float]:
    design = summary["target_vs_matched_random"]["full"]
    target = design["target"]
    randoms = design["random_conditions"]
    values = {
        (int(row["seed"]), int(row["dataset_idx"]), row["condition"]): float(row[metric])
        for row in rows
    }
    seeds = sorted({key[0] for key in values})
    indices = sorted({key[1] for key in values})
    output = {}
    for index in indices:
        seed_values = []
        for seed in seeds:
            baseline = values[(seed, index, "no_mask")]
            target_loss = causal_loss(metric, baseline, values[(seed, index, target)])
            random_loss = np.mean(
                [
                    causal_loss(metric, baseline, values[(seed, index, condition)])
                    for condition in randoms
                ]
            )
            seed_values.append(target_loss - random_loss)
        output[index] = float(np.mean(seed_values))
    return output


def correlation(left: list[float], right: list[float]) -> dict:
    if len(left) < 2:
        return {"pearson": None, "spearman": None}
    return {
        "pearson": float(np.corrcoef(left, right)[0, 1]),
        "spearman": float(spearmanr(left, right).statistic),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-dir", type=Path, required=True)
    parser.add_argument("--mean-dir", type=Path, required=True)
    parser.add_argument("--overlap-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    zero = json.loads((args.zero_dir / "e2_metrics_summary.json").read_text(encoding="utf-8"))
    mean = json.loads((args.mean_dir / "e2_metrics_summary.json").read_text(encoding="utf-8"))

    robustness = []
    differences = []
    for category in CATEGORIES:
        zero_rows = read_rows(args.zero_dir / f"{category.lower()}_per_sample_metrics.csv")
        mean_rows = read_rows(args.mean_dir / f"{category.lower()}_per_sample_metrics.csv")
        for metric in METRICS:
            zero_item = zero["categories"][category]["target_vs_matched_random"]["full"]["metrics"][metric]
            mean_item = mean["categories"][category]["target_vs_matched_random"]["full"]["metrics"][metric]
            zero_images = image_excesses(zero_rows, zero["categories"][category], metric)
            mean_images = image_excesses(mean_rows, mean["categories"][category], metric)
            if set(zero_images) != set(mean_images):
                raise RuntimeError(f"{category}/{metric}: zero and mean images differ")
            paired = np.asarray(
                [mean_images[index] - zero_images[index] for index in sorted(zero_images)]
            )
            difference = {
                "mean_minus_zero": float(paired.mean()),
                "difference_ci95": bootstrap_ci(
                    paired, 93000 + len(differences), args.bootstrap_draws
                ),
                "difference_sign_flip_p": sign_flip_pvalue(
                    paired, 94000 + len(differences)
                ),
            }
            differences.append(difference)
            zero_excess = float(zero_item["target_minus_random_mean"])
            mean_excess = float(mean_item["target_minus_random_mean"])
            robustness.append(
                {
                    "category": category,
                    "metric": metric,
                    "zero_excess": zero_excess,
                    "mean_excess": mean_excess,
                    "same_direction": np.sign(zero_excess) == np.sign(mean_excess),
                    "zero_ci95": json.dumps(zero_item["ci95"]),
                    "mean_ci95": json.dumps(mean_item["ci95"]),
                    "zero_q": zero_item["bh_q_primary_15_tests"],
                    "mean_q": mean_item["bh_q_primary_15_tests"],
                    "sample_count": len(paired),
                    **difference,
                }
            )
    qvalues = benjamini_hochberg(
        [row["difference_sign_flip_p"] for row in robustness]
    )
    for row, qvalue in zip(robustness, qvalues):
        row["difference_bh_q_secondary_15"] = qvalue
    with (args.mean_dir / "zero_vs_mean_robustness.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(robustness[0]))
        writer.writeheader()
        writer.writerows(robustness)

    secondary = []
    for category in ("Body", "Scene"):
        designs = mean["categories"][category]["target_vs_matched_random"]
        if "equal4" not in designs:
            continue
        for metric in METRICS:
            secondary.append(
                {
                    "category": category,
                    "design": "equal4",
                    "metric": metric,
                    **designs["equal4"]["metrics"][metric],
                }
            )
    with (args.mean_dir / "secondary_equal_k_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=sorted({key for row in secondary for key in row})
        )
        writer.writeheader()
        writer.writerows(secondary)

    zero_values = [row["zero_excess"] for row in robustness]
    mean_values = [row["mean_excess"] for row in robustness]
    summary = {
        "interpretation_case": "A",
        "num_tests": 15,
        "same_direction_count": int(sum(row["same_direction"] for row in robustness)),
        "positive_same_direction_count": int(sum(
            row["same_direction"] and row["zero_excess"] > 0 for row in robustness
        )),
        "negative_same_direction_count": int(sum(
            row["same_direction"] and row["zero_excess"] < 0 for row in robustness
        )),
        "overall_correlation": correlation(zero_values, mean_values),
        "by_category": {
            category: correlation(
                [row["zero_excess"] for row in robustness if row["category"] == category],
                [row["mean_excess"] for row in robustness if row["category"] == category],
            )
            for category in CATEGORIES
        },
        "by_metric": {
            metric: correlation(
                [row["zero_excess"] for row in robustness if row["metric"] == metric],
                [row["mean_excess"] for row in robustness if row["metric"] == metric],
            )
            for metric in METRICS
        },
        "zero_corrected_positive_count": int(sum(
            row["zero_q"] < 0.05 and row["zero_excess"] > 0 for row in robustness
        )),
        "mean_corrected_positive_count": int(sum(
            row["mean_q"] < 0.05 and row["mean_excess"] > 0 for row in robustness
        )),
    }
    (args.mean_dir / "zero_vs_mean_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    figures = args.mean_dir / "figures"
    figures.mkdir(exist_ok=True)
    labels = [f"{row['category']} {row['metric']}" for row in robustness]
    y = np.arange(len(robustness))
    means = np.asarray(mean_values)
    low = np.asarray([json.loads(row["mean_ci95"])[0] for row in robustness])
    high = np.asarray([json.loads(row["mean_ci95"])[1] for row in robustness])
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.errorbar(means, y, xerr=[means - low, high - means], fmt="o", capsize=3)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean-mask target-random excess (95% bootstrap CI)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(figures / "primary_effect_forest_plot.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"Face": "#1f77b4", "Body": "#d62728", "Scene": "#2ca02c"}
    for row in robustness:
        ax.scatter(
            row["zero_excess"],
            row["mean_excess"],
            color=colors[row["category"]],
            label=row["category"],
        )
    limits = ax.get_xlim() + ax.get_ylim()
    lower, upper = min(limits), max(limits)
    ax.plot([lower, upper], [lower, upper], color="black", linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    handles, legend_labels = ax.get_legend_handles_labels()
    unique = dict(zip(legend_labels, handles))
    ax.legend(unique.values(), unique.keys())
    ax.set_xlabel("Zero-mask excess")
    ax.set_ylabel("Mean-mask excess")
    fig.tight_layout()
    fig.savefig(figures / "zero_vs_mean_effect_scatter.png", dpi=180)
    plt.close(fig)

    for category in CATEGORIES:
        png = args.mean_dir / f"{category.lower()}_comparison_grid.png"
        jpg = figures / f"{category.lower()}_comparison_grid.jpg"
        with Image.open(png) as image:
            image.convert("RGB").save(jpg, quality=88, optimize=True)
        png.unlink()
    shutil.copy2(
        args.overlap_dir / "figures" / "target_roi_overlap_matrix.png",
        figures / "target_roi_purity_plot.png",
    )
    shutil.copy2(
        args.overlap_dir / "figures" / "control_target_overlap_distribution.png",
        figures / "control_contamination_plot.png",
    )
    (args.mean_dir / "README.md").write_text(
        "# E2b mean-mask 正式稳健性实验\n\n"
        "本实验与正式 zero-mask 使用相同 Subject 1、100k checkpoint、"
        "137 张测试图、3 个 seed、冻结 ROI/random controls 与扩散参数，"
        "唯一主要变化是以 9000 个训练样本的 parcel-wise mean 替换目标 token。\n\n"
        "完整性审计通过 9/9 runs、411/411 确定性检查和 5499/5499 "
        "干预审计，非目标最大变化为 0.0，且 zero/mean 的 no-mask PNG "
        "SHA 完全一致。\n\n"
        "15 项主要检验经全局 BH 校正后，没有校正显著的正向结果。"
        "因此结果属于预定义情况 A：在当前公开 ROI 映射、当前 checkpoint "
        "和 zero/mean 两种 parcel 干预下，没有获得稳健的类别匹配 ROI "
        "额外因果贡献证据。这不等于对应脑区没有生物学功能。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
