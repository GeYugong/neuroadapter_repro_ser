#!/usr/bin/env python3
"""Compute lightweight image metrics for current PNG-based decode outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover
    structural_similarity = None


def load_rgb(path: Path, size: tuple[int, int] = (425, 425)) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    x = a.reshape(-1).astype(np.float64)
    y = b.reshape(-1).astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return 0.0
    return float(np.dot(x, y) / denom)


def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float | None:
    if structural_similarity is None:
        return None
    return float(
        structural_similarity(
            a,
            b,
            channel_axis=2,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=1.0,
        )
    )


def evaluate(summary_path: Path) -> dict:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    per_sample = []
    for record in summary["records"]:
        best_idx = int(record["best_candidate_index"])
        gt = load_rgb(Path(record["gt"]))
        pred = load_rgb(Path(record["candidate_paths"][best_idx]))
        pixel_corr = pearson(gt, pred)
        ssim_value = ssim_rgb(gt, pred)
        per_sample.append(
            {
                "dataset_idx": int(record["dataset_idx"]),
                "best_candidate_index": best_idx,
                "brain_encoder_score": float(record["scores"][best_idx]["mean"]),
                "pixel_corr": pixel_corr,
                "ssim": ssim_value,
            }
        )

    pixel_values = [row["pixel_corr"] for row in per_sample]
    ssim_values = [row["ssim"] for row in per_sample if row["ssim"] is not None]
    brain_values = [row["brain_encoder_score"] for row in per_sample]
    return {
        "run_name": summary["run_name"],
        "checkpoint_step": summary["checkpoint_step"],
        "summary_path": str(summary_path),
        "num_samples": len(per_sample),
        "brain_encoder_mean": float(np.mean(brain_values)),
        "brain_encoder_positive": int(sum(value > 0 for value in brain_values)),
        "pixel_corr_mean": float(np.mean(pixel_values)),
        "pixel_corr_std": float(np.std(pixel_values)),
        "ssim_mean": float(np.mean(ssim_values)) if ssim_values else None,
        "ssim_std": float(np.std(ssim_values)) if ssim_values else None,
        "per_sample": per_sample,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    results = [evaluate(path) for path in args.summaries]
    print("| step | brain positive | brain mean | pixel corr | SSIM | run |")
    print("|---:|---:|---:|---:|---:|---|")
    for result in results:
        ssim_text = "n/a" if result["ssim_mean"] is None else f"{result['ssim_mean']:.4f}"
        print(
            f"| {result['checkpoint_step']} | "
            f"{result['brain_encoder_positive']}/{result['num_samples']} | "
            f"{result['brain_encoder_mean']:.4f} | "
            f"{result['pixel_corr_mean']:.4f} | "
            f"{ssim_text} | "
            f"{result['run_name']} |"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
