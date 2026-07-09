#!/usr/bin/env python3
"""Convert NeuroAdapter decode summaries to metric_brain_adapter.py input format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def load_rgb(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def score_means(record: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(item["mean"]) for item in record["scores"]], dtype=np.float32)


def convert_summary(summary_path: Path, output_dir: Path) -> dict[str, Any]:
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_indices: list[int] = []
    samples: list[dict[str, Any]] = []

    for record in summary["records"]:
        dataset_idx = int(record["dataset_idx"])
        dataset_indices.append(dataset_idx)

        gt_image = load_rgb(record["gt"])
        candidate_images = np.stack([load_rgb(path) for path in record["candidate_paths"]], axis=0)
        correlation_scores = score_means(record)
        best_idx = int(record["best_candidate_index"])

        sample_path = output_dir / f"sample_{dataset_idx:06d}.npz"
        np.savez_compressed(
            sample_path,
            groundtruth_image=gt_image,
            candidate_images=candidate_images,
            correlation_scores=correlation_scores,
            predicted_image=candidate_images[best_idx],
            best_candidate_index=np.asarray(best_idx, dtype=np.int64),
            dataset_idx=np.asarray(dataset_idx, dtype=np.int64),
        )

        samples.append(
            {
                "dataset_idx": dataset_idx,
                "sample_file": sample_path.name,
                "groundtruth_path": record["gt"],
                "candidate_paths": record["candidate_paths"],
                "best_candidate_index": best_idx,
                "best_candidate_path": record["best_candidate_path"],
                "best_candidate_score": float(correlation_scores[best_idx]),
            }
        )

    metadata = {
        "source_summary": str(summary_path),
        "source_run_name": summary.get("run_name"),
        "checkpoint": summary.get("checkpoint"),
        "checkpoint_step": summary.get("checkpoint_step"),
        "topk": summary.get("topk"),
        "num_predictions": summary.get("num_predictions"),
        "denoising_steps": summary.get("denoising_steps"),
        "selection_metric": summary.get("brain_encoder"),
        "format": "metric_brain_adapter.py compatible npz",
    }
    sample_summary = {
        "mode": "subset",
        "num_samples": len(dataset_indices),
        "dataset_indices": dataset_indices,
        "samples": samples,
    }

    with (output_dir / "evaluation_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    with (output_dir / "sample_summary.json").open("w", encoding="utf-8") as f:
        json.dump(sample_summary, f, indent=2, ensure_ascii=False)

    return {
        "output_dir": str(output_dir),
        "num_samples": len(dataset_indices),
        "first_dataset_idx": dataset_indices[0] if dataset_indices else None,
        "last_dataset_idx": dataset_indices[-1] if dataset_indices else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path, help="Path to decode summary.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    result = convert_summary(args.summary, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
