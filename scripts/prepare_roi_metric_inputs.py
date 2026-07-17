#!/usr/bin/env python3
"""Convert one-condition ROI-ablation summaries to official metric input NPZs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    indices = []
    for record in summary["records"]:
        index = int(record["dataset_idx"])
        gt, pred = load_image(record["gt"]), load_image(record["pred"])
        np.savez_compressed(
            args.output_dir / f"sample_{index:06d}.npz",
            groundtruth_image=gt,
            candidate_images=pred[None, ...],
            correlation_scores=np.asarray([0.0], dtype=np.float32),
            predicted_image=pred,
            best_candidate_index=np.asarray(0, dtype=np.int64),
            dataset_idx=np.asarray(index, dtype=np.int64),
        )
        indices.append(index)
    metadata = {"source_summary": str(args.summary), "source_run_name": summary["run_name"], "checkpoint": summary["checkpoint"], "checkpoint_step": summary["checkpoint_step"], "format": "ROI ablation single-prediction metric input"}
    sample_summary = {"mode": "subset", "num_samples": len(indices), "dataset_indices": indices, "samples": [{"dataset_idx": index, "sample_file": f"sample_{index:06d}.npz"} for index in indices]}
    (args.output_dir / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "sample_summary.json").write_text(json.dumps(sample_summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
