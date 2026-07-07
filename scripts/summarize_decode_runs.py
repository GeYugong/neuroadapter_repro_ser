#!/usr/bin/env python3
"""Summarize NeuroAdapter brain-encoder selection JSON outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    best_scores = []
    best_indices = []
    for record in data["records"]:
        best_idx = int(record["best_candidate_index"])
        best_indices.append(best_idx)
        best_scores.append(float(record["scores"][best_idx]["mean"]))

    return {
        "run_name": data["run_name"],
        "checkpoint_step": data.get("checkpoint_step"),
        "num_samples": data["num_samples"],
        "elapsed_sec": float(data["elapsed_sec"]),
        "positive_count": sum(score > 0 for score in best_scores),
        "positive_rate": sum(score > 0 for score in best_scores) / len(best_scores),
        "mean_best_score": sum(best_scores) / len(best_scores),
        "min_best_score": min(best_scores),
        "max_best_score": max(best_scores),
        "first10_best_scores": best_scores[:10],
        "best_indices": best_indices,
        "summary_path": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    rows = [summarize(path) for path in args.summaries]

    print("| step | positive | mean | min | max | elapsed_min | run |")
    print("|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        print(
            f"| {row['checkpoint_step']} | "
            f"{row['positive_count']}/{row['num_samples']} | "
            f"{row['mean_best_score']:.4f} | "
            f"{row['min_best_score']:.4f} | "
            f"{row['max_best_score']:.4f} | "
            f"{row['elapsed_sec'] / 60:.2f} | "
            f"{row['run_name']} |"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
