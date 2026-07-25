#!/usr/bin/env python3
"""Launch E2b category-seed jobs with one sequential queue per GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mean-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def completed(run_dir: Path, expected_samples: int, expected_conditions: int) -> bool:
    summary = run_dir / "run_summary.json"
    if not summary.exists():
        return False
    data = json.loads(summary.read_text(encoding="utf-8"))
    return (
        int(data["num_samples"]) == expected_samples
        and len(data["conditions"]) == expected_conditions
        and len(data.get("determinism_checks", [])) == expected_samples
        and all(item.get("passed") for item in data.get("determinism_checks", []))
    )


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    launch_dir = args.plan.parent / f"launch_{args.mode}"
    launch_dir.mkdir(parents=True, exist_ok=True)
    logs = args.project_root / "logs" / args.experiment_name
    logs.mkdir(parents=True, exist_ok=True)
    tasks = []
    seeds = [plan["seeds"][0]] if args.mode == "smoke" else plan["seeds"]
    for seed in seeds:
        for category, category_plan in plan["categories"].items():
            spec = launch_dir / f"{category}_conditions.json"
            indices = launch_dir / f"{category}_indices.json"
            spec.write_text(
                json.dumps(
                    {
                        "category": category,
                        "conditions": category_plan["conditions"],
                        "matching_audit": category_plan["matching_audit"],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            selected = category_plan["dataset_indices"][:1] if args.mode == "smoke" else category_plan["dataset_indices"]
            indices.write_text(json.dumps(selected, indent=2), encoding="utf-8")
            relative = Path(args.experiment_name) / f"seed_{seed}" / category
            run_dir = args.output_root / relative
            expected_samples = len(selected)
            if run_dir.exists():
                if completed(run_dir, expected_samples, len(category_plan["conditions"])):
                    print(f"SKIP complete: {relative}")
                    continue
                raise RuntimeError(f"Incomplete existing run; refusing to delete: {run_dir}")
            command = [
                "conda",
                "run",
                "-n",
                "neuroadapter",
                "python",
                "scripts/decode_roi_ablation_batch.py",
                "--checkpoint",
                str(args.checkpoint),
                "--condition-spec",
                str(spec),
                "--dataset-indices-file",
                str(indices),
                "--run-name",
                str(relative),
                "--seed",
                str(seed),
                "--denoising-steps",
                str(plan["denoising_steps"]),
                "--noise-factor",
                str(plan["noise_factor"]),
                "--condition-batch-size",
                str(plan["condition_batch_size"]),
                "--mean-token-path",
                str(args.mean_cache),
                "--project-root",
                str(args.project_root),
                "--output-root",
                str(args.output_root),
            ]
            tasks.append((seed, category, command))
    for number, (seed, category, command) in enumerate(tasks):
        print(f"TASK {number + 1}: seed={seed} category={category}")
        print(" ".join(command))
    if args.dry_run:
        return

    queues = {gpu: [] for gpu in gpus}
    for index, task in enumerate(tasks):
        queues[gpus[index % len(gpus)]].append(task)
    failures = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        for seed, category, command in queues[gpu]:
            log_path = logs / f"seed_{seed}_{category}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            upstream = args.project_root / "code" / "NeuroAdapter"
            env["PYTHONPATH"] = f"{upstream}{os.pathsep}{Path.cwd() / 'src'}"
            with log_path.open("w", encoding="utf-8") as handle:
                result = subprocess.run(
                    command,
                    cwd=Path.cwd(),
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if result.returncode:
                with lock:
                    failures.append(
                        {"gpu": gpu, "seed": seed, "category": category, "log": str(log_path)}
                    )
                return

    threads = [
        threading.Thread(target=worker, args=(gpu,), name=f"gpu-{gpu}")
        for gpu in gpus
        if queues[gpu]
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    summary = {
        "experiment": args.experiment_name,
        "mode": args.mode,
        "tasks_requested": len(tasks),
        "gpus": gpus,
        "failures": failures,
    }
    summary_path = logs / "launch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} E2b tasks failed")


if __name__ == "__main__":
    main()
