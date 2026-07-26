#!/usr/bin/env python3
"""Launch exactly one E3 engineering-smoke image per category."""

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
    parser.add_argument("--gpus", default="2,3,4")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan["name"] not in {"E3_interaction", "E3_joint_redundancy"}:
        raise ValueError("This launcher only accepts a frozen E3 plan")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    launch_dir = args.plan.parent / f"{plan['name']}_smoke_specs"
    launch_dir.mkdir(parents=True, exist_ok=True)
    logs = args.project_root / "logs" / plan["name"] / "smoke"
    logs.mkdir(parents=True, exist_ok=True)
    tasks = []
    seed = int(plan["seeds"][0])
    for category, category_plan in plan["categories"].items():
        spec = launch_dir / f"{category}_conditions.json"
        indices = launch_dir / f"{category}_indices.json"
        spec.write_text(
            json.dumps(
                {
                    "experiment": plan["name"],
                    "category": category,
                    "conditions": category_plan["conditions"],
                    "matching_audit": category_plan["matching_audit"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        indices.write_text(
            json.dumps(category_plan["dataset_indices"][:1], indent=2),
            encoding="utf-8",
        )
        relative = Path(plan["name"]) / "smoke" / f"seed_{seed}" / category
        run_dir = args.output_root / relative
        if run_dir.exists():
            summary = run_dir / "run_summary.json"
            if summary.is_file():
                print(f"SKIP existing completed smoke: {relative}")
                continue
            raise RuntimeError(f"Existing incomplete smoke run: {run_dir}")
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
        tasks.append((category, command))
    for category, command in tasks:
        print(category, " ".join(command))
    if args.dry_run:
        return

    failures: list[dict[str, str | int]] = []
    lock = threading.Lock()

    def worker(gpu: str, category: str, command: list[str]) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        upstream = args.project_root / "code" / "NeuroAdapter"
        env["PYTHONPATH"] = f"{upstream}{os.pathsep}{Path.cwd() / 'src'}"
        log_path = logs / f"seed_{seed}_{category}.log"
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
                    {
                        "gpu": int(gpu),
                        "category": category,
                        "returncode": result.returncode,
                        "log": str(log_path),
                    }
                )

    threads = [
        threading.Thread(target=worker, args=(gpus[index % len(gpus)], *task))
        for index, task in enumerate(tasks)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    summary = {
        "experiment": plan["name"],
        "mode": "engineering_smoke_only",
        "seed": seed,
        "tasks": len(tasks),
        "failures": failures,
    }
    (logs / "launch_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} E3 smoke tasks failed")


if __name__ == "__main__":
    main()
