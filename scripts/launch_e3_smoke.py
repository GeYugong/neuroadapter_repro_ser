#!/usr/bin/env python3
"""Launch a frozen E3 smoke or one-seed pilot plan."""

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
    parser.add_argument(
        "--mode",
        choices=("smoke", "pilot"),
        default="smoke",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Output label below the experiment directory.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan["name"] not in {"E3_interaction", "E3_joint_redundancy"}:
        raise ValueError("This launcher only accepts a frozen E3 plan")
    image_count = int(
        plan["execution"][f"{args.mode}_images_per_category"]
    )
    seed_count = int(plan["execution"][f"{args.mode}_seeds"])
    if args.mode == "smoke" and (image_count != 1 or seed_count != 1):
        raise ValueError("The engineering smoke must remain one image and one seed")
    if args.mode == "pilot":
        if plan.get("plan_scope") != "pilot":
            raise ValueError("Pilot launcher requires an independent pilot plan")
        if image_count != 10 or seed_count != 1:
            raise ValueError("E3 pilot must remain ten images and one seed")
        if len(plan["seeds"]) != 1:
            raise ValueError("Independent pilot plan must freeze exactly one seed")
    run_label = args.run_label or args.mode
    if Path(run_label).name != run_label:
        raise ValueError("--run-label must be one safe path component")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU is required")
    launch_dir = args.plan.parent / f"{plan['name']}_{run_label}_specs"
    launch_dir.mkdir(parents=True, exist_ok=True)
    logs = args.project_root / "logs" / plan["name"] / run_label
    logs.mkdir(parents=True, exist_ok=True)
    tasks = []
    selected_seeds = [int(seed) for seed in plan["seeds"][:seed_count]]
    for seed in selected_seeds:
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
            selected_indices = category_plan["dataset_indices"][:image_count]
            if len(selected_indices) != image_count:
                raise ValueError(
                    f"{category} plan has fewer than {image_count} images"
                )
            indices.write_text(
                json.dumps(selected_indices, indent=2),
                encoding="utf-8",
            )
            relative = (
                Path(plan["name"])
                / run_label
                / f"seed_{seed}"
                / category
            )
            run_dir = args.output_root / relative
            if run_dir.exists():
                summary = run_dir / "run_summary.json"
                if summary.is_file():
                    print(f"SKIP existing completed run: {relative}")
                    continue
                raise RuntimeError(f"Existing incomplete run: {run_dir}")
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
    for seed, category, command in tasks:
        print(seed, category, " ".join(command))
    if args.dry_run:
        return

    failures: list[dict[str, str | int]] = []
    lock = threading.Lock()

    def worker(
        gpu: str,
        seed: int,
        category: str,
        command: list[str],
    ) -> None:
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
        threading.Thread(
            target=worker,
            args=(gpus[index % len(gpus)], *task),
        )
        for index, task in enumerate(tasks)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    summary = {
        "experiment": plan["name"],
        "mode": f"engineering_{args.mode}_only",
        "run_label": run_label,
        "seeds": selected_seeds,
        "images_per_category": image_count,
        "tasks": len(tasks),
        "failures": failures,
    }
    (logs / "launch_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if failures:
        raise RuntimeError(f"{len(failures)} E3 {args.mode} tasks failed")


if __name__ == "__main__":
    main()
