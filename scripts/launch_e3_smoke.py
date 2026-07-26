#!/usr/bin/env python3
"""Launch a frozen E3 smoke or one-seed pilot plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import threading
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        choices=("smoke", "pilot", "formal"),
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
    for label, path, plan_path_key, plan_hash_key in (
        (
            "checkpoint",
            args.checkpoint,
            "checkpoint",
            "checkpoint_sha256",
        ),
        (
            "mean cache",
            args.mean_cache,
            "mean_token_cache",
            "mean_token_cache_sha256",
        ),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Frozen {label} is missing: {path}")
        if str(path.resolve()) != plan.get(plan_path_key):
            raise ValueError(f"{label} path differs from the frozen plan")
        if file_sha256(path) != plan.get(plan_hash_key):
            raise ValueError(f"{label} SHA-256 differs from the frozen plan")
    image_counts = {
        category: (
            len(category_plan["dataset_indices"])
            if args.mode == "formal"
            else int(plan["execution"][f"{args.mode}_images_per_category"])
        )
        for category, category_plan in plan["categories"].items()
    }
    seed_count = (
        len(plan["seeds"])
        if args.mode == "formal"
        else int(plan["execution"][f"{args.mode}_seeds"])
    )
    image_count_values = set(image_counts.values())
    if args.mode == "smoke" and (
        image_count_values != {1} or seed_count != 1
    ):
        raise ValueError("The engineering smoke must remain one image and one seed")
    if args.mode == "pilot":
        if plan.get("plan_scope") != "pilot":
            raise ValueError("Pilot launcher requires an independent pilot plan")
        if image_count_values != {10} or seed_count != 1:
            raise ValueError("E3 pilot must remain ten images and one seed")
        if len(plan["seeds"]) != 1:
            raise ValueError("Independent pilot plan must freeze exactly one seed")
    if args.mode == "formal":
        if plan.get("plan_scope") != "formal":
            raise ValueError("Formal launcher requires a formal frozen plan")
        if image_counts != {"Face": 37, "Body": 50, "Scene": 50}:
            raise ValueError(
                f"Formal image counts must be 37/50/50: {image_counts}"
            )
        if [int(seed) for seed in plan["seeds"]] != [12345, 23456, 34567]:
            raise ValueError("Formal seeds differ from the preregistered set")
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
            image_count = image_counts[category]
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

    task_queue: queue.Queue[tuple[int, str, list[str]]] = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    def gpu_worker(gpu: str) -> None:
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return
            try:
                worker(gpu, *task)
            finally:
                task_queue.task_done()

    threads = [
        threading.Thread(target=gpu_worker, args=(gpu,))
        for gpu in gpus[: min(len(gpus), len(tasks))]
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
        "images_per_category": image_counts,
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
