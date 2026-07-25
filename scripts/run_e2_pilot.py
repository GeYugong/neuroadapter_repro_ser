#!/usr/bin/env python3
"""Plan and optionally execute the manifest-aware E2 causal pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "src"))

from neuro_roi_causal.e2 import (
    build_category_conditions,
    read_csv,
    select_manifest_indices,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPRO_ROOT / path


def build_plan(config: dict) -> dict:
    manifest_path = resolve_repo_path(config["stimulus_manifest"])
    inventory_path = resolve_repo_path(config["mapping_inventory"])
    manifest = read_csv(manifest_path)
    inventory = read_csv(inventory_path)
    category_counts = {
        category: int(count)
        for category, count in config["categories"].items()
    }
    selected = select_manifest_indices(
        manifest,
        category_counts,
        index_column=config.get("dataset_index_column", "dataset_idx"),
    )
    category_plans = {}
    for category_index, category in enumerate(category_counts):
        conditions, matching_audit = build_category_conditions(
            inventory,
            category,
            mask_modes=config["mask_modes"],
            random_replicates=int(config["matched_random_controls"]),
            equal_k=int(config.get("equal_k", 4)),
            seed=int(config.get("control_seed", 20260718)) + category_index * 10000,
        )
        category_plans[category] = {
            "dataset_indices": selected[category],
            "conditions": conditions,
            "matching_audit": matching_audit,
        }
    return {
        "name": config["name"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "subject": int(config["subject"]),
        "checkpoint_step": int(config["checkpoint_step"]),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "mapping_inventory": str(inventory_path),
        "mapping_inventory_sha256": file_sha256(inventory_path),
        "seeds": [int(seed) for seed in config["seeds"]],
        "denoising_steps": int(config["denoising_steps"]),
        "noise_factor": float(config["noise_factor"]),
        "condition_batch_size": int(config.get("condition_batch_size", 8)),
        "categories": category_plans,
    }


def write_plan_files(plan: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "e2_pilot_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for category, category_plan in plan["categories"].items():
        (output_dir / f"{category.lower()}_indices.json").write_text(
            json.dumps(
                {"category": category, "dataset_indices": category_plan["dataset_indices"]},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (output_dir / f"{category.lower()}_conditions.json").write_text(
            json.dumps(
                {
                    "category": category,
                    "conditions": category_plan["conditions"],
                    "matching_audit": category_plan["matching_audit"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def execute_plan(
    plan: dict,
    output_dir: Path,
    project_root: Path,
    checkpoint: Path,
    mean_token_path: Path | None,
) -> None:
    decoder = REPRO_ROOT / "scripts" / "decode_roi_ablation_batch.py"
    for seed in plan["seeds"]:
        for category in plan["categories"]:
            command = [
                sys.executable,
                str(decoder),
                "--checkpoint",
                str(checkpoint),
                "--condition-spec",
                str(output_dir / f"{category.lower()}_conditions.json"),
                "--dataset-indices-file",
                str(output_dir / f"{category.lower()}_indices.json"),
                "--run-name",
                f"{plan['name']}/seed_{seed}/{category}",
                "--seed",
                str(seed),
                "--denoising-steps",
                str(plan["denoising_steps"]),
                "--noise-factor",
                str(plan["noise_factor"]),
                "--condition-batch-size",
                str(plan["condition_batch_size"]),
                "--project-root",
                str(project_root),
            ]
            if any(
                condition["mask_mode"] == "mean"
                for condition in plan["categories"][category]["conditions"]
            ):
                if mean_token_path is None:
                    raise ValueError(
                        f"{category} contains mean conditions but no mean-token path"
                    )
                command.extend(["--mean-token-path", str(mean_token_path)])
            subprocess.run(command, cwd=REPRO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPRO_ROOT / "configs" / "experiments" / "E2_pilot.yaml",
    )
    parser.add_argument(
        "--plan-output-dir",
        type=Path,
        default=REPRO_ROOT / ".audit" / "E2_pilot",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(os.environ.get("NEUROADAPTER_PROJECT_ROOT", REPRO_ROOT.parent)),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mean-token-path", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    plan = build_plan(config)
    write_plan_files(plan, args.plan_output_dir)
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if not args.execute:
        return
    checkpoint = args.checkpoint
    if checkpoint is None and config.get("checkpoint"):
        checkpoint = args.project_root / config["checkpoint"]
    mean_token_path = args.mean_token_path
    if mean_token_path is None and config.get("mean_token_path"):
        mean_token_path = args.project_root / config["mean_token_path"]
    if checkpoint is None:
        raise ValueError("--execute requires a checkpoint path from CLI or config")
    execute_plan(
        plan,
        args.plan_output_dir,
        args.project_root.resolve(),
        checkpoint.resolve(),
        mean_token_path.resolve() if mean_token_path is not None else None,
    )


if __name__ == "__main__":
    main()
