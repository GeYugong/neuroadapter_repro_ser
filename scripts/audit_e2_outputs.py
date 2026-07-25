#!/usr/bin/env python3
"""Independently audit E2 output completeness and pairing invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "src"))

from neuro_roi_causal.e2_audit import no_mask_record_failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--zero-run-root",
        type=Path,
        help="When auditing mean outputs, require cross-experiment no-mask SHA equality.",
    )
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    failures = []
    run_results = []
    total_determinism = 0
    total_audits = 0
    max_non_target_delta = 0.0
    checkpoint_hashes = set()
    mean_cache_hashes = set()
    repository_commits = set()

    for seed in plan["seeds"]:
        for category, category_plan in plan["categories"].items():
            category_root = args.run_root / f"seed_{seed}" / category
            run_summary_path = category_root / "run_summary.json"
            if not run_summary_path.exists():
                failures.append(f"Missing {run_summary_path}")
                continue
            run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
            expected_indices = list(map(int, category_plan["dataset_indices"]))
            expected_names = {
                condition["name"] for condition in category_plan["conditions"]
            }
            if run_summary["dataset_indices"] != expected_indices:
                failures.append(f"{seed}/{category}: dataset indices differ from plan")
            actual_names = {
                condition["name"] for condition in run_summary["conditions"]
            }
            if actual_names != expected_names:
                failures.append(f"{seed}/{category}: condition names differ from plan")
            if run_summary["conditions"] != category_plan["conditions"]:
                failures.append(
                    f"{seed}/{category}: condition order, indices, or metadata differ from plan"
                )
            checkpoint_hashes.add(run_summary.get("checkpoint_sha256"))
            repository_commits.add(run_summary.get("repository_commit"))
            if plan.get("mean_token_cache_sha256"):
                mean_cache_hashes.add(run_summary.get("mean_token_cache_sha256"))
                if (
                    run_summary.get("mean_token_cache_sha256")
                    != plan["mean_token_cache_sha256"]
                ):
                    failures.append(f"{seed}/{category}: mean cache hash differs from plan")
            checks = run_summary.get("determinism_checks", [])
            if len(checks) != len(expected_indices) or not all(
                check.get("passed") for check in checks
            ):
                failures.append(f"{seed}/{category}: determinism checks incomplete")
            total_determinism += sum(bool(check.get("passed")) for check in checks)
            if len(run_summary.get("shared_diffusion_state", [])) != len(expected_indices):
                failures.append(f"{seed}/{category}: shared-state audits incomplete")

            category_audits = 0
            for name in expected_names:
                summary_path = category_root / name / "summary.json"
                if not summary_path.exists():
                    failures.append(f"Missing {summary_path}")
                    continue
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                records = summary["records"]
                if [int(row["dataset_idx"]) for row in records] != expected_indices:
                    failures.append(f"{seed}/{category}/{name}: records differ from plan")
                for record in records:
                    for key in ("gt", "pred", "comparison"):
                        if not Path(record[key]).exists():
                            failures.append(
                                f"{seed}/{category}/{name}: missing {key} file "
                                f"for {record['dataset_idx']}"
                            )
                for audit in summary["intervention_audits"]:
                    category_audits += 1
                    total_audits += 1
                    delta = float(audit["max_abs_delta_non_target"])
                    max_non_target_delta = max(max_non_target_delta, delta)
                    if delta != 0.0:
                        failures.append(
                            f"{seed}/{category}/{name}: non-target delta {delta}"
                        )
                    if audit["changed_indices"] != audit["masked_indices"]:
                        failures.append(
                            f"{seed}/{category}/{name}: changed indices differ"
                        )
                    if audit["mode"] == "mean" and audit["masked_indices"]:
                        if not audit["masked_norm_after"] or all(
                            float(value) == 0.0 for value in audit["masked_norm_after"]
                        ):
                            failures.append(
                                f"{seed}/{category}/{name}: mean target became all zero"
                            )
            if args.zero_run_root is not None:
                for name in ("no_mask", "no_mask_repeat"):
                    mean_summary = json.loads(
                        (category_root / name / "summary.json").read_text(encoding="utf-8")
                    )
                    zero_summary = json.loads(
                        (
                            args.zero_run_root
                            / f"seed_{seed}"
                            / category
                            / name
                            / "summary.json"
                        ).read_text(encoding="utf-8")
                    )
                    failures.extend(
                        f"{seed}/{category}/{name}: {failure}"
                        for failure in no_mask_record_failures(
                            mean_summary["records"], zero_summary["records"]
                        )
                    )
            expected_audits = len(expected_indices) * len(expected_names)
            if category_audits != expected_audits:
                failures.append(
                    f"{seed}/{category}: {category_audits} audits, "
                    f"expected {expected_audits}"
                )
            run_results.append(
                {
                    "seed": seed,
                    "category": category,
                    "images": len(expected_indices),
                    "conditions": len(expected_names),
                    "elapsed_sec": float(run_summary["elapsed_sec"]),
                    "determinism_checks_passed": sum(
                        bool(check.get("passed")) for check in checks
                    ),
                    "intervention_audits": category_audits,
                }
            )

    result = {
        "run_root": str(args.run_root),
        "plan": str(args.plan),
        "passed": not failures,
        "runs": run_results,
        "total_runs": len(run_results),
        "total_image_seed_pairs": sum(run["images"] for run in run_results),
        "total_determinism_checks_passed": total_determinism,
        "total_intervention_audits": total_audits,
        "max_abs_delta_non_target": max_non_target_delta,
        "checkpoint_hashes": sorted(str(value) for value in checkpoint_hashes),
        "mean_cache_hashes": sorted(str(value) for value in mean_cache_hashes),
        "repository_commits": sorted(str(value) for value in repository_commits),
        "failures": failures,
    }
    if len(checkpoint_hashes) != 1:
        failures.append("Runs do not share one checkpoint hash")
    if plan.get("mean_token_cache_sha256") and len(mean_cache_hashes) != 1:
        failures.append("Runs do not share one mean cache hash")
    if len(repository_commits) != 1:
        failures.append("Runs do not share one repository commit")
    result["passed"] = not failures
    output = args.output or args.run_root / "e2_output_audit.json"
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
