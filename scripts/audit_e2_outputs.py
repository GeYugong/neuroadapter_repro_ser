#!/usr/bin/env python3
"""Independently audit E2 output completeness and pairing invariants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    failures = []
    run_results = []
    total_determinism = 0
    total_audits = 0
    max_non_target_delta = 0.0
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
        "failures": failures,
    }
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
