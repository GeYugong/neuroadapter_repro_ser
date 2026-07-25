#!/usr/bin/env python3
"""Create a frozen E2b mean plan from the completed zero plan."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "src"))

from neuro_roi_causal.mean_cache import file_sha256
from neuro_roi_causal.mean_plan import convert_zero_plan, equivalence_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-plan", type=Path, required=True)
    parser.add_argument("--mean-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    zero = json.loads(args.zero_plan.read_text(encoding="utf-8"))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPRO_ROOT, text=True
    ).strip()
    mean = convert_zero_plan(
        zero,
        source_zero_plan=str(args.zero_plan.resolve()),
        source_zero_plan_sha256=file_sha256(args.zero_plan),
        mean_token_cache=str(args.mean_cache.resolve()),
        mean_token_cache_sha256=file_sha256(args.mean_cache),
        repository_commit=commit,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    audit = equivalence_audit(zero, mean)
    if not audit["passed"]:
        raise RuntimeError(f"Zero/mean plan equivalence failed: {audit}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(mean, indent=2), encoding="utf-8")
    args.audit_output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
