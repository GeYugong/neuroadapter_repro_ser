#!/usr/bin/env python3
"""Create the preregistered primary ROI-ablation condition specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi-groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--without-mean", action="store_true", help="Omit mean-mask robustness conditions")
    args = parser.parse_args()
    group_data = json.loads(args.roi_groups.read_text(encoding="utf-8"))
    groups = group_data["groups"]
    primary = ["V1", "V2", "V3", "V4", "Face", "Body", "Scene", "Word"]
    conditions = [{"name": "no_mask", "masked_token_indices": [], "mask_mode": "zero"}]
    for name in ("low_level", "high_level"):
        conditions.append({"name": f"{name}_zero", "masked_token_indices": groups[name], "mask_mode": "zero"})
        if not args.without_mean:
            conditions.append({"name": f"{name}_mean", "masked_token_indices": groups[name], "mask_mode": "mean"})
    for name in primary:
        conditions.append({"name": f"{name}_zero", "masked_token_indices": groups[name], "mask_mode": "zero"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "description": "Primary Subject 1 ROI ablations; source mapping uses >50% parcel overlap.",
        "mapping_file": str(args.roi_groups),
        "conditions": conditions,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
