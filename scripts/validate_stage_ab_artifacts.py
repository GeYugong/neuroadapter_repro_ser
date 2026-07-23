#!/usr/bin/env python3
"""Validate the committed E0/E1 artifacts without project dependencies."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_e0() -> None:
    root = ROOT / "experiments" / "E0_mapping"
    all_rows = read_csv(root / "algonauts_full_parcel_inventory_subj01.csv")
    top_rows = read_csv(root / "algonauts_top200_mapping_subj01.csv")
    metadata = json.loads((root / "mapping_metadata.json").read_text(encoding="utf-8"))
    hashes = json.loads((root / "file_hashes.json").read_text(encoding="utf-8"))

    assert len(all_rows) == metadata["num_all_parcels"] == 1000
    assert len(top_rows) == metadata["num_top_snr_parcels"] == 200
    assert sum(row["in_top_snr_200"].lower() == "true" for row in all_rows) == 200
    assert sorted(int(row["top200_token_index"]) for row in top_rows) == list(range(200))
    assert dict(Counter(row["dominant_roi"] for row in top_rows)) == metadata["top200_counts"]

    labeled_all = sum(row["dominant_roi"] != "Unlabeled" for row in all_rows)
    labeled_top = sum(row["dominant_roi"] != "Unlabeled" for row in top_rows)
    assert labeled_all == labeled_top == 97
    for row in all_rows:
        overlaps = json.loads(row["all_roi_overlaps"])
        assert all(
            {"collection", "roi", "group", "level", "overlap"} <= set(item)
            for item in overlaps
        )
        assert all(0.0 <= float(item["overlap"]) <= 1.0 for item in overlaps)

    coverage = read_csv(root / "roi_coverage_all_vs_top200.csv")
    assert all(
        float(row["selection_rate"]) == 1.0
        for row in coverage
        if row["roi_group"] != "Unlabeled"
    )
    generated_prefix = "generated:"
    generated = {
        value["path_role"][len(generated_prefix):]: value["sha256"]
        for value in hashes.values()
        if value["path_role"].startswith(generated_prefix)
    }
    for relative, expected in generated.items():
        assert sha256_file(root / relative) == expected


def validate_e1() -> None:
    root = ROOT / "experiments" / "E1_stimulus_manifest"
    all_images = read_csv(root / "all_test_images.csv")
    confirmatory = read_csv(root / "confirmatory_manifest.csv")
    exploratory = read_csv(root / "exploratory_manifest.csv")
    metadata = json.loads((root / "manifest_metadata.json").read_text(encoding="utf-8"))

    assert len(all_images) == metadata["num_test_images"] == 1000
    candidate_counts = {
        category: sum(
            row[f"{category.lower()}_flag"].lower() == "true" for row in all_images
        )
        for category in ("Face", "Body", "Scene", "Word")
    }
    assert candidate_counts == metadata["candidate_counts"]

    selected_counts = Counter(row["confirmatory_category"] for row in confirmatory)
    assert dict(selected_counts) == {
        category: metadata["selected_counts"][category]
        for category in ("Face", "Body", "Scene")
    }
    confirmatory_ids = {row["dataset_idx"] for row in confirmatory}
    exploratory_ids = {row["dataset_idx"] for row in exploratory}
    assert len(confirmatory_ids) == len(confirmatory) == 137
    assert len(exploratory_ids) == len(exploratory) == 21
    assert confirmatory_ids.isdisjoint(exploratory_ids)
    assert metadata["word_confirmatory"] is False
    assert metadata["selection_uses_reconstruction_outputs"] is False

    for relative, expected in metadata["output_hashes"].items():
        assert sha256_file(root / relative) == expected


def main() -> None:
    validate_e0()
    validate_e1()
    print("E0 PASS: 1000 parcels, 200 selected, all 97 labeled parcels retained")
    print("E1 PASS: Face=37, Body=50, Scene=50, Word exploratory=21")
    print("Stage A/B artifact validation: PASS")


if __name__ == "__main__":
    main()
