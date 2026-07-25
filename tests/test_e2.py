from pathlib import Path

import pytest

from neuro_roi_causal.e2 import (
    build_category_conditions,
    read_csv,
    select_manifest_indices,
)


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = (
    ROOT
    / "experiments"
    / "E0_mapping"
    / "algonauts_top200_mapping_subj01.csv"
)
MANIFEST = (
    ROOT
    / "experiments"
    / "E1_stimulus_manifest"
    / "confirmatory_manifest.csv"
)


def test_manifest_selection_supports_non_contiguous_indices():
    selected = select_manifest_indices(
        read_csv(MANIFEST),
        {"Face": 10, "Body": 10, "Scene": 10},
    )
    assert {category: len(indices) for category, indices in selected.items()} == {
        "Face": 10,
        "Body": 10,
        "Scene": 10,
    }
    assert selected["Face"][:3] == [973, 756, 287]
    assert any(
        right != left + 1
        for indices in selected.values()
        for left, right in zip(indices, indices[1:])
    )
    assert len({index for indices in selected.values() for index in indices}) == 30


@pytest.mark.parametrize(
    ("category", "full_count", "condition_count"),
    [("Face", 4, 9), ("Body", 24, 15), ("Scene", 31, 15)],
)
def test_e2_conditions_are_complete_and_matched(
    category,
    full_count,
    condition_count,
):
    inventory = read_csv(INVENTORY)
    by_index = {int(row["top200_token_index"]): row for row in inventory}
    conditions, audit = build_category_conditions(
        inventory,
        category,
        mask_modes=["zero"],
        random_replicates=5,
        equal_k=4,
        seed=20260718,
    )
    assert audit["full_target_count"] == full_count
    assert len(conditions) == condition_count
    assert conditions[0]["name"] == "no_mask"
    assert conditions[-1]["name"] == "no_mask_repeat"
    assert len(conditions) > 8

    names = {condition["name"] for condition in conditions}
    assert len(names) == len(conditions)
    assert f"{category}_full_zero" in names
    assert not any(condition["mask_mode"] == "mean" for condition in conditions)

    for design in audit["designs"].values():
        target_indices = set(design["target_indices"])
        target_hemispheres = sorted(
            by_index[index]["hemisphere"] for index in target_indices
        )
        assert len(design["random_controls"]) == 5
        for control in design["random_controls"]:
            assert len(control["indices"]) == len(target_indices)
            assert not target_indices.intersection(control["indices"])
            assert sorted(
                by_index[index]["hemisphere"] for index in control["indices"]
            ) == target_hemispheres
            assert all(
                by_index[index]["dominant_roi"] != category
                for index in control["indices"]
            )
            assert all("control_roi" in pair for pair in control["pairs"])
            assert control["mean_distance"] >= 0
            assert control["max_distance"] >= control["mean_distance"]
