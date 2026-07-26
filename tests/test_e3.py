from pathlib import Path

from neuro_roi_causal.e2 import read_csv
from neuro_roi_causal.e3 import (
    ROI_GROUPS,
    build_interaction_category,
    build_joint_category,
    max_overlap,
)
from neuro_roi_causal.stats import causal_loss


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "experiments" / "E0_mapping" / "algonauts_top200_mapping_subj01.csv"


def assert_controls_are_pure_and_matched(audit, design_key):
    inventory = read_csv(INVENTORY)
    by_index = {int(row["top200_token_index"]): row for row in inventory}
    for label, design in audit[design_key].items():
        targets = design["target_indices"]
        target_hemispheres = sorted(by_index[index]["hemisphere"] for index in targets)
        controls = design["pure_random_controls"]
        assert len(controls) == 5
        assert len({tuple(control["indices"]) for control in controls}) == 5
        for control in controls:
            assert len(control["indices"]) == len(targets)
            assert control["max_target_group_overlap"] < 0.10
            assert sorted(
                by_index[index]["hemisphere"] for index in control["indices"]
            ) == target_hemispheres
            assert all(
                max_overlap(by_index[index], control["target_groups"]) < 0.10
                for index in control["indices"]
            )


def test_e3a_is_complete_three_by_three_design():
    inventory = read_csv(INVENTORY)
    conditions, audit = build_interaction_category(
        inventory,
        "Face",
        k=4,
        replicates=5,
        seed=20260726,
        purity_threshold=0.10,
    )
    assert len(conditions) == 20
    assert conditions[0]["name"] == "no_mask"
    assert conditions[-1]["name"] == "no_mask_repeat"
    targets = [
        condition
        for condition in conditions
        if condition["control_type"] == "target_roi"
    ]
    assert {condition["masked_roi"] for condition in targets} == set(ROI_GROUPS)
    assert all(len(condition["masked_token_indices"]) == 4 for condition in targets)
    assert_controls_are_pure_and_matched(audit, "roi_designs")


def test_e3b_controls_match_each_joint_condition_count():
    inventory = read_csv(INVENTORY)
    conditions, audit = build_joint_category(
        inventory,
        "Face",
        k=4,
        replicates=5,
        seed=20260726,
        purity_threshold=0.10,
    )
    assert len(conditions) == 32
    expected = {"Face": 4, "Face+Body": 8, "Face+Scene": 8, "Body+Scene": 8, "Face+Body+Scene": 12}
    assert {
        label: design["target_count"]
        for label, design in audit["joint_designs"].items()
    } == expected
    assert_controls_are_pure_and_matched(audit, "joint_designs")


def test_local_metric_directions_are_explicit():
    assert causal_loss("face_dino", 0.8, 0.6) == 0.2
    assert causal_loss("background_clip", 0.8, 0.6) == 0.2
    assert causal_loss("face_lpips", 0.2, 0.4) == 0.2
