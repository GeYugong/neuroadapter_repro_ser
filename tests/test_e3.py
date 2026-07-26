from pathlib import Path

import pytest

from neuro_roi_causal.e2 import read_csv
from neuro_roi_causal.e3 import (
    cell_result_rows,
    descriptive_distribution_rows,
    ROI_GROUPS,
    build_interaction_category,
    build_joint_category,
    interaction_rows,
    joint_result_rows,
    max_overlap,
    monotonicity_rows,
)
from neuro_roi_causal.stats import benjamini_hochberg
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
    assert causal_loss("face_dino", 0.8, 0.6) == pytest.approx(0.2)
    assert causal_loss("background_clip", 0.8, 0.6) == pytest.approx(0.2)
    assert causal_loss("face_lpips", 0.2, 0.4) == pytest.approx(0.2)


def synthetic_effects():
    rows = []
    for category_index, category in enumerate(ROI_GROUPS):
        for roi_index, roi in enumerate(ROI_GROUPS):
            for dataset_idx in range(4):
                rows.append(
                    {
                        "image_category": category,
                        "masked_roi": roi,
                        "metric": "dino",
                        "dataset_idx": dataset_idx,
                        "excess_causal_loss": (
                            0.2
                            if category == roi
                            else -0.01 * (category_index + roi_index + 1)
                        ),
                    }
                )
    return rows


def test_e3a_formal_statistics_apply_one_bh_family():
    rows = interaction_rows(
        synthetic_effects(),
        metrics=["dino"],
        draws=200,
        formal=True,
        analysis_status="formal",
    )
    assert len(rows) == 3
    assert all(row["num_matched_images"] == 4 for row in rows)
    assert all(row["num_nonmatched_images"] == 8 for row in rows)
    assert all(row["permutation_p"] is not None for row in rows)
    assert all(row["bh_q_e3a"] is not None for row in rows)
    assert all(row["nonmatched_category_weighting"] == "equal_category_mean" for row in rows)


def test_e3a_nonmatched_categories_are_weighted_equally():
    rows = []
    for dataset_idx in range(3):
        rows.append(
            {
                "image_category": "Face",
                "masked_roi": "Face",
                "metric": "dino",
                "dataset_idx": dataset_idx,
                "excess_causal_loss": 1.0,
            }
        )
    rows.append(
        {
            "image_category": "Body",
            "masked_roi": "Face",
            "metric": "dino",
            "dataset_idx": 0,
            "excess_causal_loss": 0.0,
        }
    )
    for dataset_idx in range(9):
        rows.append(
            {
                "image_category": "Scene",
                "masked_roi": "Face",
                "metric": "dino",
                "dataset_idx": dataset_idx,
                "excess_causal_loss": 0.8,
            }
        )
    result = interaction_rows(
        rows,
        metrics=["dino"],
        draws=20,
        formal=False,
        analysis_status="test",
    )[0]
    assert result["matched_minus_nonmatched"] == pytest.approx(0.6)


def test_e3b_formal_statistics_use_separate_bh_family():
    rows = joint_result_rows(
        synthetic_effects(),
        draws=200,
        formal=True,
        analysis_status="formal",
    )
    assert len(rows) == 9
    assert all(row["sign_flip_p"] is not None for row in rows)
    assert all(row["bh_q_e3b"] is not None for row in rows)


def test_e3a_local_statistics_form_one_separate_family():
    metrics = {category: [f"{category.lower()}_local"] for category in ROI_GROUPS}
    effects = []
    for category in ROI_GROUPS:
        for roi in ROI_GROUPS:
            for dataset_idx in range(4):
                effects.append(
                    {
                        "image_category": category,
                        "masked_roi": roi,
                        "metric": metrics[category][0],
                        "dataset_idx": dataset_idx,
                        "excess_causal_loss": 0.1,
                    }
                )
    rows = cell_result_rows(
        effects,
        metrics_by_category=metrics,
        draws=200,
        formal=True,
        analysis_status="formal",
        q_column="bh_q_e3a_local",
    )
    assert len(rows) == 9
    assert all(row["metric_family"] == "local" for row in rows)
    assert all(row["bh_q_e3a_local"] is not None for row in rows)


def test_e3b_bh_is_applied_separately_to_global_and_local_metrics():
    effects = []
    for metric in ("dino", "face_local"):
        for dataset_idx, value in enumerate((0.3, 0.2, 0.1, -0.05)):
            effects.append(
                {
                    "image_category": "Face",
                    "masked_roi": "Face",
                    "metric": metric,
                    "dataset_idx": dataset_idx,
                    "excess_causal_loss": value,
                }
            )
    rows = joint_result_rows(
        effects,
        draws=200,
        formal=True,
        analysis_status="formal",
    )
    for family in ("global", "local"):
        family_rows = [row for row in rows if row["metric_family"] == family]
        expected = benjamini_hochberg(
            [row["sign_flip_p"] for row in family_rows]
        )
        assert [row["bh_q_e3b"] for row in family_rows] == expected


def test_e3b_monotonicity_uses_preregistered_4_8_12_levels():
    local_metrics = {category: [f"{category.lower()}_local"] for category in ROI_GROUPS}
    effects = []
    doubles = ("Face+Body", "Face+Scene", "Body+Scene")
    for category in ROI_GROUPS:
        for metric in ("dino", local_metrics[category][0]):
            for dataset_idx in range(4):
                values = {category: 0.1, "Face+Body+Scene": 0.3}
                values.update(
                    {
                        label: (0.2 if category in label.split("+") else -0.1)
                        for label in doubles
                    }
                )
                for masked_roi, value in values.items():
                    effects.append(
                        {
                            "image_category": category,
                            "masked_roi": masked_roi,
                            "metric": metric,
                            "dataset_idx": dataset_idx,
                            "excess_causal_loss": value,
                        }
                    )
    rows = monotonicity_rows(
        effects,
        metrics_by_category=local_metrics,
        draws=200,
        formal=True,
        analysis_status="formal",
    )
    assert len(rows) == 6
    assert all(row["mean_excess_k4"] == pytest.approx(0.1) for row in rows)
    assert all(row["mean_excess_k8"] == pytest.approx(0.2) for row in rows)
    assert all(row["mean_excess_k12"] == pytest.approx(0.3) for row in rows)
    assert all(row["mean_slope_per_level"] == pytest.approx(0.1) for row in rows)
    assert all(row["monotonic_non_decreasing_fraction"] == 1.0 for row in rows)
    assert all(row["bh_q_e3b_monotonicity"] is not None for row in rows)


def test_descriptive_distribution_rows_have_no_inference_fields():
    rows = descriptive_distribution_rows(synthetic_effects())
    assert len(rows) == 9
    assert all(row["num_images"] == 4 for row in rows)
    assert all(row["analysis_status"] == "descriptive_only" for row in rows)
    assert all(0.0 <= row["positive_fraction"] <= 1.0 for row in rows)
    assert all("sign_flip_p" not in row for row in rows)
