from neuro_roi_causal.mean_plan import convert_zero_plan, equivalence_audit


def zero_plan():
    return {
        "name": "E2_top_snr_causal_full",
        "created_at": "old",
        "categories": {
            "Face": {
                "dataset_indices": [1, 5],
                "conditions": [
                    {"name": "no_mask", "mask_mode": "zero", "masked_token_indices": []},
                    {
                        "name": "Face_full_zero",
                        "mask_mode": "zero",
                        "masked_token_indices": [2, 8],
                    },
                    {
                        "name": "no_mask_repeat",
                        "mask_mode": "zero",
                        "masked_token_indices": [],
                    },
                ],
                "matching_audit": {"indices": [2, 8]},
            }
        },
    }


def convert(plan):
    return convert_zero_plan(
        plan,
        source_zero_plan="/zero.json",
        source_zero_plan_sha256="zero",
        mean_token_cache="/mean.pt",
        mean_token_cache_sha256="mean",
        repository_commit="commit",
        created_at="new",
    )


def test_zero_to_mean_only_changes_allowed_fields():
    zero = zero_plan()
    mean = convert(zero)
    assert equivalence_audit(zero, mean)["passed"]
    assert mean["categories"]["Face"]["dataset_indices"] == [1, 5]
    assert mean["categories"]["Face"]["conditions"][1]["masked_token_indices"] == [2, 8]
    assert mean["categories"]["Face"]["conditions"][1]["name"] == "Face_full_mean"


def test_index_change_fails_equivalence():
    zero = zero_plan()
    mean = convert(zero)
    mean["categories"]["Face"]["conditions"][1]["masked_token_indices"] = [3, 8]
    assert not equivalence_audit(zero, mean)["passed"]
