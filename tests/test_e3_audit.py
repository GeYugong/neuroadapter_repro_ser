from neuro_roi_causal.e3_audit import (
    DDPM_STRATEGY,
    SHARED_POLICY,
    audit_shared_diffusion_state,
)


def summary():
    names = ["no_mask", "mask_Face_mean", "no_mask_repeat"]
    return {
        "seed": 12345,
        "condition_batch_size": 2,
        "condition_batches_padded_to_fixed_size": True,
        "shared_latent_noise_policy": SHARED_POLICY,
        "shared_diffusion_state": [
            {
                "dataset_idx": 10,
                "sample_seed": 12355,
                "ddpm_denoising_seed": 1012355,
                "initial_latent_sha256": "a" * 64,
                "diffusion_noise_sha256": "b" * 64,
                "shape": [1, 4, 64, 64],
                "dtype": "torch.float16",
                "ddpm_noise_strategy": DDPM_STRATEGY,
                "reused_condition_count": 3,
                "reused_condition_names": names,
                "condition_batch_count": 2,
            }
        ],
    }, names


def test_shared_state_audit_proves_every_condition_reuses_one_pair():
    payload, names = summary()
    audit = audit_shared_diffusion_state(
        payload,
        expected_indices=[10],
        expected_condition_names=names,
    )
    assert audit["passed"]
    assert audit["records"][0]["checks"]["reused_for_every_condition"]


def test_shared_state_audit_rejects_missing_condition_reuse():
    payload, names = summary()
    payload["shared_diffusion_state"][0]["reused_condition_count"] = 2
    audit = audit_shared_diffusion_state(
        payload,
        expected_indices=[10],
        expected_condition_names=names,
    )
    assert not audit["passed"]
