"""Runtime audits for E3 paired-condition decoding."""

from __future__ import annotations

import re
from typing import Any

from .decoding import sample_seed


SHA256 = re.compile(r"^[0-9a-f]{64}$")
SHARED_POLICY = (
    "one initial latent and one diffusion-noise tensor per image reused "
    "across every condition"
)
DDPM_STRATEGY = "one generator per condition, identical seed and sequence"


def audit_shared_diffusion_state(
    summary: dict[str, Any],
    *,
    expected_indices: list[int],
    expected_condition_names: list[str],
) -> dict[str, Any]:
    records = summary.get("shared_diffusion_state", [])
    batch_size = int(summary["condition_batch_size"])
    expected_batches = (
        len(expected_condition_names) + batch_size - 1
    ) // batch_size
    checks = {
        "policy_declared": summary.get("shared_latent_noise_policy") == SHARED_POLICY,
        "condition_batches_fixed_shape": bool(
            summary.get("condition_batches_padded_to_fixed_size")
        ),
        "one_state_record_per_image": len(records) == len(expected_indices),
        "dataset_indices_identical": (
            [int(record.get("dataset_idx", -1)) for record in records]
            == list(map(int, expected_indices))
        ),
    }
    record_audits = []
    for record, dataset_idx in zip(records, expected_indices):
        expected_seed = sample_seed(int(summary["seed"]), int(dataset_idx))
        record_checks = {
            "sample_seed_correct": int(record.get("sample_seed", -1)) == expected_seed,
            "ddpm_seed_correct": (
                int(record.get("ddpm_denoising_seed", -1))
                == expected_seed + 1_000_000
            ),
            "latent_hash_valid": bool(
                SHA256.fullmatch(str(record.get("initial_latent_sha256", "")))
            ),
            "noise_hash_valid": bool(
                SHA256.fullmatch(str(record.get("diffusion_noise_sha256", "")))
            ),
            "latent_and_noise_distinct": (
                record.get("initial_latent_sha256")
                != record.get("diffusion_noise_sha256")
            ),
            "shape_valid": (
                isinstance(record.get("shape"), list)
                and len(record["shape"]) == 4
                and all(int(value) > 0 for value in record["shape"])
            ),
            "dtype_recorded": str(record.get("dtype", "")).startswith("torch."),
            "ddpm_strategy_declared": (
                record.get("ddpm_noise_strategy") == DDPM_STRATEGY
            ),
            "reused_for_every_condition": (
                int(record.get("reused_condition_count", -1))
                == len(expected_condition_names)
                and record.get("reused_condition_names")
                == expected_condition_names
            ),
            "condition_batch_count_correct": (
                int(record.get("condition_batch_count", -1)) == expected_batches
            ),
        }
        record_audits.append(
            {
                "dataset_idx": int(dataset_idx),
                "passed": all(record_checks.values()),
                "checks": record_checks,
                "initial_latent_sha256": record.get("initial_latent_sha256"),
                "diffusion_noise_sha256": record.get("diffusion_noise_sha256"),
            }
        )
    passed = all(checks.values()) and len(record_audits) == len(expected_indices)
    passed = passed and all(item["passed"] for item in record_audits)
    return {
        "passed": passed,
        "policy": summary.get("shared_latent_noise_policy"),
        "checks": checks,
        "records": record_audits,
    }

