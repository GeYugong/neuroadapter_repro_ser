import torch

from neuro_roi_causal.diffusion_pairing import shared_state_audit, tensor_sha256


def test_shared_state_hashes_are_stable_and_sensitive():
    latent = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    noise = latent + 1
    first = shared_state_audit(latent, noise, 12345)
    second = shared_state_audit(latent.clone(), noise.clone(), 12345)
    assert first == second
    assert first["initial_latent_sha256"] == tensor_sha256(latent)
    changed = latent.clone()
    changed[0, 0, 0, 0] += 1
    assert tensor_sha256(changed) != tensor_sha256(latent)


def test_shared_state_requires_matching_shapes():
    latent = torch.zeros(1, 2)
    noise = torch.zeros(2, 1)
    try:
        shared_state_audit(latent, noise, 1)
    except ValueError as error:
        assert "shapes must match" in str(error)
    else:
        raise AssertionError("Expected mismatched shapes to fail")
