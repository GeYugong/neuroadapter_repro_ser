"""Validated, provenance-rich parcel-mean token caches."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = 1
INTERVENTION_STAGE = "after_parcel_mapper_before_token_mapper"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_selected_parcels(value: Any) -> dict[str, list[int]]:
    if not isinstance(value, dict) or set(value) != {"lh", "rh"}:
        raise ValueError("selected_parcel_idx must contain exactly lh and rh")
    result = {hemi: [int(index) for index in value[hemi]] for hemi in ("lh", "rh")}
    if any(len(indices) == 0 for indices in result.values()):
        raise ValueError("selected_parcel_idx cannot contain an empty hemisphere")
    return result


def build_cache(
    mean_parcel_tokens: torch.Tensor,
    *,
    subject: int,
    num_train_samples: int,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_step: int,
    selected_parcel_idx: Any,
    sub_approach: str,
    git_commit: str,
    created_at: str,
) -> dict[str, Any]:
    tensor = mean_parcel_tokens.detach().float().cpu()
    selected = normalize_selected_parcels(selected_parcel_idx)
    norms = tensor.norm(dim=-1)
    return {
        "schema_version": SCHEMA_VERSION,
        "mean_parcel_tokens": tensor,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "subject": int(subject),
        "split": "train",
        "num_train_samples": int(num_train_samples),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_step": int(checkpoint_step),
        "selected_parcel_idx": selected,
        "selected_parcel_idx_sha256": canonical_sha256(selected),
        "num_parcels": int(tensor.shape[0]),
        "embedding_dim": int(tensor.shape[1]),
        "sub_approach": sub_approach,
        "intervention_stage": INTERVENTION_STAGE,
        "git_commit": git_commit,
        "created_at": created_at,
        "token_norm_mean": float(norms.mean()),
        "token_norm_std": float(norms.std(unbiased=False)),
        "token_value_mean": float(tensor.mean()),
        "token_value_std": float(tensor.std(unbiased=False)),
        "contains_nan": bool(torch.isnan(tensor).any()),
        "contains_inf": bool(torch.isinf(tensor).any()),
    }


def validate_cache(
    cache: dict[str, Any],
    *,
    checkpoint_sha256: str | None = None,
    selected_parcel_idx: Any | None = None,
    subject: int = 1,
) -> None:
    required = {
        "schema_version",
        "mean_parcel_tokens",
        "shape",
        "dtype",
        "subject",
        "split",
        "num_train_samples",
        "checkpoint_sha256",
        "checkpoint_step",
        "selected_parcel_idx",
        "selected_parcel_idx_sha256",
        "num_parcels",
        "embedding_dim",
        "sub_approach",
        "intervention_stage",
        "git_commit",
        "created_at",
    }
    missing = sorted(required.difference(cache))
    if missing:
        raise ValueError(f"Legacy or incomplete mean cache; missing: {missing}")
    tensor = cache["mean_parcel_tokens"]
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("mean_parcel_tokens must be a torch.Tensor")
    if list(tensor.shape) != [200, 768] or cache["shape"] != [200, 768]:
        raise ValueError(f"Expected mean token shape [200, 768], got {list(tensor.shape)}")
    if int(cache["num_parcels"]) != 200 or int(cache["embedding_dim"]) != 768:
        raise ValueError("Mean cache parcel or embedding dimensions are inconsistent")
    if int(cache["subject"]) != int(subject) or cache["split"] != "train":
        raise ValueError("Mean cache subject or split is inconsistent")
    if cache["intervention_stage"] != INTERVENTION_STAGE:
        raise ValueError("Mean cache intervention stage is incorrect")
    if int(cache["num_train_samples"]) <= 0:
        raise ValueError("Mean cache has no training samples")
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError("Mean cache contains NaN or Inf")
    if torch.count_nonzero(tensor) == 0:
        raise ValueError("Mean cache is all zero")
    selected = normalize_selected_parcels(cache["selected_parcel_idx"])
    if canonical_sha256(selected) != cache["selected_parcel_idx_sha256"]:
        raise ValueError("Mean cache selected parcel hash is invalid")
    if checkpoint_sha256 is not None and cache["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Mean cache checkpoint hash mismatch")
    if selected_parcel_idx is not None:
        expected = normalize_selected_parcels(selected_parcel_idx)
        if selected != expected:
            raise ValueError("Mean cache selected parcel indices mismatch")


def atomic_torch_save(cache: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(cache, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def load_validated_cache(
    path: Path,
    *,
    checkpoint_sha256: str | None = None,
    selected_parcel_idx: Any | None = None,
    subject: int = 1,
) -> dict[str, Any]:
    cache = torch.load(path, map_location="cpu")
    if not isinstance(cache, dict):
        raise TypeError("Mean cache must contain a dictionary")
    validate_cache(
        cache,
        checkpoint_sha256=checkpoint_sha256,
        selected_parcel_idx=selected_parcel_idx,
        subject=subject,
    )
    return cache
