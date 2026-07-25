"""Content-addressed deterministic metrics for E2 image pairs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity


def image_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rgb(path: str | Path, size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(
        (size, size), Image.Resampling.BILINEAR
    )
    return np.asarray(image, dtype=np.float32) / 255.0


def pixel_corr(gt: np.ndarray, pred: np.ndarray) -> float:
    x = gt.reshape(-1).astype(np.float64)
    y = pred.reshape(-1).astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return 0.0 if denominator == 0 else float(np.dot(x, y) / denominator)


def ssim_gray(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(
        structural_similarity(
            rgb2gray(gt),
            rgb2gray(pred),
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=1.0,
        )
    )


class ContentAddressedCache:
    """Cache metric inputs by image bytes, not path or batch position."""

    def __init__(self) -> None:
        self._path_hashes: dict[Path, str] = {}
        self.embeddings: dict[tuple[str, str], torch.Tensor] = {}
        self.pairs: dict[tuple[str, str, str], float] = {}
        self.embedding_computations: dict[str, int] = {}
        self.pair_computations: dict[str, int] = {}

    def sha(self, path: str | Path) -> str:
        resolved = Path(path).resolve()
        if resolved not in self._path_hashes:
            self._path_hashes[resolved] = image_sha256(resolved)
        return self._path_hashes[resolved]

    def embedding(
        self, kind: str, path: str | Path, compute: Callable[[Path], torch.Tensor]
    ) -> torch.Tensor:
        key = (kind, self.sha(path))
        if key not in self.embeddings:
            self.embeddings[key] = compute(Path(path)).detach().cpu()
            self.embedding_computations[kind] = self.embedding_computations.get(kind, 0) + 1
        return self.embeddings[key]

    def pair_value(
        self,
        kind: str,
        gt: str | Path,
        pred: str | Path,
        compute: Callable[[Path, Path], float],
    ) -> float:
        key = (kind, self.sha(gt), self.sha(pred))
        if key not in self.pairs:
            self.pairs[key] = float(compute(Path(gt), Path(pred)))
            self.pair_computations[kind] = self.pair_computations.get(kind, 0) + 1
        return self.pairs[key]


def representation_metrics(
    pairs: list[tuple[Path, Path]],
    models: dict,
    batch_size: int,
    cache: ContentAddressedCache,
) -> list[dict[str, float]]:
    device = models["device"]
    unique: dict[str, Path] = {}
    for gt, pred in pairs:
        unique.setdefault(cache.sha(gt), gt)
        unique.setdefault(cache.sha(pred), pred)
    with torch.inference_mode():
        for kind, model, preprocess in (
            ("clip", models["clip"], models["clip_preprocess"]),
            ("dino", models["dino"], models["dino_preprocess"]),
        ):
            missing = [
                (sha, path)
                for sha, path in unique.items()
                if (kind, sha) not in cache.embeddings
            ]
            for start in range(0, len(missing), batch_size):
                batch = missing[start : start + batch_size]
                tensors = torch.stack(
                    [preprocess(Image.open(path).convert("RGB")) for _, path in batch]
                ).to(device)
                features = (
                    model.encode_image(tensors) if kind == "clip" else model(tensors)
                ).float().cpu()
                for (sha, _), feature in zip(batch, features):
                    cache.embeddings[(kind, sha)] = feature
                    cache.embedding_computations[kind] = (
                        cache.embedding_computations.get(kind, 0) + 1
                    )

        unique_pairs = {}
        for gt, pred in pairs:
            key = ("lpips", cache.sha(gt), cache.sha(pred))
            if key not in cache.pairs:
                unique_pairs.setdefault(key, (gt, pred))
        items = list(unique_pairs.items())
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            gt_tensors = torch.stack(
                [
                    models["lpips_preprocess"](Image.open(gt).convert("RGB"))
                    for _, (gt, _) in batch
                ]
            ).to(device)
            pred_tensors = torch.stack(
                [
                    models["lpips_preprocess"](Image.open(pred).convert("RGB"))
                    for _, (_, pred) in batch
                ]
            ).to(device)
            values = models["lpips"](gt_tensors, pred_tensors).flatten().float().cpu()
            for (key, _), value in zip(batch, values):
                cache.pairs[key] = float(value)
                cache.pair_computations["lpips"] = (
                    cache.pair_computations.get("lpips", 0) + 1
                )

    output = []
    for gt, pred in pairs:
        gt_sha, pred_sha = cache.sha(gt), cache.sha(pred)
        output.append(
            {
                "lpips": cache.pairs[("lpips", gt_sha, pred_sha)],
                "clip": float(
                    torch.nn.functional.cosine_similarity(
                        cache.embeddings[("clip", gt_sha)].unsqueeze(0),
                        cache.embeddings[("clip", pred_sha)].unsqueeze(0),
                    ).item()
                ),
                "dino": float(
                    torch.nn.functional.cosine_similarity(
                        cache.embeddings[("dino", gt_sha)].unsqueeze(0),
                        cache.embeddings[("dino", pred_sha)].unsqueeze(0),
                    ).item()
                ),
            }
        )
    return output
