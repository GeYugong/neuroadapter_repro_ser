"""Reusable output-audit helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def no_mask_record_failures(
    mean_records: list[dict], zero_records: list[dict]
) -> list[str]:
    failures = []
    if len(mean_records) != len(zero_records):
        return ["zero/mean record counts differ"]
    for mean_record, zero_record in zip(mean_records, zero_records):
        index = int(mean_record["dataset_idx"])
        if index != int(zero_record["dataset_idx"]):
            failures.append("zero/mean indices differ")
        elif file_sha256(mean_record["pred"]) != file_sha256(zero_record["pred"]):
            failures.append(f"dataset_idx={index}: zero/mean SHA differs")
    return failures
