from pathlib import Path

from neuro_roi_causal.e2_audit import no_mask_record_failures


def record(index: int, path: Path) -> dict:
    return {"dataset_idx": index, "pred": str(path)}


def test_zero_mean_no_mask_accepts_identical_image_bytes(tmp_path):
    zero, mean = tmp_path / "zero.png", tmp_path / "mean.png"
    zero.write_bytes(b"same image")
    mean.write_bytes(b"same image")
    assert not no_mask_record_failures(
        [record(1, mean)], [record(0, zero), record(1, zero)]
    )


def test_zero_mean_no_mask_rejects_different_bytes_or_indices(tmp_path):
    zero, mean = tmp_path / "zero.png", tmp_path / "mean.png"
    zero.write_bytes(b"zero")
    mean.write_bytes(b"mean")
    assert no_mask_record_failures([record(1, mean)], [record(1, zero)])
    assert no_mask_record_failures([record(2, zero)], [record(1, zero)])
