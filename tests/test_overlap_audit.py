import json

import pytest

from neuro_roi_causal.overlap_audit import parse_all_roi_overlaps, summarize_parcel


def test_overlap_json_preserves_dominant_and_non_dominant_entries():
    value = json.dumps(
        [
            {"collection": "faces", "roi": "FFA", "group": "Face", "overlap": 0.6},
            {"collection": "bodies", "roi": "EBA", "group": "Body", "overlap": 0.3},
        ]
    )
    parsed = parse_all_roi_overlaps(value)
    assert [row["roi"] for row in parsed] == ["FFA", "EBA"]
    row = {
        "all_roi_overlaps": value,
        "top200_token_index": "1",
        "hemisphere": "lh",
        "schaefer_parcel_index": "10",
        "dominant_roi": "Face",
        "dominant_overlap": "0.6",
        "mean_ncsnr": "0.5",
        "num_vertices": "100",
    }
    summary = summarize_parcel(row, "Face")
    assert summary["target_group_max_overlap"] == pytest.approx(0.6)
    assert summary["max_off_target_overlap"] == pytest.approx(0.3)
    assert summary["roi_overlap__faces__FFA"] == pytest.approx(0.6)
    assert summary["roi_overlap__bodies__EBA"] == pytest.approx(0.3)


def test_overlap_group_values_are_maxima_not_sums():
    value = json.dumps(
        [
            {"collection": "prf", "roi": "V1d", "group": "V1", "overlap": 0.4},
            {"collection": "prf", "roi": "V1v", "group": "V1", "overlap": 0.35},
        ]
    )
    row = {
        "all_roi_overlaps": value,
        "top200_token_index": "1",
        "hemisphere": "lh",
        "schaefer_parcel_index": "10",
        "dominant_roi": "V1",
        "dominant_overlap": "0.4",
        "mean_ncsnr": "0.5",
        "num_vertices": "100",
    }
    assert summarize_parcel(row, "V1")["target_group_max_overlap"] == pytest.approx(0.4)
