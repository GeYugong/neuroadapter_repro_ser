import numpy as np

from neuro_roi_causal.mapping import dominant_assignment, parcel_overlaps


def test_strict_overlap_and_dominant_assignment():
    vertices = np.array([0, 1, 2, 3])
    roi_arrays = {
        "prf-visualrois": np.array([1, 1, 0, 0, 0]),
        "floc-faces": np.array([1, 1, 1, 0, 0]),
    }
    names = {
        "prf-visualrois": {1: "V1v"},
        "floc-faces": {1: "OFA"},
    }
    overlaps = parcel_overlaps(vertices, roi_arrays, names)
    group, level, overlap = dominant_assignment(overlaps, threshold=0.5)
    assert (group, level, overlap) == ("Face", "high_level", 0.75)
    assert dominant_assignment(overlaps, threshold=0.75)[0] == "Unlabeled"

