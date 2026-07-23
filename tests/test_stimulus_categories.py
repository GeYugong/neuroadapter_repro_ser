from neuro_roi_causal.stimulus_categories import assign_category


THRESHOLDS = {
    "face_area": 0.01,
    "face_score": 0.3,
    "person_area": 0.08,
    "body_max_face_area": 0.04,
    "body_score": 0.25,
    "scene_score": 0.35,
    "scene_max_person_area": 0.2,
    "scene_max_face_area": 0.03,
    "word_score": 0.45,
}


def test_face_assignment():
    result = assign_category(
        face_confidence=0.8, face_area_ratio=0.1, person_count=1, person_area_ratio=0.2,
        clip_face=0.6, clip_body=0.1, clip_scene=0.1, clip_word=0.1,
        thresholds=THRESHOLDS,
    )
    assert result["confirmatory_category"] == "Face"


def test_ambiguous_assignment_is_excluded():
    result = assign_category(
        face_confidence=0.8, face_area_ratio=0.02, person_count=1, person_area_ratio=0.1,
        clip_face=0.6, clip_body=0.7, clip_scene=0.1, clip_word=0.1,
        thresholds=THRESHOLDS,
    )
    assert result["confirmatory_category"] == ""
    assert result["exclusion_reason"] == "ambiguous_confirmatory_categories"


def test_face_detector_false_positive_without_coco_person_is_rejected():
    result = assign_category(
        face_confidence=0.99, face_area_ratio=0.15, person_count=0, person_area_ratio=0.0,
        clip_face=0.1, clip_body=0.1, clip_scene=0.3, clip_word=0.4,
        thresholds=THRESHOLDS,
    )
    assert result["category_flags"]["Face"] is False
