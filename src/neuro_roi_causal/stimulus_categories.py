"""Pure category-decision logic for stimulus manifest generation."""

from __future__ import annotations


CONFIRMATORY_CATEGORIES = ("Face", "Body", "Scene")


def assign_category(
    *,
    face_confidence: float,
    face_area_ratio: float,
    person_count: int,
    person_area_ratio: float,
    clip_face: float,
    clip_body: float,
    clip_scene: float,
    clip_word: float,
    thresholds: dict[str, float],
) -> dict:
    face = (
        person_count > 0
        and face_area_ratio >= thresholds["face_area"]
        and face_confidence >= thresholds["face_score"]
    )
    body = (
        person_area_ratio >= thresholds["person_area"]
        and face_area_ratio <= thresholds["body_max_face_area"]
        and clip_body >= thresholds["body_score"]
    )
    scene = (
        clip_scene >= thresholds["scene_score"]
        and person_area_ratio <= thresholds["scene_max_person_area"]
        and face_area_ratio <= thresholds["scene_max_face_area"]
    )
    word = clip_word >= thresholds["word_score"]
    flags = {"Face": face, "Body": body, "Scene": scene, "Word": word}
    confirmatory = [name for name in CONFIRMATORY_CATEGORIES if flags[name]]
    if len(confirmatory) == 1:
        category = confirmatory[0]
        exclusion = ""
    elif not confirmatory:
        category = ""
        exclusion = "no_confirmatory_category"
    else:
        category = ""
        exclusion = "ambiguous_confirmatory_categories"
    scores = {
        "Face": max(face_confidence, clip_face) * max(face_area_ratio, 1e-6),
        "Body": clip_body * max(person_area_ratio, 1e-6),
        "Scene": clip_scene * max(1.0 - person_area_ratio, 0.0),
        "Word": clip_word,
    }
    return {
        "category_flags": flags,
        "confirmatory_category": category,
        "exclusion_reason": exclusion,
        "category_scores": scores,
    }
