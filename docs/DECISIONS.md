# Decisions

## D001: Public ROI protocol

The primary ROI mapping is the Algonauts Project 2023 Subject 1 fsaverage
release with a strict overlap threshold greater than 0.5. The resulting
top-SNR-200 mapping is treated as a public research protocol rather than a
strict reproduction of Appendix P.

## D002: Legacy experiments

Existing ROI reproduction artifacts and the 50-sample zero-mask pilot are
retained, but their conclusions are labeled legacy/exploratory.

## D003: Intervention stage

ROI interventions operate on ParcelMapper outputs before TokenMapper. This is
required because a transformer decoder maps 200 parcel tokens to 50 decoder
queries, destroying index equivalence.

The actual step-100000 checkpoint uses `linear_projection`, not
`transformer_decoder`; its 200 ParcelMapper outputs are used directly as 200
condition tokens. The pre-TokenMapper implementation remains necessary for
compatibility with Transformer checkpoints, and that branch is unit-tested.

## D004: Stimulus categorization

The manifest builder refuses to download weights. Semantic confidence uses a
caller-supplied OpenAI CLIP RN50 checkpoint. Face geometry uses an explicitly
provided OpenCV 4.12 Haar cascade and is gated by COCO person presence. Body
and Scene use official COCO 2017 person segmentation area rather than OpenCV
HOG, which produced visible false positives. Word remains exploratory because
OCR evidence is unavailable.

## D005: Top-SNR coverage hypothesis

The full E0 inventory shows that all 97 publicly labeled functional parcels are
already in top-SNR-200. Face, Word, and V4 each have 100% retention. The
coverage-underrepresentation version of H3 is rejected for the primary public
mapping. ROI-balanced model training is not justified by coverage alone and is
paused unless another pre-registered selection question is established.

## D006: Honest stimulus counts

The final confirmatory manifest contains 37 Face, 50 Body, and 50 Scene images.
The conservative Face rules intentionally accept a small shortfall from the
preferred minimum of 40 rather than retain audited false positives or duplicate
images. Of 24 CLIP-only Word candidates, 21 do not overlap the confirmatory
manifest and are retained as exploratory samples. They remain non-confirmatory
because the required OCR evidence is absent.

## D007: Stage C gate

The legacy decoder cannot consume non-contiguous manifest indices, and the
legacy random-control generator does not consume the E0 schema. Neither is an
acceptable E2 runner as-is. Stage C must first add manifest-aware dataset
selection and validated zero/mean matched controls.
