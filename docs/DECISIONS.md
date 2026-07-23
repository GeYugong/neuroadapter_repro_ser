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

## D004: Stimulus categorization

Stage B uses only locally available, explicitly recorded tools. The manifest
builder refuses to download CLIP weights. Face and person geometry use OpenCV
built-in detectors; semantic category confidence uses a caller-supplied local
CLIP checkpoint. Word remains exploratory when OCR evidence is unavailable.

