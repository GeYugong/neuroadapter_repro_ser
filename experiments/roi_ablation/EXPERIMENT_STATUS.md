# ROI Ablation Status

## Mapping gate: not passed

The current Subject 1 mapping was created from mirrored fsaverage ROI arrays,
not from a separately verified official NSD release. With the stated `>50%`
parcel-overlap rule and the checkpoint's exact selected 200 tokens, it yields:

| Group | Tokens |
|---|---:|
| Low-level (V1-V4) | 30 |
| High-level (Face, Body, Scene, Word) | 67 |
| Labeled total | 97 |
| Unlabeled | 103 |

The paper target is approximately 50 low-level, 53 high-level, and 103 labeled
tokens. This discrepancy was not resolved from the server's available files.

## Consequence

The zero-mask 50-sample pilot and its official metrics were completed before
this gate was enforced. They are retained as exploratory artifacts only and
must not be interpreted as a paper-faithful functional-ROI result. The pending
mean-mask and random-matched-control runs were stopped before producing outputs.

## Required next action

Obtain or independently verify the official NSD Subject 1 surface ROI labels
and their space/vertex convention, then rerun `prepare_roi_mapping.py`. Resume
ROI ablations only after the mapping discrepancy is explained or corrected.
