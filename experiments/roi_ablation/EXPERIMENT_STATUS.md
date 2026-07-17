# ROI Ablation Status

## Mapping gate: not passed

The original Subject 1 mapping used mirrored fsaverage ROI arrays. Directly
resampling the official NSD Subject 1 native-surface `.mgz` labels through the
official `sphere.reg` registration changes fLoc labels substantially, proving
that the mirror is not an exact official equivalent. With the official-resampled
labels, the stated `>50%` parcel-overlap rule and this checkpoint's exact 200
selected tokens yield:

| Group | Tokens |
|---|---:|
| Low-level (V1-V4) | 30 |
| High-level (Face, Body, Scene, Word) | 77 |
| Labeled total | 107 |
| Unlabeled | 93 |

The paper reports 50 low-level, 53 high-level, and 103 labeled tokens. The
remaining discrepancy is therefore attributable to the checkpoint's selected
parcels and/or its reproduction data pipeline, rather than the earlier mirror
label source. This checkpoint cannot be described as the paper's ROI-mask setup.

## Consequence

The zero-mask 50-sample pilot and its official metrics were completed before
this gate was enforced. They are retained as exploratory artifacts only and
must not be interpreted as a paper-faithful functional-ROI result. The pending
mean-mask and random-matched-control runs were stopped before producing outputs.

## Required next action

Identify the exact NSD preprocessing and SNR parcel-selection artifacts used by
the authors (or obtain their trained Subject 1 checkpoint and parcel indices).
Resume the paper-faithful ablation only after the selected-token discrepancy is
explained or corrected.
