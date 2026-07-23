# ROI Ablation Status

## Legacy / exploratory status

This directory preserves the original Appendix P reproduction attempt. The
project has moved to a public, independently reproducible Algonauts ROI
protocol under `experiments/E0_mapping/`. The mapping discrepancy documented
below remains scientifically relevant, but it no longer blocks the new study.

The existing 50-sample zero-mask outputs and metrics are exploratory only.
They must not be presented as final functional-ROI causal evidence.

## Mapping gate: not passed

The paper's Appendix P requires a strict `>50%` parcel-overlap assignment and
reports 50 low-level, 53 high-level, and 103 labeled tokens for Subject 1.
The current checkpoint's selection is now independently verified: recomputing
mean parcel ncsnr from NSD's official `lh/rh.ncsnr.mgh` reproduces all 100
selected parcel indices in each hemisphere exactly.

Official native NSD ROI labels were resampled using the official Subject 1 and
fsaverage `sphere.reg` files. The resulting mapping for those verified 200
tokens is:

| Group | Tokens |
|---|---:|
| Low-level (V1-V4) | 30 |
| High-level (Face, Body, Scene, Word) | 77 |
| Labeled total | 107 |
| Unlabeled | 93 |

This does not reproduce the paper's 50/53/103 composition. The old mirrored
fsaverage ROI arrays are also not an official equivalent: their fLoc labels
differ substantially after direct official resampling.

## Algonauts source check

The released NeuroAdapter lineage uses the Algonauts Project 2023 ROI-mask
convention. Subject 1's five public ROI classes and five code tables were
downloaded from that release and compared byte-for-byte with the original
mirror used in this reproduction; all 15 files are identical. Re-running the
same `>50%` mapping with those Algonauts masks gives 30 low-level, 67
high-level, and 97 labeled tokens. Therefore the discrepancy is not caused by
using the wrong public ROI-mask release.

## Parcel provenance finding

The training checkpoint uses
`/public/home/mty/GeYugong/data/neuroadapter/parcels/schaefer`. Its left and
right parcel files are spatially coherent standard fsaverage Schaefer labels.
The checked-in parcel files from the released `whole_brain_encoder` repository
cannot be substituted as an author-faithful reference: the released right
hemisphere partition is identical to its left hemisphere partition. The active
left partition is the same set of vertices as the released left partition but
with a different index ordering; the active right partition does not agree with
the released right file.

The reproducible evidence is in
`mapping/parcel_provenance_audit_subj01.json`,
`mapping/official_resampled_subj01_nsd_fsaverage_reg/`, and
`mapping/official_mapping_subj01_nsd_fsaverage_reg/`.

## Consequence

The 50-sample zero-mask pilot and its official metrics were completed before
this gate was enforced. They are exploratory artifacts only and must not be
reported as a paper-faithful functional-ROI result. The pending mean-mask and
random-matched-control runs were stopped before producing outputs.

## Required next action

Obtain the authors' exact `metadata_sub-01.npy` ROI-mask fields
(`lh_rois`/`rh_rois`) or an authoritative preprocessing recipe that produces
their 50/53/103 mapping. The public code confirms that these private metadata
fields were used, but does not provide them. Resume the paper-faithful ablation
only after that mapping discrepancy is resolved.
