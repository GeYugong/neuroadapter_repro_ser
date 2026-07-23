# Current State

Last updated: 2026-07-24

## Scope

Stages A and B are complete. No diffusion pilot, full E2 decoding, or model
training was launched. The Appendix P reproduction attempt and existing
50-sample zero-mask outputs remain under `experiments/roi_ablation/` as
legacy/exploratory work.

## Stage A: Implementation

- The research plan, protocol, decision log, result template, package layout,
  path example, and E2 pilot design are present.
- ROI intervention now occurs on ParcelMapper output before an optional
  TokenMapper.
- `none`, `zero`, and training-mean replacement enforce parcel dimensions and
  indices, preserve non-target parcels bitwise, and record token-norm audits.
- Mean replacement uses training-set ParcelMapper outputs, not decoder queries.
- The upstream NeuroAdapter checkout was not modified.
- Server verification: `15 passed`.

## Stage B: E0 Mapping

The primary mapping is the public Algonauts Project 2023 Subject 1 fsaverage
mapping with a strict overlap rule greater than 0.5.

| ROI | All 1000 parcels | Top-SNR-200 | Retention |
| --- | ---: | ---: | ---: |
| V1 | 10 | 10 | 100% |
| V2 | 9 | 9 | 100% |
| V3 | 6 | 6 | 100% |
| V4 | 5 | 5 | 100% |
| Face | 4 | 4 | 100% |
| Body | 24 | 24 | 100% |
| Scene | 31 | 31 | 100% |
| Word | 8 | 8 | 100% |
| Unlabeled | 903 | 103 | 11.4% |

All 97 parcels assigned a public functional ROI label are already included in
top-SNR-200. Under this mapping, top-SNR selection does **not** reduce Face,
Word, or V4 coverage. It instead strongly enriches labeled functional parcels
relative to the overall 20% parcel-selection rate. The original coverage-based
premise for an ROI-balanced-200 model is therefore unsupported.

Generated artifacts:

```text
experiments/E0_mapping/
  algonauts_full_parcel_inventory_subj01.csv
  algonauts_top200_mapping_subj01.csv
  roi_coverage_all_vs_top200.csv
  mapping_metadata.json
  file_hashes.json
  figures/
```

## Stage B: E1 Stimuli

Selection uses only ground-truth NSD stimuli. CLIP RN50 supplies semantic
scores, OpenCV Haar supplies face geometry, and official COCO 2017 instance
annotations supply person segmentation area. The manifest records dependency
sources, hashes, software versions, thresholds, and inference parameters.

| Category | Candidates | Selected | Role |
| --- | ---: | ---: | --- |
| Face | 63 | 37 | Confirmatory, below preferred minimum of 40 |
| Body | 136 | 50 | Confirmatory |
| Scene | 387 | 50 | Confirmatory |
| Word | 24 | 21 | Exploratory only; no OCR evidence |

The audit grid samples evenly spaced score ranks rather than only the highest
scores. Visual inspection confirmed the Face, Body, and Scene audit samples.
Word contains CLIP false positives and is not eligible for confirmatory tests.
Its 21 selected samples exclude every confirmatory dataset index. No samples
were duplicated to reach a target count.

Generated artifacts:

```text
experiments/E1_stimulus_manifest/
  all_test_images.csv
  face_candidates.csv
  body_candidates.csv
  scene_candidates.csv
  word_candidates.csv
  confirmatory_manifest.csv
  exploratory_manifest.csv
  figures/category_audit_grid.png
  manifest_metadata.json
```

## Open Issues Before E2

1. The legacy batch decoder accepts a contiguous `start_idx` range and cannot
   yet consume the non-contiguous category-specific dataset indices.
2. The legacy random-control generator expects old mapping column names and
   emits only zero-mask controls. It must be adapted to the E0 schema, emit
   zero/mean conditions, and record matching distances.
3. E2 still requires a deterministic dry-run proving that every condition for
   one image shares latent/noise across condition batches.
4. Face has 37 audited samples, so final statistical power must use the true
   sample count rather than assuming 50.

Do not run the current legacy decoder as the E2 pilot until items 1-3 are
resolved.

## Exact Next Command

The next safe commands re-verify the completed A/B gate:

```bash
cd "$NEUROADAPTER_PROJECT_ROOT/repro"
PYTHONPATH="$NEUROADAPTER_PROJECT_ROOT/tools/test-deps:src" \
  conda run -n neuroadapter python -m pytest -q
python scripts/validate_stage_ab_artifacts.py
```

The next development task is to make `configs/experiments/E2_pilot.yaml`
executable through a manifest-aware pilot runner. Full GPU decoding remains
outside the current authorization.
