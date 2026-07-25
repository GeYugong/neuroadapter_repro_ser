# NeuroAdapter Functional ROI Study

## Scope

- This repository contains the reproducible research code and small artifacts.
- The upstream NeuroAdapter checkout is a read-only dependency.
- Do not commit NSD data, model weights, checkpoints, generated image sets, or full logs.
- New code must use CLI arguments, environment variables, or YAML/JSON configuration instead of server-specific absolute paths.

## Scientific Invariants

- ROI indices always refer to parcel tokens.
- Apply interventions after `ParcelMapper` and before an optional `TokenMapper`.
- Paired conditions for one image must share fMRI, latent, diffusion noise, seed, checkpoint, and decoding parameters.
- Report full-group, equal-k, and matched-random controls separately.
- Treat the legacy 50-sample zero-mask runs as exploratory.

## Execution Gates

- Complete unit tests, dry-runs, E0 mapping, and E1 stimulus audit before GPU pilots.
- Do not start full E2 decoding or E3 training without explicit user confirmation.
- Stop when parcel dimensions, mapping indices, model stages, or required external weights cannot be verified.

## Experiment Logging

- `EXPERIMENT_LOG.md` is the single chronological source of truth for project work.
- Append every development, test, training, decoding, and evaluation event in Chinese.
- Record commands, commits, inputs, checkpoints, outputs, GPU use, runtime, metrics, figures, problems, conclusions, and the next step.
- Never overwrite or delete historical log entries. Corrections must be appended as new entries.
- `docs/CURRENT_STATE.md` is only the latest snapshot, and `docs/DECISIONS.md` contains only scientific decisions.
