# Experiment Protocol

## Pairing

For a given subject, stimulus, checkpoint, and seed, every condition must use
the same fMRI sample, initial VAE latent, diffusion noise, denoising schedule,
guidance scale, and model weights. Only the parcel intervention may differ.

## Intervention Point

```text
fMRI beta
  -> ParcelMapper
  -> parcel tokens [B, P, D]
  -> zero / training-mean / no intervention
  -> optional TokenMapper
  -> diffusion condition tokens
```

Parcel indices must never be applied to decoder-query tokens. Training-mean
replacement is the mean of ParcelMapper outputs for the corresponding parcel.

## Comparisons

Each confirmatory ROI analysis includes:

- no-mask baseline;
- full-group zero and mean replacement;
- equal-k zero and mean replacement;
- hemisphere/SNR/parcel-size matched random controls;
- at least one unrelated functional ROI control.

Face, Body, and Scene are confirmatory. Word is exploratory unless at least 20
audited stimuli are available.

## Provenance

Each run records the repository commit, checkpoint and hash, mapping and hash,
selected parcel indices, subject, dataset indices, seed, diffusion parameters,
intervention stage and mode, masked indices, and software environment.

Large images, checkpoints, datasets, caches, and complete logs remain outside
Git. Small CSV/JSON summaries and audit figures may be committed.

