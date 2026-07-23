# Functional ROI Causal Study

## Objective

This project studies whether functional fMRI regions make category-specific
causal contributions to image reconstruction and whether top-SNR parcel
selection biases those conclusions.

Strict reproduction of Appendix P's 50/53/103 parcel counts is no longer the
primary objective. The main protocol uses the public Algonauts Project 2023
Subject 1 fsaverage ROI masks.

## Research Questions

1. Does masking a category-matched ROI degrade matched stimuli more than an
   unrelated ROI or matched random parcels?
2. Do zero replacement and training-mean replacement agree in direction?
3. Does top-SNR-200 underrepresent functional ROIs such as Face, Word, or V4?
4. After the core study, does attention magnitude predict masking effects?

## Stages

- E0: build a complete public ROI inventory and quantify top-SNR selection bias.
- E1: construct stimulus manifests for Face, Body, Scene, and exploratory Word.
- E2: run paired causal masking on the existing top-SNR-200 model.
- E3: compare newly trained top-SNR-200 and ROI-balanced-200 models.
- E4: optional attention-to-causal-effect analysis.

## Current Scope

The current implementation covers stages A and B: documentation, parcel-level
interventions, tests, E0 mapping, E1 manifest generation, and an E2 pilot
configuration. It does not authorize full GPU decoding or model training.

