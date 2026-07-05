#!/usr/bin/env bash
set -euo pipefail

ROOT=/public/home/mty/GeYugong
DATA="$ROOT/data/nsd"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
mkdir -p "$DATA/experiments/nsd"
mkdir -p "$DATA/stimuli"
mkdir -p "$DATA/betas/subj01/fsaverage/betas_fithrf_GLMdenoise_RR"

aws_s3() {
  PYTHONNOUSERSITE=1 conda run -n neuroadapter env \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
    aws s3 "$@"
}

copy_if_missing() {
  local src="$1"
  local dst="$2"
  if [ -s "$dst" ]; then
    local size
    size=$(du -h "$dst" | cut -f1)
    echo "[skip] $dst exists ($size)"
  else
    mkdir -p "$(dirname "$dst")"
    echo "[copy] $src -> $dst"
    aws_s3 cp --no-sign-request "$src" "$dst"
  fi
}

sync_prefix() {
  local src="$1"
  local dst="$2"
  mkdir -p "$dst"
  echo "[sync] $src -> $dst"
  aws_s3 sync --no-sign-request "$src" "$dst"
}

START_TS=$(date '+%F %T')
echo "NSD subject 1 download started at $START_TS"
echo "Target: $DATA"

copy_if_missing \
  s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_expdesign.mat \
  "$DATA/experiments/nsd/nsd_expdesign.mat"

copy_if_missing \
  s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.csv \
  "$DATA/experiments/nsd/nsd_stim_info_merged.csv"

copy_if_missing \
  s3://natural-scenes-dataset/nsddata/experiments/nsd/nsd_stim_info_merged.pkl \
  "$DATA/experiments/nsd/nsd_stim_info_merged.pkl"

copy_if_missing \
  s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 \
  "$DATA/stimuli/nsd_stimuli.hdf5"

sync_prefix \
  s3://natural-scenes-dataset/nsddata_betas/ppdata/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/ \
  "$DATA/betas/subj01/fsaverage/betas_fithrf_GLMdenoise_RR/"

END_TS=$(date '+%F %T')
echo "NSD subject 1 download finished at $END_TS"
echo "Downloaded size:"
du -sh "$DATA"
