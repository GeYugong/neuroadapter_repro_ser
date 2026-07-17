#!/usr/bin/env bash
set -euo pipefail

project_root=/public/home/mty/GeYugong/projects/neuroadapter-iclr2026
run_root="$project_root/outputs/roi_ablation/20260718-s01-primary-zero-roi-50x50"
code_root="$project_root/code/NeuroAdapter"

for result_dir in "$run_root"/official_metrics/*; do
  name=$(basename "$result_dir")
  echo "[$(date -Is)] evaluating $name"
  (
    cd "$code_root"
    CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH="$code_root" \
      conda run -n neuroadapter python metric_brain_adapter.py \
      --results_dir "$result_dir" --evaluation_mode subset --create_visualization
  )
  echo "[$(date -Is)] completed $name"
done
