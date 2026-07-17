#!/usr/bin/env bash
set -euo pipefail

project_root=/public/home/mty/GeYugong/projects/neuroadapter-iclr2026
mean_root="$project_root/outputs/roi_ablation/20260718-s01-mean-roi-50x50"
code_root="$project_root/code/NeuroAdapter"
checkpoint="$project_root/outputs/neuroadapter/20260707-topk100-bs4-ddp4-resume50000-to100000/checkpoint-step-100000.pt"
spec="$project_root/repro/experiments/roi_ablation/mapping/random_control_spec.json"

for _ in $(seq 1 720); do
  if [[ -f "$mean_root/run_summary.json" ]]; then
    break
  fi
  sleep 60
done
[[ -f "$mean_root/run_summary.json" ]] || { echo "mean robustness run did not complete in 12 hours"; exit 1; }

cd "$code_root"
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONPATH="$code_root" \
  conda run -n neuroadapter python "$project_root/repro/scripts/decode_roi_ablation_batch.py" \
  --checkpoint "$checkpoint" --condition-spec "$spec" \
  --run-name 20260718-s01-random-controls-20x8-50x50 \
  --num-samples 50 --denoising-steps 50 --condition-batch-size 8
