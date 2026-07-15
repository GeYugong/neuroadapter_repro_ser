#!/usr/bin/env bash
# Compare two otherwise identical continuations from a checkpoint that contains
# AdamW state. One branch restores only model weights; the other restores both
# model weights and optimizer state.
set -euo pipefail

PROJECT_ROOT="/public/home/mty/GeYugong/projects/neuroadapter-iclr2026"
REPRO_ROOT="$PROJECT_ROOT/repro"
CODE_ROOT="$PROJECT_ROOT/code/NeuroAdapter"
OUTPUT_ROOT="/public/home/mty/GeYugong/outputs/neuroadapter"
LOG_ROOT="$PROJECT_ROOT/logs"
INIT_CHECKPOINT="$OUTPUT_ROOT/20260709-optimizer-state-save-smoke-2steps/checkpoint-step-21002.pt"
ADDITIONAL_STEPS="${1:-2000}"
STAMP="${2:-$(date +%Y%m%d)}"

MODEL_ONLY_RUN="${STAMP}-optimizer-ablation-model-only-21002-plus${ADDITIONAL_STEPS}"
STATEFUL_RUN="${STAMP}-optimizer-ablation-with-state-21002-plus${ADDITIONAL_STEPS}"

mkdir -p "$LOG_ROOT"
test -f "$INIT_CHECKPOINT"

run_branch() {
  local device="$1"
  local run_name="$2"
  local restore_optimizer="$3"
  local log_path="$LOG_ROOT/${run_name}.log"
  local args=(
    scripts/train_limited.py
    --run-name "$run_name"
    --init-checkpoint "$INIT_CHECKPOINT"
    --max-steps "$ADDITIONAL_STEPS"
    --topk 100
    --batch-size 4
    --gradient-accumulation-steps 2
    --lr 1e-5
    --save-every 500
    --mixed-precision no
    --cuda-device "$device"
  )

  if [[ "$restore_optimizer" == "true" ]]; then
    args+=(--resume-optimizer-state)
  fi

  (
    cd "$CODE_ROOT"
    export PYTHONNOUSERSITE=1
    export PYTHONPATH="$CODE_ROOT"
    export CUDA_VISIBLE_DEVICES="$device"
    exec conda run --no-capture-output -n neuroadapter python "$REPRO_ROOT/${args[0]}" "${args[@]:1}"
  ) >"$log_path" 2>&1 &
  echo "$! $run_name $log_path"
}

echo "Starting optimizer-resume ablation from $INIT_CHECKPOINT"
run_branch 3 "$MODEL_ONLY_RUN" false
run_branch 5 "$STATEFUL_RUN" true
