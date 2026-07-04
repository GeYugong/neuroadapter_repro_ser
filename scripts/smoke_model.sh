#!/usr/bin/env bash
set -euo pipefail

PROJECT=/public/home/mty/GeYugong/code/NeuroAdapter
cd "$PROJECT"

PYTHONNOUSERSITE=1 PYTHONPATH="$PROJECT" conda run -n neuroadapter python brain_adapter/model.py
