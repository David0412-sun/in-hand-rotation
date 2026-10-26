#!/usr/bin/env bash
set -euo pipefail

# Run from any directory; remaining arguments are ordinary Hydra overrides.
# task.env.objSet=cross/C/custom selects one object; ball selects two balls.
# Object presets, e.g. +object=box_50x50x50, also set a matching initial pose.
GPU_ID=${1:-0}
if [ "$#" -gt 0 ]; then
    shift
fi
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES="$GPU_ID" python isaacgymenvs/train.py \
    task=RapidHandRotation \
    headless=True wandb_activate=False \
    "$@"
