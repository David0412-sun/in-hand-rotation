#!/usr/bin/env bash
set -euo pipefail

# Same task/object overrides as teacher_rapidhand.sh; no training is started.
GPU_ID=${1:-0}
if [ "$#" -gt 0 ]; then
    shift
fi
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

CUDA_VISIBLE_DEVICES="$GPU_ID" python isaacgymenvs/collect.py \
    task=RapidHandRotation test=True headless=True force_render=False \
    task.env.numEnvs=1 task.dataset.enabled=True \
    train.params.config.player.deterministic=True \
    wandb_activate=False "$@"
