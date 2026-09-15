#!/usr/bin/env bash
# One-command RoboMME 3-task selection + joint training.
# Usage:
#   bash personal/work2/robomme_pipeline/run_robomme_all.sh METHOD MODEL [EPISODES] [GPU] [SEED] [VISUAL_VARIANT] [MODE]
# Examples:
#   bash personal/work2/robomme_pipeline/run_robomme_all.sh our_v6 smovla 100 0 42 l2 full
#   bash personal/work2/robomme_pipeline/run_robomme_all.sh random minivla 100 0 42 l2 full
#   bash personal/work2/robomme_pipeline/run_robomme_all.sh grid-uniform smovla 100 0 42 l2 selection-only
set -euo pipefail

METHOD=${1:-our_v6}
MODEL=${2:-smovla}
EPISODES=${3:-100}
GPU_ID=${4:-0}
SEED=${5:-42}
VISUAL_VARIANT=${6:-l2}
MODE=${7:-full}

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

ARGS=(
  --method "$METHOD"
  --model "$MODEL"
  --episodes-per-task "$EPISODES"
  --gpu-id "$GPU_ID"
  --seed "$SEED"
  --visual-variant "$VISUAL_VARIANT"
)

if [[ "$MODE" == "selection-only" ]]; then
  ARGS+=(--selection-only)
elif [[ "$MODE" != "full" ]]; then
  echo "ERROR: MODE must be full or selection-only" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export SAPIEN_VULKAN_DEVICE_INDEX="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/personal/work2:${PYTHONPATH:-}"

python personal/work2/robomme_pipeline/run_robomme_all.py "${ARGS[@]}"
