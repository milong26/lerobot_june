#!/usr/bin/env bash
# One-command MetaWorld 3-task selection + joint VLA training.
#
# Usage:
#   bash personal/work2/metaworld_pipeline/run_metaworld_all.sh \
#     METHOD MODEL [EPISODES_PER_TASK] [GPU] [SEED] [VISUAL_VARIANT] [MODE] [OUR_V6_ABLATION]
#
# METHOD:
#   our_v6 | random | grid_uniform | deminf | fps
# MODEL:
#   smovla | smolvla | minivla | tinyvla_s | tinyvla_b
# MODE:
#   full | selection-only
# OUR_V6_ABLATION:
#   full | wo_action | wo_adaptive_priority
#
# Examples:
#   bash personal/work2/metaworld_pipeline/run_metaworld_all.sh our_v6 smovla 112 0 42 l2 full full
#   bash personal/work2/metaworld_pipeline/run_metaworld_all.sh random minivla 100 0 42 l2 full
#   bash personal/work2/metaworld_pipeline/run_metaworld_all.sh grid_uniform tinyvla_s 112 0 42 l2 full
#   bash personal/work2/metaworld_pipeline/run_metaworld_all.sh fps smolvla 112 0 42 l2 selection-only

set -euo pipefail

METHOD=${1:-our_v6}
MODEL=${2:-smovla}
EPISODES=${3:-112}
GPU_ID=${4:-0}
SEED=${5:-42}
VISUAL_VARIANT=${6:-l2}
MODE=${7:-full}
OUR_V6_ABLATION=${8:-full}

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook 2>/dev/null)" || true
  conda activate lb_server || true
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="$GPU_ID"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO_ROOT/personal/work2${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
fi

ARGS=(
  --method "$METHOD"
  --model "$MODEL"
  --episodes-per-task "$EPISODES"
  --gpu-id "$GPU_ID"
  --seed "$SEED"
  --visual-variant "$VISUAL_VARIANT"
  --our-v6-ablation "$OUR_V6_ABLATION"
)

if [[ "$MODE" == "selection-only" ]]; then
  ARGS+=(--selection-only)
elif [[ "$MODE" != "full" ]]; then
  echo "ERROR: MODE must be full or selection-only" >&2
  exit 2
fi

python personal/work2/metaworld_pipeline/run_metaworld_all.py "${ARGS[@]}"
