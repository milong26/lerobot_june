#!/bin/bash
# Evaluate all MiniVLA checkpoints in a training run
# Usage: bash eval_all_checkpoints_minivla.sh <gpu_id> <dataset_name> [checkpoint_dir]
#
# Auto-discovers all checkpoints in the training directory and evaluates each one.
# Output format matches differentvlm eval results.

set -e

GPU_ID=${1:-0}
DATASET_NAME=${2:-disassemble-v3_corner}
CHECKPOINT_DIR=${3:-""}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

# Auto-detect checkpoint dir if not provided
if [ -z "$CHECKPOINT_DIR" ]; then
    OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/minivla/grid_uniform_42_${DATASET_NAME}"
    CHECKPOINT_DIR="$OUTPUT_BASE_DIR/checkpoints"
fi

echo "=============================================="
echo "  MiniVLA Checkpoint Evaluation"
echo "=============================================="
echo "  GPU:            $GPU_ID"
echo "  Dataset:        $DATASET_NAME"
echo "  Checkpoint Dir: $CHECKPOINT_DIR"
echo "=============================================="
echo ""

export CUDA_VISIBLE_DEVICES=$GPU_ID
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

cd "$PROJECT_ROOT"

python personal/work2/differentvlm/minivla/scripts/eval_checkpoints.py \
    --gpu $GPU_ID \
    --dataset $DATASET_NAME \
    --checkpoint-dir "$CHECKPOINT_DIR"