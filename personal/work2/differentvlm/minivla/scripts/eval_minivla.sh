#!/bin/bash
# Evaluate MiniVLA checkpoints
# Usage: bash eval_minivla.sh <gpu_id> <dataset_name> <checkpoint_dir>
#
# Evaluates all MiniVLA checkpoints in the specified directory.
# Results are saved to the experiment's eval_results directory.

set -e

GPU_ID=${1:-0}
DATASET_NAME=${2:-disassemble-v3_corner}
CHECKPOINT_DIR=${3:-""}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

if [ -z "$CHECKPOINT_DIR" ]; then
    # Default to the latest experiment directory
    EXP_BASE="/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/minivla/experiments"
    LATEST_EXP=$(ls -td "$EXP_BASE"/*/ 2>/dev/null | head -1)
    if [ -z "$LATEST_EXP" ]; then
        echo "ERROR: No experiment directories found in $EXP_BASE"
        exit 1
    fi
    CHECKPOINT_DIR="$LATEST_EXP/checkpoints"
    echo "Using latest experiment: $LATEST_EXP"
fi

echo "=============================================="
echo "  MiniVLA Evaluation"
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

echo ""
echo "Evaluation complete."