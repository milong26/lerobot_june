#!/bin/bash
# TinyVLA Temporal Aggregation Comparison Script
# Compares different action execution strategies using the same checkpoint

set -e

# Configuration
CHECKPOINT_PATH="personal/work2/differentvlm/tinyvla/tinyvla_s_random_ep200_seed42_disassemble-v3_corner/checkpoints/050000/pretrained_model"
ENV_TYPE="metaworld"
ENV_TASK="disassemble-v3"
CAMERA_NAMES="corner,gripperPOV"
BATCH_SIZE=8
N_EPISODES=50
GPU_ID=1
LOG_DIR="personal/work2/eval_model"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "TinyVLA Temporal Aggregation Comparison"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Episodes per config: $N_EPISODES"
echo "GPU: $GPU_ID"
echo "=========================================="

# Setup environment variables
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=$GPU_ID
export MUJOCO_EGL_DEVICE_ID=$GPU_ID

# Test 1: n_action_steps=16 (baseline)
echo ""
echo "[1/4] Testing n_action_steps=16 (baseline)..."
lerobot-eval \
    --policy.path=$CHECKPOINT_PATH \
    --env.type=$ENV_TYPE \
    --env.task=$ENV_TASK \
    --env.camera_name=$CAMERA_NAMES \
    --env.use_self_mw=true \
    --eval.batch_size=$BATCH_SIZE \
    --eval.n_episodes=$N_EPISODES \
    --policy.device=cuda \
    --policy.use_amp=false \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    2>&1 | tee "$LOG_DIR/eval_nact16_baseline.log"

# Test 2: n_action_steps=8
echo ""
echo "[2/4] Testing n_action_steps=8..."
lerobot-eval \
    --policy.path=$CHECKPOINT_PATH \
    --env.type=$ENV_TYPE \
    --env.task=$ENV_TASK \
    --env.camera_name=$CAMERA_NAMES \
    --env.use_self_mw=true \
    --eval.batch_size=$BATCH_SIZE \
    --eval.n_episodes=$N_EPISODES \
    --policy.device=cuda \
    --policy.use_amp=false \
    --policy.n_action_steps=8 \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    2>&1 | tee "$LOG_DIR/eval_nact8.log"

# Test 3: n_action_steps=4
echo ""
echo "[3/4] Testing n_action_steps=4..."
lerobot-eval \
    --policy.path=$CHECKPOINT_PATH \
    --env.type=$ENV_TYPE \
    --env.task=$ENV_TASK \
    --env.camera_name=$CAMERA_NAMES \
    --env.use_self_mw=true \
    --eval.batch_size=$BATCH_SIZE \
    --eval.n_episodes=$N_EPISODES \
    --policy.device=cuda \
    --policy.use_amp=false \
    --policy.n_action_steps=4 \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    2>&1 | tee "$LOG_DIR/eval_nact4.log"

# Test 4: n_action_steps=1 + temporal aggregation
echo ""
echo "[4/4] Testing n_action_steps=1 + temporal aggregation..."
python personal/work2/differentvlm/tinyvla/eval_tinyvla_temporal_aggregation.py \
    --policy_path=$CHECKPOINT_PATH \
    --env_type=$ENV_TYPE \
    --env_task=$ENV_TASK \
    --env_camera_name=$CAMERA_NAMES \
    --env_use_self_mw=true \
    --eval_batch_size=$BATCH_SIZE \
    --eval_n_episodes=$N_EPISODES \
    --policy_device=cuda \
    --policy_use_amp=false \
    --chunk_size=16 \
    --aggregation_decay=0.01 \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' \
    2>&1 | tee "$LOG_DIR/eval_nact1_temporal_agg.log"

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
echo "Results saved to:"
echo "  - $LOG_DIR/eval_nact16_baseline.log"
echo "  - $LOG_DIR/eval_nact8.log"
echo "  - $LOG_DIR/eval_nact4.log"
echo "  - $LOG_DIR/eval_nact1_temporal_agg.log"
echo ""
echo "To view success rates:"
echo "  grep -i 'success' $LOG_DIR/eval_*.log"
echo "=========================================="