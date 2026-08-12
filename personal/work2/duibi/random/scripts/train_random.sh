#!/bin/bash
# Train with random subset
# Usage: bash train_random.sh <num_episodes> <seed> <gpu_id>

set -e

NUM_EPISODES=$1
SEED=$2
GPU_ID=$3

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUBSET_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/random/subsets"
OUTPUT_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/random"
DATASET_DIR="/data/zhonglinye/jun/lerobot/personal/work2/dataset"
LOG_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/random/logs"

# Create directories
mkdir -p "$SUBSET_DIR" "$OUTPUT_DIR" "$LOG_DIR"

# Select random episodes
echo "=== Selecting $NUM_EPISODES random episodes (seed=$SEED) ==="
python "$SCRIPT_DIR/select_random_episodes.py" \
    --num-episodes "$NUM_EPISODES" \
    --seed "$SEED" \
    --output-dir "$SUBSET_DIR"

# Load episode indices
SUBSET_FILE="$SUBSET_DIR/random_${NUM_EPISODES}_seed${SEED}.json"
EPISODES=$(python -c "import json; data=json.load(open('$SUBSET_FILE')); print('[' + ','.join(str(x) for x in data['selected_episode_indices']) + ']')")

# Output directory name
EXP_NAME="random_${NUM_EPISODES}_seed${SEED}"
EXP_OUTPUT_DIR="$OUTPUT_DIR/$EXP_NAME"

# Log file
LOG_FILE="$LOG_DIR/${EXP_NAME}.log"

echo "=== Training with $NUM_EPISODES episodes (seed=$SEED) on GPU $GPU_ID ==="
echo "Episodes: $EPISODES"
echo "Output: $EXP_OUTPUT_DIR"
echo "Log: $LOG_FILE"

# Training command
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=$GPU_ID

EXP_NAME="random_${NUM_EPISODES}_seed${SEED}"

lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --policy.empty_cameras=1 \
    --dataset.repo_id=lerobot/metaworld_pick_place \
    --dataset.root=$DATASET_DIR \
    --dataset.episodes="$EPISODES" \
    --dataset.eval_split=0.0 \
    --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
    --env.type=metaworld \
    --env.task=pick-place-v3 \
    --env.camera_name="corner2,behindGripper" \
    --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.train_state_proj=false \
    --policy.optimizer_lr=1e-4 \
    --steps=20000 \
    --batch_size=64 \
    --eval.n_episodes=100 \
    --eval.batch_size=1 \
    --env_eval_freq=2000 \
    --seed=$SEED \
    --job_name=smolvla_${EXP_NAME} \
    --output_dir=$EXP_OUTPUT_DIR \
    --remove_features='["observation.environment_state"]' \
    --wandb.enable=true \
    2>&1 | tee "$LOG_FILE"

echo "=== Training complete: $EXP_NAME ==="
echo "Log saved to: $LOG_FILE"