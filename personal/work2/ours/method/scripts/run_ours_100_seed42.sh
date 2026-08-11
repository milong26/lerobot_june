#!/bin/bash
# Train with SIC-selected subset (N=100, seed=42)
set -e

NUM_EPISODES=100
SEED=42
GPU_ID=${1:-0}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
METHOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SELECTION_FILE="$METHOD_DIR/results/subsets_by_budget.json"
OUTPUT_DIR="$METHOD_DIR/training_output/N${NUM_EPISODES}/seed${SEED}"
DATASET_DIR="/data/zhonglinye/jun/lerobot/personal/work2/dataset"
LOG_DIR="$METHOD_DIR/training_output/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

EPISODES=$(python -c "
import json
data = json.load(open('$SELECTION_FILE'))
eps = data.get('N_${NUM_EPISODES}', [])
print('[' + ','.join(str(x) for x in eps) + ']')
")

EXP_NAME="ours_${NUM_EPISODES}_seed${SEED}"
LOG_FILE="$LOG_DIR/${EXP_NAME}.log"

echo "=== Training with SIC-selected $NUM_EPISODES episodes (seed=$SEED) on GPU $GPU_ID ==="
echo "Episodes: $EPISODES"
echo "Output: $OUTPUT_DIR"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=$GPU_ID

lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/metaworld_pick_place \
    --dataset.root=$DATASET_DIR \
    --dataset.episodes="$EPISODES" \
    --dataset.eval_split=0.0 \
    --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
    --env.type=metaworld \
    --env.task=pick-place-v3 \
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
    --output_dir=$OUTPUT_DIR \
    --remove_features='["observation.environment_state"]' \
    --wandb.enable=true \
    2>&1 | tee "$LOG_FILE"

echo "=== Training complete: $EXP_NAME ==="