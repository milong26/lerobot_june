#!/bin/bash
# TinyVLA Training Script
# Usage: bash train_tinyvla.sh <policy_type> <num_episodes> <seed> <gpu_id> <output_base_dir> <dataset_name> <selection_method> <subset_file>

set -e

POLICY_TYPE=$1          # tinyvla_s or tinyvla_b
NUM_EPISODES=$2
SEED=$3
GPU_ID=$4
OUTPUT_BASE_DIR=$5
DATASET_NAME=$6
SELECTION_METHOD=$7
SUBSET_FILE=$8

EXP_NAME="${POLICY_TYPE}_${SELECTION_METHOD}_ep${NUM_EPISODES}_seed${SEED}"
LOG_DIR="$OUTPUT_BASE_DIR/logs"
TIME_FILE="$LOG_DIR/$EXP_NAME.time"
PID_FILE="$LOG_DIR/$EXP_NAME.pid"
DATASET_ROOT="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/${DATASET_NAME}"

mkdir -p "$LOG_DIR"

echo $$ > "$PID_FILE"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" > "$TIME_FILE"
echo "PID: $$" >> "$TIME_FILE"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~="
echo "Experiment: $EXP_NAME"
echo "Policy: $POLICY_TYPE"
echo "Selection: $SELECTION_METHOD"
echo "GPU: $GPU_ID"
echo "Episodes: $NUM_EPISODES"
echo "Seed: $SEED"
echo "Dataset: $DATASET_NAME"
echo "Dataset Root: $DATASET_ROOT"
echo "Subset File: $SUBSET_FILE"
echo "PID: $$"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~="
echo ""

# Read episode indices from subset file
EPISODES=$(python3 -c "import json; data=json.load(open('$SUBSET_FILE')); print('[' + ','.join(str(x) for x in data['selected_episodes']) + ']')")

echo "Episode indices: $EPISODES"
echo ""

# Training parameters (fixed)
TRAIN_STEPS=50000
SAVE_FREQ=2000
ENV_EVAL_FREQ=2000
LOG_FREQ=200
BATCH_SIZE=4
NUM_WORKERS=4
WARMUP_STEPS=250       # 0.5% of 50k
DECAY_STEPS=50000
DECAY_LR=2.5e-6
PEAK_LR=2e-4
EVAL_SPLIT=0

echo "~~~ Starting TinyVLA Training ~~~"
echo "Steps: $TRAIN_STEPS"
echo "Save freq: $SAVE_FREQ"
echo "Env eval freq: $ENV_EVAL_FREQ"
echo "Log freq: $LOG_FREQ"
echo "Batch size: $BATCH_SIZE"
echo "Warmup steps: $WARMUP_STEPS"
echo "Decay steps: $DECAY_STEPS"
echo ""

export CUDA_VISIBLE_DEVICES=$GPU_ID
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

cd /data/zhonglinye/jun/lerobot

lerobot-train \
  --policy.type=$POLICY_TYPE \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.repo_id=1/2 \
  --dataset.root=$DATASET_ROOT \
  --dataset.episodes=$EPISODES \
  --dataset.eval_split=$EVAL_SPLIT \
  --env.type=metaworld \
  --env.task=disassemble-v3 \
  --env.camera_name=corner,gripperPOV \
  --env.use_self_mw=true \
  --policy.optimizer_lr=$PEAK_LR \
  --policy.optimizer_weight_decay=0 \
  --policy.scheduler_warmup_steps=$WARMUP_STEPS \
  --policy.scheduler_decay_steps=$DECAY_STEPS \
  --policy.scheduler_decay_lr=$DECAY_LR \
  --steps=$TRAIN_STEPS \
  --batch_size=$BATCH_SIZE \
  --num_workers=$NUM_WORKERS \
  --log_freq=$LOG_FREQ \
  --save_freq=$SAVE_FREQ \
  --env_eval_freq=$ENV_EVAL_FREQ \
  --seed=$SEED \
  --job_name=$EXP_NAME \
  --output_dir=$OUTPUT_BASE_DIR/checkpoints \
  --wandb.enable=true \
  --remove_features='["observation.environment_state"]' \
  2>&1 | tee -a "$LOG_DIR/$EXP_NAME.log"

echo "" >> "$TIME_FILE"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$TIME_FILE"
echo "Status: completed" >> "$TIME_FILE"

echo ""
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~="
echo "Training completed: $EXP_NAME"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~="