#!/bin/bash
# Train and evaluate with selected episodes
# Usage: bash train_and_eval.sh <mode> <num_episodes> <seed> <gpu_id> <output_base_dir>
#   mode: uniform, random, dynamicanchor, or subzerocore

set -e

MODE=$1
NUM_EPISODES=${2:-112}
SEED=$3
GPU_ID=$4
OUTPUT_BASE_DIR=$5
DATASET_NAME=${6:-corner3}

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_${DATASET_NAME}"
OUR_DIR="/data/zhonglinye/jun/lerobot/personal/work2/our"
SUBSET_DIR="$OUTPUT_BASE_DIR/subsets"
LOG_DIR="$OUTPUT_BASE_DIR/logs"
EVAL_DIR="$OUTPUT_BASE_DIR/eval_results"

# Create directories
mkdir -p "$SUBSET_DIR" "$OUTPUT_BASE_DIR" "$LOG_DIR" "$EVAL_DIR"

# Experiment name
if [ "$MODE" = "uniform" ]; then
    EXP_NAME="${MODE}_${NUM_EPISODES}_seed${SEED}"
    SELECT_SCRIPT="$SCRIPT_DIR/select_uniform_episodes.py"
elif [ "$MODE" = "random" ]; then
    EXP_NAME="${MODE}_${NUM_EPISODES}_seed${SEED}"
    SELECT_SCRIPT="$SCRIPT_DIR/select_random_episodes.py"
elif [ "$MODE" = "dynamicanchor" ]; then
    EXP_NAME="${MODE}_${NUM_EPISODES}_seed${SEED}"
    SELECT_SCRIPT=""
elif [ "$MODE" = "subzerocore" ]; then
    EXP_NAME="${MODE}_${NUM_EPISODES}_seed${SEED}"
    SELECT_SCRIPT=""
else
    echo "Error: mode must be 'uniform', 'random', 'dynamicanchor', or 'subzerocore'"
    exit 1
fi

EXP_OUTPUT_DIR="$OUTPUT_BASE_DIR/$EXP_NAME"
LOG_FILE="$LOG_DIR/${EXP_NAME}.log"
EVAL_FILE="$EVAL_DIR/${EXP_NAME}_eval.json"

SUBSET_FILE="$SUBSET_DIR/${MODE}_${NUM_EPISODES}_seed${SEED}.json"

# Auto-detect checkpoint for resume
LATEST_CKPT=$(ls -d "$EXP_OUTPUT_DIR"/checkpoints/*/ 2>/dev/null | sort -V | tail -n1)
RESUME_ARGS=""
if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT/pretrained_model/train_config.json" ]; then
    echo "=== Found existing checkpoint: $LATEST_CKPT ==="
    echo "=== Resuming training from checkpoint ==="
    RESUME_ARGS="--resume=true --config_path=$LATEST_CKPT/pretrained_model"
    echo "=== Loading existing episode subset: $SUBSET_FILE ==="
else
    echo "=== No checkpoint found, starting fresh training ==="
    
    if [ "$MODE" = "dynamicanchor" ]; then
        # DynamicAnchor: extract embeddings, run offline simulation, select episodes
        echo "=== Running DynamicAnchor pipeline ==="
        
        EMBEDDINGS_DIR="$OUTPUT_BASE_DIR/embeddings"
        RESULTS_DIR="$OUTPUT_BASE_DIR/results"
        PLANNING_RESULT="$RESULTS_DIR/planning_result.json"
        
        mkdir -p "$EMBEDDINGS_DIR" "$RESULTS_DIR"
        
        # Step 1: Extract embeddings
        echo "=== Step 1: Extracting VLM embeddings ==="
        python "$OUR_DIR/embeddings/extract_embeddings.py" \
            --dataset-dir "$DATASET_DIR" \
            --output-dir "$EMBEDDINGS_DIR" \
            --n-components 32 \
            --device cuda
        
        # Step 2: Run iterative SIC selection
        echo "=== Step 2: Running iterative SIC episode selection ==="
        python "$OUR_DIR/experiments/iterative_select_episodes.py" \
            --embeddings-dir "$EMBEDDINGS_DIR" \
            --output-dir "$RESULTS_DIR" \
            --b0-size 18 \
            --target-size "$NUM_EPISODES" \
            --n-add-per-round 9 \
            --seed "$SEED" \
            --alpha 1.0 \
            --lambda-wrist 1.0
        
        # Copy subset file to expected location
        cp "$RESULTS_DIR/subset.json" "$SUBSET_FILE"
        echo "=== Iterative selection complete ==="
    elif [ "$MODE" = "subzerocore" ]; then
        # SubZeroCore: load embeddings, run facility-location selection
        echo "=== Running SubZeroCore pipeline ==="

        SUBZEROCORE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/SubZeroCore"
        EMBEDDINGS_DIR="$OUTPUT_BASE_DIR/embeddings"
        RESULTS_DIR="$OUTPUT_BASE_DIR/results"

        mkdir -p "$EMBEDDINGS_DIR" "$RESULTS_DIR"

        # Step 1: Extract embeddings (reuse our_v4 embedding extraction)
        echo "=== Step 1: Extracting VLM embeddings ==="
        python "$OUR_DIR/embeddings/extract_embeddings.py" \
            --dataset-dir "$DATASET_DIR" \
            --output-dir "$EMBEDDINGS_DIR" \
            --n-components 32 \
            --device cuda

        # Step 2: Run SubZeroCore selection
        echo "=== Step 2: Running SubZeroCore episode selection ==="
        python "$SUBZEROCORE_DIR/experiments/select_episodes_subzerocore.py" \
            --dataset-root "$DATASET_DIR" \
            --embedding-dir "$EMBEDDINGS_DIR" \
            --output-dir "$OUTPUT_BASE_DIR" \
            --num-selected "$NUM_EPISODES" \
            --seed "$SEED"

        # Copy subset file to expected location
        cp "$OUTPUT_BASE_DIR/subsets/subzerocore_${NUM_EPISODES}_seed${SEED}.json" "$SUBSET_FILE"
        echo "=== SubZeroCore selection complete ==="
    else
        # Select episodes using uniform or random
        echo "=== Selecting $NUM_EPISODES $MODE episodes (seed=$SEED) ==="
        python "$SELECT_SCRIPT" \
            --num-episodes "$NUM_EPISODES" \
            --seed "$SEED" \
            --dataset-root "$DATASET_DIR" \
            --output-dir "$SUBSET_DIR"
    fi
fi

# Load episode indices
EPISODES=$(python -c "import json; data=json.load(open('$SUBSET_FILE')); print('[' + ','.join(str(x) for x in data['selected_episode_indices']) + ']')")

echo "=== Training with $NUM_EPISODES $MODE episodes (seed=$SEED) on GPU $GPU_ID ==="
echo "Episodes: $EPISODES"
echo "Output: $EXP_OUTPUT_DIR"
echo "Log: $LOG_FILE"

# Training command
# Fix torchcodec/FFmpeg library loading issues
# 之前训练的时候一直报错
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

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
    --env.camera_name="corner,gripperPOV" \
    --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.train_state_proj=false \
    --policy.optimizer_lr=1e-4 \
    --save_freq=2000 \
    --steps=12000 \
    --batch_size=64 \
    --num_workers=16 \
    --eval.n_episodes=16 \
    --eval.batch_size=16 \
    --env_eval_freq=12000 \
    --seed=$SEED \
    --job_name=smolvla_${EXP_NAME} \
    --output_dir=$EXP_OUTPUT_DIR \
    --remove_features='["observation.environment_state"]' \
    --wandb.enable=true \
    $RESUME_ARGS \
    2>&1 | tee -a "$LOG_FILE"

echo "=== Training complete: $EXP_NAME ==="
echo "Log saved to: $LOG_FILE"

# Extract eval results from log
echo "=== Extracting evaluation results ==="
python -c "
import json
import re

log_file = '$LOG_FILE'
eval_file = '$EVAL_FILE'

success_rates = []
with open(log_file, 'r') as f:
    for line in f:
        if 'eval' in line.lower() and 'success' in line.lower():
            match = re.search(r'success_rate[:\s]+([0-9.]+)', line)
            if match:
                success_rates.append(float(match.group(1)))

result = {
    'experiment': '$EXP_NAME',
    'mode': '$MODE',
    'num_episodes': $NUM_EPISODES,
    'seed': $SEED,
    'gpu_id': $GPU_ID,
    'success_rates': success_rates,
    'final_success_rate': success_rates[-1] if success_rates else None,
    'max_success_rate': max(success_rates) if success_rates else None,
    'avg_success_rate': sum(success_rates) / len(success_rates) if success_rates else None
}

with open(eval_file, 'w') as f:
    json.dump(result, f, indent=2)

print(f'Evaluation results saved to: {eval_file}')
if success_rates:
    print(f'Final success rate: {success_rates[-1]:.4f}')
    print(f'Max success rate: {max(success_rates):.4f}')
"

echo "=== All complete: $EXP_NAME ==="
echo "Results saved to: $OUTPUT_BASE_DIR"