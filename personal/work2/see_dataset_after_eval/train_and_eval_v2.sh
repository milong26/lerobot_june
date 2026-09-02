#!/bin/bash
# Train and evaluate with selected episodes (V2 support)
# Usage: bash train_and_eval_v2.sh <mode> <num_episodes> <seed> <gpu_id> <output_base_dir>
#   mode: uniform, random, dynamicanchor, or dynamicanchor_v2

set -e

MODE=$1
NUM_EPISODES=${2:-112}
SEED=$3
GPU_ID=$4
OUTPUT_BASE_DIR=$5
DATASET_NAME=${6:-corner3}

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK2_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATASET_DIR="$WORK2_ROOT/dataset_view/pick_place_${DATASET_NAME}"
OUR_DIR="$WORK2_ROOT/our"
V2_DIR="$WORK2_ROOT/see_dataset_after_eval"
TRAIN_SCRIPT_DIR="$WORK2_ROOT/duibi/train_and_eval_scripts"
SUBSET_DIR="$OUTPUT_BASE_DIR/subsets"
LOG_DIR="$OUTPUT_BASE_DIR/logs"
EVAL_DIR="$OUTPUT_BASE_DIR/eval_results"

# Create directories
mkdir -p "$SUBSET_DIR" "$OUTPUT_BASE_DIR" "$LOG_DIR" "$EVAL_DIR"

# Experiment name
if [ "$MODE" = "uniform" ]; then
    EXP_NAME="${MODE}_${NUM_EPISODES}_seed${SEED}"
    SELECT_SCRIPT="$TRAIN_SCRIPT_DIR/select_uniform_episodes.py"
elif [ "$MODE" = "random" ]; then
    EXP_NAME="${MODE}_${NUM_EPISODES}_seed${SEED}"
    SELECT_SCRIPT="$TRAIN_SCRIPT_DIR/select_random_episodes.py"
elif [ "$MODE" = "dynamicanchor" ]; then
    EXP_NAME="${MODE}_${NUM_EPISODES}_seed${SEED}"
    SELECT_SCRIPT=""
elif [ "$MODE" = "dynamicanchor_v2" ]; then
    EXP_NAME="${MODE}_${NUM_EPISODES}_seed${SEED}"
    SELECT_SCRIPT=""
else
    echo "Error: mode must be 'uniform', 'random', 'dynamicanchor', or 'dynamicanchor_v2'"
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
        # DynamicAnchor V1: use shared embedding cache, run offline simulation, select episodes
        echo "=== Running DynamicAnchor V1 pipeline ==="

        EMBEDDING_UTIL="$WORK2_ROOT/embedding_utils/ensure_embeddings.py"
        EMBEDDING_PATH_FILE="$OUTPUT_BASE_DIR/shared_embedding_path.txt"
        RESULTS_DIR="$OUTPUT_BASE_DIR/results"

        mkdir -p "$RESULTS_DIR"

        # Step 1: Ensure shared embedding cache exists
        echo "=== Step 1: Ensuring shared embedding cache ==="
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        
        python "$EMBEDDING_UTIL" \
            --dataset-root "$DATASET_DIR" \
            --dataset-name "$DATASET_NAME" \
            --gpu-id "$GPU_ID" \
            --pca-dim 32 \
            --path-file "$EMBEDDING_PATH_FILE"
        
        if [ $? -ne 0 ]; then
            echo "ERROR: ensure_embeddings.py failed"
            exit 1
        fi
        
        EMBEDDINGS_DIR="$(cat "$EMBEDDING_PATH_FILE")"
        echo "Shared embedding directory: $EMBEDDINGS_DIR"
        
        if [ ! -d "$EMBEDDINGS_DIR" ]; then
            echo "ERROR: Shared embedding directory does not exist: $EMBEDDINGS_DIR"
            exit 1
        fi

        # Step 2: Run iterative SIC selection (V1)
        echo "=== Step 2: Running iterative SIC episode selection (V1) ==="
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
        echo "=== V1 Iterative selection complete ==="

    elif [ "$MODE" = "dynamicanchor_v2" ]; then
        # DynamicAnchor V2: use shared embedding cache, run V2 selection
        echo "=== Running DynamicAnchor V2 pipeline ==="

        EMBEDDING_UTIL="$WORK2_ROOT/embedding_utils/ensure_embeddings.py"
        EMBEDDING_PATH_FILE="$OUTPUT_BASE_DIR/shared_embedding_path.txt"
        RESULTS_DIR="$OUTPUT_BASE_DIR/results"

        mkdir -p "$RESULTS_DIR"

        # Step 1: Ensure shared embedding cache exists
        echo "=== Step 1: Ensuring shared embedding cache ==="
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        
        python "$EMBEDDING_UTIL" \
            --dataset-root "$DATASET_DIR" \
            --dataset-name "$DATASET_NAME" \
            --gpu-id "$GPU_ID" \
            --pca-dim 32 \
            --path-file "$EMBEDDING_PATH_FILE"
        
        if [ $? -ne 0 ]; then
            echo "ERROR: ensure_embeddings.py failed"
            exit 1
        fi
        
        EMBEDDINGS_DIR="$(cat "$EMBEDDING_PATH_FILE")"
        echo "Shared embedding directory: $EMBEDDINGS_DIR"
        
        if [ ! -d "$EMBEDDINGS_DIR" ]; then
            echo "ERROR: Shared embedding directory does not exist: $EMBEDDINGS_DIR"
            exit 1
        fi

        # Step 2: Run iterative SIC selection (V2)
        echo "=== Step 2: Running fixed-anchor sequential SIC episode selection (V2) ==="
        python "$V2_DIR/iterative_select_episodes_v2.py" \
            --embeddings-dir "$EMBEDDINGS_DIR" \
            --output-dir "$RESULTS_DIR" \
            --b0-size 18 \
            --target-size "$NUM_EPISODES" \
            --n-add-per-round 9 \
            --seed "$SEED" \
            --alpha 1.0 \
            --lambda-wrist 1.0 \
            --b0-strategy random

        # Copy subset file to expected location
        cp "$RESULTS_DIR/subset.json" "$SUBSET_FILE"
        echo "=== V2 Iterative selection complete ==="
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
    --steps=16000 \
    --batch_size=64 \
    --num_workers=16 \
    --eval.n_episodes=16 \
    --eval.batch_size=16 \
    --env_eval_freq=16000 \
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