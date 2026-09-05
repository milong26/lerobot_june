#!/bin/bash
# Train and evaluate with selected episodes
# Usage: bash train_and_eval.sh <mode> <num_episodes> <seed> <gpu_id> <output_base_dir>
#   mode: uniform, random, dynamicanchor, subzerocore, or deminf

set -e

MODE=$1
NUM_EPISODES=${2:-112}
SEED=$3
GPU_ID=$4
OUTPUT_BASE_DIR=${5:-}
DATASET_NAME=${6:-corner3}

# Auto-generate OUTPUT_BASE_DIR if not provided
if [ -z "$OUTPUT_BASE_DIR" ]; then
    OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/${MODE}_${NUM_EPISODES}_seed${SEED}_${DATASET_NAME}"
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATASET_DIR="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/${DATASET_NAME}"
OUR_DIR="/data/zhonglinye/jun/lerobot/personal/work2/our"
SUBSET_DIR="$OUTPUT_BASE_DIR/subsets"
LOG_DIR="$OUTPUT_BASE_DIR/logs"
EVAL_DIR="$OUTPUT_BASE_DIR/eval_results"

# Dataset path configuration file (optional override)
DATASET_PATH_FILE="$OUTPUT_BASE_DIR/dataset_path.txt"

# Create directories (only for logs, subsets, and eval results - not EXP_OUTPUT_DIR)
# lerobot-train will automatically create EXP_OUTPUT_DIR for model checkpoints
mkdir -p "$SUBSET_DIR" "$LOG_DIR" "$EVAL_DIR"

# Override DATASET_DIR if path file exists
if [ -f "$DATASET_PATH_FILE" ]; then
    DATASET_DIR="$(cat "$DATASET_PATH_FILE")"
    echo "Using dataset path from config file: $DATASET_DIR"
fi

# Experiment name
if [ "$MODE" = "uniform" ]; then
    EXP_NAME="${MODE}_${NUM_EPISODES}_seed${SEED}"
    SELECT_SCRIPT="$SCRIPT_DIR/select_uniform_episodes.py"
elif [ "$MODE" = "state_uniform" ]; then
    EXP_NAME="${MODE}_${NUM_EPISODES}_seed${SEED}"
    SELECT_SCRIPT="$SCRIPT_DIR/select_new_uniform.py"
elif [ "$MODE" = "random" ]; then
    EXP_NAME="${MODE}_${NUM_EPISODES}_seed${SEED}"
    SELECT_SCRIPT="$SCRIPT_DIR/select_random_episodes.py"
elif [ "$MODE" = "dynamicanchor" ]; then
    EXP_NAME="${MODE}_${NUM_EPISODES}_seed${SEED}"
    SELECT_SCRIPT=""
elif [ "$MODE" = "subzerocore" ]; then
    EXP_NAME="${MODE}_${NUM_EPISODES}_seed${SEED}"
    SELECT_SCRIPT=""
elif [ "$MODE" = "deminf" ]; then
    EXP_NAME="${MODE}_${NUM_EPISODES}_seed${SEED}"
    SELECT_SCRIPT=""
elif [ "$MODE" = "our_v5" ]; then
    EXP_NAME="${MODE}_${NUM_EPISODES}_seed${SEED}"
    SELECT_SCRIPT=""
else
    echo "Error: mode must be 'uniform', 'state_uniform', 'random', 'dynamicanchor', 'subzerocore', 'deminf', or 'our_v5'"
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
        # DynamicAnchor: use shared embedding cache, run offline simulation, select episodes
        echo "=== Running DynamicAnchor pipeline ==="
        
        EMBEDDING_UTIL="/data/zhonglinye/jun/lerobot/personal/work2/embedding_utils/ensure_embeddings.py"
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
        # SubZeroCore: use shared embedding cache, run facility-location selection
        echo "=== Running SubZeroCore pipeline ==="

        SUBZEROCORE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/SubZeroCore"
        EMBEDDING_UTIL="/data/zhonglinye/jun/lerobot/personal/work2/embedding_utils/ensure_embeddings.py"
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

        # Step 2: Run SubZeroCore selection
        echo "=== Step 2: Running SubZeroCore episode selection ==="
        python "$SUBZEROCORE_DIR/experiments/select_episodes_subzerocore.py" \
            --dataset-root "$DATASET_DIR" \
            --embedding-dir "$EMBEDDINGS_DIR" \
            --output-dir "$OUTPUT_BASE_DIR" \
            --num-selected "$NUM_EPISODES" \
            --seed "$SEED"

        # Verify subset file exists (SubZeroCore outputs directly to SUBSET_DIR)
        if [ ! -f "$SUBSET_FILE" ]; then
            echo "Error: SubZeroCore subset file not found: $SUBSET_FILE"
            exit 1
        fi
        echo "=== SubZeroCore selection complete ==="
    elif [ "$MODE" = "deminf" ]; then
        # DemInf: run selection first, then train
        echo "=== Running DemInf pipeline ==="
        
        DEMINF_DIR="/data/zhonglinye/jun/lerobot/personal/work2/deminf"
        DEMINF_OUTPUT_DIR="/data/zhonglinye/jun/lerobot/personal/work2/deminf_results/${DATASET_NAME}"
        DEMINF_SUBSET="$DEMINF_OUTPUT_DIR/subsets/deminf_${NUM_EPISODES}_seed${SEED}.json"
        
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        
        # Step 1: Run DemInf selection if subset doesn't exist
        if [ ! -f "$DEMINF_SUBSET" ]; then
            echo "=== Step 1: Running DemInf episode selection ==="
            python "$DEMINF_DIR/run_deminf.py" \
                --dataset-path "$DATASET_DIR" \
                --output-dir "$DEMINF_OUTPUT_DIR" \
                --target-episodes "$NUM_EPISODES" \
                --seed "$SEED" \
                --device "cuda"
            
            if [ $? -ne 0 ]; then
                echo "ERROR: DemInf selection failed"
                exit 1
            fi
            
            if [ ! -f "$DEMINF_SUBSET" ]; then
                echo "Error: DemInf subset file not generated: $DEMINF_SUBSET"
                exit 1
            fi
            echo "=== DemInf selection complete ==="
        else
            echo "=== Using pre-computed DemInf subset ==="
        fi
        
        cp "$DEMINF_SUBSET" "$SUBSET_FILE"
        echo "Copied DemInf subset from: $DEMINF_SUBSET"
        echo "=== DemInf subset ready ==="
    elif [ "$MODE" = "our_v5" ]; then
        # Our V5: use vision-action embedding, run iterative SIC selection
        echo "=== Running Our V5 pipeline ==="

        EMBEDDING_UTIL="/data/zhonglinye/jun/lerobot/personal/work2/embedding_utils/ensure_embeddings_v5.py"
        EMBEDDING_PATH_FILE="$OUTPUT_BASE_DIR/shared_embedding_path.txt"
        RESULTS_DIR="$OUTPUT_BASE_DIR/results"

        mkdir -p "$RESULTS_DIR"

        # Step 1: Ensure V5 embedding cache exists
        echo "=== Step 1: Ensuring V5 shared embedding cache ==="
        export CUDA_VISIBLE_DEVICES=$GPU_ID

        python "$EMBEDDING_UTIL" \
            --dataset-root "$DATASET_DIR" \
            --dataset-name "$DATASET_NAME" \
            --gpu-id "$GPU_ID" \
            --pca-dim 32 \
            --path-file "$EMBEDDING_PATH_FILE"

        if [ $? -ne 0 ]; then
            echo "ERROR: ensure_embeddings_v5.py failed"
            exit 1
        fi

        EMBEDDINGS_DIR="$(cat "$EMBEDDING_PATH_FILE")"
        echo "V5 shared embedding directory: $EMBEDDINGS_DIR"

        if [ ! -d "$EMBEDDINGS_DIR" ]; then
            echo "ERROR: V5 shared embedding directory does not exist: $EMBEDDINGS_DIR"
            exit 1
        fi

        # Step 2: Run V5 episode selection
        echo "=== Step 2: Running V5 episode selection ==="
        V5_SELECT_SCRIPT="/data/zhonglinye/jun/lerobot/personal/work2/our_v5/select_episodes_v5.py"
        python "$V5_SELECT_SCRIPT" \
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
        echo "=== V5 selection complete ==="
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

# Extract task name from dataset name
# e.g., "disassemble-v3_corner" -> "disassemble-v3"
# e.g., "pick_place_corner3" -> "pick-place-v3"
# Remove common view suffixes: _corner, _top, _corner3, _gripper, etc.
DATASET_BASE=$(echo "$DATASET_NAME" | sed -E 's/_(corner|top|gripper|left|right|front|back|view)[0-9]*$//')
# Convert underscores to hyphens
TASK_BASE=$(echo "$DATASET_BASE" | tr '_' '-')
# Add -v3 if not already present
if [[ "$TASK_BASE" != *"v3"* ]] && [[ "$TASK_BASE" != *"v2"* ]]; then
    TASK_NAME="${TASK_BASE}-v3"
else
    TASK_NAME="$TASK_BASE"
fi

echo "Dataset name: $DATASET_NAME"
echo "Extracted task name: $TASK_NAME"

# Set camera names based on dataset name
if [[ "$DATASET_NAME" == *"corner"* ]]; then
    CAMERA_NAMES="corner,gripperPOV"
elif [[ "$DATASET_NAME" == *"top"* ]]; then
    CAMERA_NAMES="top,gripperPOV"
elif [[ "$DATASET_NAME" == *"left"* ]]; then
    CAMERA_NAMES="left,gripperPOV"
elif [[ "$DATASET_NAME" == *"right"* ]]; then
    CAMERA_NAMES="right,gripperPOV"
else
    CAMERA_NAMES="corner,gripperPOV"
fi

echo "Camera names: $CAMERA_NAMES"

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
    --env.task=$TASK_NAME \
    --env.camera_name="$CAMERA_NAMES" \
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