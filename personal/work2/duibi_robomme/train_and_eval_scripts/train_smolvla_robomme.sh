#!/bin/bash
# Train SmolVLA on merged Robomme dataset
# Usage: bash train_smolvla_robomme.sh --merged-dataset-dir <dir> --output-dir <dir> --seed <seed> --method <method> --k <k> [--gpu-id <id>] [--steps <n>] [--batch-size <n>]
#
# This script runs a single lerobot-train call on the merged 3-task Robomme dataset.

set -euo pipefail

# Default values
GPU_ID=0
STEPS=12000
BATCH_SIZE=64
NUM_WORKERS=16
LEARNING_RATE=1e-4
SAVE_FREQ=2000
EVAL_N_EPISODES=0
EVAL_BATCH_SIZE=16
ENV_EVAL_FREQ=0
WANDB_ENABLE=true
RESUME_ARGS=""

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --merged-dataset-dir)
            MERGED_DATASET_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --k)
            K="$2"
            shift 2
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --save-freq)
            SAVE_FREQ="$2"
            shift 2
            ;;
        --eval-n-episodes)
            EVAL_N_EPISODES="$2"
            shift 2
            ;;
        --eval-batch-size)
            EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --env-eval-freq)
            ENV_EVAL_FREQ="$2"
            shift 2
            ;;
        --wandb-enable)
            WANDB_ENABLE="$2"
            shift 2
            ;;
        --no-wandb)
            WANDB_ENABLE=false
            shift
            ;;
        --force)
            FORCE=1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "${MERGED_DATASET_DIR:-}" ]; then
    echo "Error: --merged-dataset-dir is required"
    exit 1
fi
if [ -z "${OUTPUT_DIR:-}" ]; then
    echo "Error: --output-dir is required"
    exit 1
fi
if [ -z "${SEED:-}" ]; then
    echo "Error: --seed is required"
    exit 1
fi
if [ -z "${METHOD:-}" ]; then
    echo "Error: --method is required"
    exit 1
fi
if [ -z "${K:-}" ]; then
    echo "Error: --k is required"
    exit 1
fi

# Resolve repo root
REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$REPO_ROOT"

echo "========================================"
echo "SmolVLA Training on Merged Robomme Dataset"
echo "========================================"
echo "Method: $METHOD"
echo "Seed: $SEED"
echo "K per task: $K"
echo "Total episodes: $((K * 3))"
echo "GPU ID: $GPU_ID"
echo "Steps: $STEPS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Merged dataset: $MERGED_DATASET_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "========================================"

# Check merged dataset exists
if [ ! -d "$MERGED_DATASET_DIR" ]; then
    echo "Error: Merged dataset directory not found: $MERGED_DATASET_DIR"
    exit 1
fi

# Check for resume
LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoints/*/ 2>/dev/null | sort -V | tail -n1 || true)
if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT/pretrained_model/train_config.json" ]; then
    if [ "${FORCE:-0}" != "1" ]; then
        echo "Found existing checkpoint: $LATEST_CKPT"
        echo "Training already completed. Use --force to retrain."
        exit 0
    else
        echo "Force retraining despite existing checkpoint"
    fi
fi

# Save config snapshot
mkdir -p "$OUTPUT_DIR"
cat > "$OUTPUT_DIR/train_config_snapshot.json" << EOF
{
    "method": "$METHOD",
    "seed": $SEED,
    "k_per_task": $K,
    "total_episodes": $((K * 3)),
    "gpu_id": $GPU_ID,
    "steps": $STEPS,
    "batch_size": $BATCH_SIZE,
    "learning_rate": "$LEARNING_RATE",
    "save_freq": $SAVE_FREQ,
    "merged_dataset_dir": "$MERGED_DATASET_DIR",
    "output_dir": "$OUTPUT_DIR",
    "policy_type": "smolvla",
    "policy_path": "lerobot/smolvla_base",
    "vlm_model_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
}
EOF

# Set environment variables
export CUDA_VISIBLE_DEVICES=$GPU_ID
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# Run training
echo ""
echo "Starting lerobot-train..."

lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id=local/robomme_merged \
    --dataset.root="$MERGED_DATASET_DIR" \
    --dataset.eval_split=0.0 \
    --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.wrist_image":"observation.images.camera2"}' \
    --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.train_state_proj=false \
    --policy.optimizer_lr=$LEARNING_RATE \
    --save_freq=$SAVE_FREQ \
    --steps=$STEPS \
    --batch_size=$BATCH_SIZE \
    --num_workers=$NUM_WORKERS \
    --seed=$SEED \
    --job_name=smolvla_robomme_${METHOD}_k${K}_seed${SEED} \
    --output_dir="$OUTPUT_DIR" \
    --wandb.enable=$WANDB_ENABLE \
    $RESUME_ARGS

echo ""
echo "Training complete!"
echo "Checkpoint directory: $OUTPUT_DIR/checkpoints/"

# Find final checkpoint
FINAL_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoints/*/ 2>/dev/null | sort -V | tail -n1 || true)
if [ -n "$FINAL_CKPT" ]; then
    echo "Final checkpoint: $FINAL_CKPT"
    echo "$FINAL_CKPT" > "$OUTPUT_DIR/final_checkpoint.txt"
else
    echo "Warning: No checkpoint found after training"
    exit 1
fi