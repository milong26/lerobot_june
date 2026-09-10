#!/bin/bash
# Launch MiniVLA random experiment in tmux
# Usage: bash launch_random.sh --gpu-id <id> --num-episodes <num> --dataset-name <name> [--seed <seed>]

set -e

# Default values
GPU_ID=""
NUM_EPISODES=""
DATASET_NAME="disassemble-v3_corner"
SEED=42

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --num-episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --dataset-name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash launch_random.sh --gpu-id <id> --num-episodes <num> --dataset-name <name> [--seed <seed>]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$GPU_ID" ]; then
    echo "Error: --gpu-id is required"
    exit 1
fi

if [ -z "$NUM_EPISODES" ]; then
    echo "Error: --num-episodes is required"
    exit 1
fi

# Call the main launch script with random selection mode
bash "$(dirname "$0")/launch_minivla.sh" \
    --gpu-id "$GPU_ID" \
    --num-episodes "$NUM_EPISODES" \
    --dataset-name "$DATASET_NAME" \
    --seed "$SEED" \
    --selection-mode "random"