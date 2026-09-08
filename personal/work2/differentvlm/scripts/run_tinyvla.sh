#!/bin/bash
#
# TinyVLA Experiment Launcher
#
# This is the entry point for running TinyVLA experiments.
#
# Usage:
#   bash run_tinyvla.sh tinyvla-s --dataset pick_place-v3_corner --gpu 0 --num_episodes 112
#   bash run_tinyvla.sh tinyvla-b --dataset pick_place-v3_corner --gpu 1 --num_episodes 112 --selection-mode grid_uniform
#   bash run_tinyvla.sh tinyvla-s --dataset pick_place-v3_corner --gpu 0 --num_episodes 112 --selection-mode random
#
# Features:
# - Creates unique tmux session per experiment
# - Auto-sets CUDA_VISIBLE_DEVICES
# - Auto-creates logs directory
# - Survives SSH disconnection
# - Prints startup info (session name, log path, GPU)
# - Supports multiple selection modes: v5, grid_uniform, random

set -e

# Default values
POLICY_TYPE=""
DATASET_NAME="pick_place-v3_corner"
GPU_ID=0
NUM_EPISODES=112
SELECTION_MODE="v5"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        tinyvla-s|tinyvla_s)
            POLICY_TYPE="tinyvla_s"
            shift
            ;;
        tinyvla-b|tinyvla_b)
            POLICY_TYPE="tinyvla_b"
            shift
            ;;
        --dataset|--dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --selection-mode)
            SELECTION_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_tinyvla.sh <tinyvla-s|tinyvla-b> --dataset <name> --gpu <id> --num_episodes <n> [--selection-mode v5|grid_uniform|random]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$POLICY_TYPE" ]; then
    echo "Error: Policy type is required"
    echo "Usage: bash run_tinyvla.sh <tinyvla-s|tinyvla-b> --dataset <name> --gpu <id> --num_episodes <n> [--selection-mode v5|grid_uniform|random]"
    echo "Available policy types: tinyvla-s, tinyvla-b"
    exit 1
fi

# Validate selection mode
if [[ "$SELECTION_MODE" != "v5" && "$SELECTION_MODE" != "grid_uniform" && "$SELECTION_MODE" != "random" ]]; then
    echo "Error: Invalid selection mode '$SELECTION_MODE'"
    echo "Available selection modes: v5, grid_uniform, random"
    exit 1
fi

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Create unique session name (include selection mode to avoid conflicts)
SESSION_NAME="tinyvla_${POLICY_TYPE}_${SELECTION_MODE}_${DATASET_NAME}"

# Create log directory (include selection mode in path for better organization)
LOG_DIR="$PROJECT_ROOT/personal/work2/differentvlm/experiments/${POLICY_TYPE}_${SELECTION_MODE}_${DATASET_NAME}/logs"
mkdir -p "$LOG_DIR"

# Log file path
LOG_FILE="$LOG_DIR/experiment.log"

# Print startup info
echo "=============================================="
echo "  TinyVLA Experiment Launcher"
echo "=============================================="
echo "  Policy Type:    $POLICY_TYPE"
echo "  Dataset:        $DATASET_NAME"
echo "  GPU:            $GPU_ID"
echo "  Num Episodes:   $NUM_EPISODES"
echo "  Selection Mode: $SELECTION_MODE"
echo "  Tmux Session:   $SESSION_NAME"
echo "  Log File:       $LOG_FILE"
echo "  Project Root:   $PROJECT_ROOT"
echo "=============================================="
echo ""
echo "Starting experiment in tmux session: $SESSION_NAME"
echo "To attach: tmux attach -t $SESSION_NAME"
echo "To detach: Ctrl+B, then D"
echo "To kill:   tmux kill-session -t $SESSION_NAME"
echo ""

# Check if session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "WARNING: Session '$SESSION_NAME' already exists!"
    echo "Kill it first: tmux kill-session -t $SESSION_NAME"
    exit 1
fi

# Create tmux session and run experiment
tmux new-session -d -s "$SESSION_NAME" \
    "export HF_HUB_OFFLINE=1 && \
     export CUDA_VISIBLE_DEVICES=$GPU_ID && \
     cd $PROJECT_ROOT && \
     python personal/work2/differentvlm/scripts/run_tinyvla.py \
     --policy_type $POLICY_TYPE \
     --dataset_name $DATASET_NAME \
     --gpu $GPU_ID \
     --num_episodes $NUM_EPISODES \
     --selection-mode $SELECTION_MODE \
     2>&1 | tee -a $LOG_FILE"

echo "Experiment started successfully."
echo "Monitor progress: tail -f $LOG_FILE"