#!/bin/bash
#
# TinyVLA Experiment Launcher
#
# This is the entry point for running TinyVLA experiments.
#
# Usage:
#   bash run_tinyvla.sh tinyvla-s --gpu 0 --camera corner --num_episodes 112
#   bash run_tinyvla.sh tinyvla-b --gpu 1 --camera corner --num_episodes 112
#
# Features:
# - Creates unique tmux session per experiment
# - Auto-sets CUDA_VISIBLE_DEVICES
# - Auto-creates logs directory
# - Survives SSH disconnection
# - Prints startup info (session name, log path, GPU)

set -e

# Default values
POLICY_TYPE=""
GPU_ID=0
CAMERA="corner"
NUM_EPISODES=112
DATASET_ROOT=""

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
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --camera)
            CAMERA="$2"
            shift 2
            ;;
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --dataset_root)
            DATASET_ROOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_tinyvla.sh <tinyvla-s|tinyvla-b> --gpu <id> --camera <name> --num_episodes <n>"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$POLICY_TYPE" ]; then
    echo "Error: Policy type is required"
    echo "Usage: bash run_tinyvla.sh <tinyvla-s|tinyvla-b> --gpu <id> --camera <name> --num_episodes <n>"
    echo "Available policy types: tinyvla-s, tinyvla-b"
    exit 1
fi

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Create unique session name
SESSION_NAME="tinyvla_${POLICY_TYPE}_${CAMERA}"

# Create log directory
LOG_DIR="$PROJECT_ROOT/personal/work2/differentvlm/logs/tinyvla_${POLICY_TYPE}_${CAMERA}"
mkdir -p "$LOG_DIR"

# Log file path
LOG_FILE="$LOG_DIR/experiment.log"

# Print startup info
echo "=============================================="
echo "  TinyVLA Experiment Launcher"
echo "=============================================="
echo "  Policy Type:  $POLICY_TYPE"
echo "  GPU:          $GPU_ID"
echo "  Camera:       $CAMERA"
echo "  Num Episodes: $NUM_EPISODES"
echo "  Tmux Session: $SESSION_NAME"
echo "  Log File:     $LOG_FILE"
echo "  Project Root: $PROJECT_ROOT"
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

# Build dataset root argument
DATASET_ARG=""
if [ -n "$DATASET_ROOT" ]; then
    DATASET_ARG="--dataset_root $DATASET_ROOT"
fi

# Create tmux session and run experiment
tmux new-session -d -s "$SESSION_NAME" \
    "export CUDA_VISIBLE_DEVICES=$GPU_ID && \
     cd $PROJECT_ROOT && \
     python personal/work2/differentvlm/scripts/run_tinyvla.py \
     --policy_type $POLICY_TYPE \
     --gpu $GPU_ID \
     --camera $CAMERA \
     --num_episodes $NUM_EPISODES \
     $DATASET_ARG \
     2>&1 | tee $LOG_FILE"

echo "Experiment started successfully."
echo "Monitor progress: tail -f $LOG_FILE"