#!/bin/bash
#
# MiniVLA Experiment Launcher (tmux)
#
# Usage:
#   bash run_tmux_minivla.sh --gpu 0 --dataset disassemble-v3_corner
#   bash run_tmux_minivla.sh --gpu 1 --dataset pick_place-v3_corner
#
# Features:
# - Creates unique tmux session per experiment
# - Auto-sets CUDA_VISIBLE_DEVICES
# - Auto-creates logs directory
# - Survives SSH disconnection
# - Prints startup info (session name, log path, GPU)

set -e

# Default values
GPU_ID=0
DATASET="disassemble-v3_corner"
NUM_EPISODES=112
SEED=42
SELECTION_MODE="grid_uniform"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --num-episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --selection-mode)
            SELECTION_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_tmux_minivla.sh --gpu <id> --dataset <name> [--num-episodes <n>] [--seed <s>] [--selection-mode <mode>]"
            exit 1
            ;;
    esac
done

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

# Create unique session name
SESSION_NAME="minivla_${DATASET}_gpu${GPU_ID}"

# Create log directory
LOG_DIR="$PROJECT_ROOT/personal/work2/differentvlm/minivla/logs/${DATASET}_gpu${GPU_ID}"
mkdir -p "$LOG_DIR"

# Log file path
LOG_FILE="$LOG_DIR/experiment.log"

# Print startup info
echo "=============================================="
echo "  MiniVLA Experiment Launcher"
echo "=============================================="
echo "  GPU:            $GPU_ID"
echo "  Dataset:        $DATASET"
echo "  Num Episodes:   $NUM_EPISODES"
echo "  Seed:           $SEED"
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
    "export CUDA_VISIBLE_DEVICES=$GPU_ID && \
     cd $PROJECT_ROOT && \
     python personal/work2/differentvlm/minivla/scripts/run_minivla.py \
         --gpu $GPU_ID \
         --dataset $DATASET \
         --num-episodes $NUM_EPISODES \
         --seed $SEED \
         --selection-mode $SELECTION_MODE \
         2>&1 | tee -a $LOG_FILE"

echo "Experiment started successfully!"
echo "Monitor with: tmux attach -t $SESSION_NAME"
echo "Or watch log: tail -f $LOG_FILE"