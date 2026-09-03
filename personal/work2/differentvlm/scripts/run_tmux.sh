#!/bin/bash
#
# DifferentVLM Experiment Launcher
#
# This is the ONLY entry point for running experiments.
# Do NOT run python scripts directly.
#
# Usage:
#   bash run_tmux.sh --vlm llava_pythia400m --gpu 0 --camera corner
#   bash run_tmux.sh --vlm prismatic_qwen25_05b --gpu 1 --camera corner
#
# Features:
# - Creates unique tmux session per experiment
# - Auto-sets CUDA_VISIBLE_DEVICES
# - Auto-creates logs directory
# - Survives SSH disconnection
# - Prints startup info (session name, log path, GPU)

set -e

# Default values
VLM_NAME=""
GPU_ID=0
CAMERA="corner"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vlm)
            VLM_NAME="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --camera)
            CAMERA="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash run_tmux.sh --vlm <name> --gpu <id> --camera <name>"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$VLM_NAME" ]; then
    echo "Error: --vlm is required"
    echo "Usage: bash run_tmux.sh --vlm <name> --gpu <id> --camera <name>"
    echo "Available VLMs: llava_pythia400m, prismatic_qwen25_05b"
    exit 1
fi

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

# Create unique session name
SESSION_NAME="differentvlm_${VLM_NAME}_${CAMERA}"

# Create log directory
LOG_DIR="$PROJECT_ROOT/personal/work2/differentvlm/logs/${VLM_NAME}_${CAMERA}"
mkdir -p "$LOG_DIR"

# Log file path
LOG_FILE="$LOG_DIR/experiment.log"

# Print startup info
echo "=============================================="
echo "  DifferentVLM Experiment Launcher"
echo "=============================================="
echo "  VLM:          $VLM_NAME"
echo "  GPU:          $GPU_ID"
echo "  Camera:       $CAMERA"
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

# Create tmux session and run experiment
tmux new-session -d -s "$SESSION_NAME" \
    "export CUDA_VISIBLE_DEVICES=$GPU_ID && \
     cd $PROJECT_ROOT && \
     python personal/work2/differentvlm/scripts/run_different_vlm.py \
         --vlm $VLM_NAME \
         --gpu $GPU_ID \
         --camera $CAMERA \
         2>&1 | tee $LOG_FILE"

echo "Experiment started successfully!"
echo "Monitor with: tmux attach -t $SESSION_NAME"
echo "Or watch log: tail -f $LOG_FILE"