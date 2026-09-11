#!/bin/bash
# Launch MiniVLA experiment in tmux
# Usage: bash launch_minivla.sh --gpu-id <id> --num-episodes <num> --dataset-name <name> --selection-mode <mode> [--seed <seed>]

set -e
set -o pipefail

# Default values
GPU_ID=""
NUM_EPISODES=""
DATASET_NAME="disassemble-v3_corner"
SEED=42
SELECTION_MODE="random"

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
        --selection-mode)
            SELECTION_MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash launch_minivla.sh --gpu-id <id> --num-episodes <num> --dataset-name <name> --selection-mode <mode> [--seed <seed>]"
            echo "Supported selection modes: random, grid_uniform, uniform, our_v5, deminf, subzerocore"
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

# Configuration
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MINIVLA_DIR="/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/minivla"
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/minivla/experiments/${SELECTION_MODE}_${NUM_EPISODES}_seed${SEED}_${DATASET_NAME}_minivla_${SELECTION_MODE}"
LOG_DIR="$OUTPUT_BASE_DIR/logs"
EXP_NAME="${SELECTION_MODE}_${NUM_EPISODES}_seed${SEED}_${DATASET_NAME}_minivla_${SELECTION_MODE}"
TMUX_SESSION="minivla_${SELECTION_MODE}_ep${NUM_EPISODES}_s${SEED}_${DATASET_NAME}"

# Auto-construct dataset root path from dataset name
DATASET_ROOT="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/${DATASET_NAME}"

mkdir -p "$LOG_DIR"

# Kill existing session if exists
tmux kill-session -t $TMUX_SESSION 2>/dev/null || true

# Create runner script
RUNNER_SCRIPT="$LOG_DIR/${EXP_NAME}_runner.sh"
cat > "$RUNNER_SCRIPT" << RUNNER_EOF
#!/bin/bash

# Initialize conda
eval "\$(conda shell.bash hook 2>/dev/null)"
conda activate lb_server

# Change to lerobot root directory for module imports
cd /data/zhonglinye/jun/lerobot

# Parse named arguments
GPU_ID=""
SEED_VAL=""
DATASET_NAME=""
NUM_EPISODES=""
SELECTION_MODE=""

while [[ \$# -gt 0 ]]; do
    case \$1 in
        --gpu-id)
            GPU_ID="\$2"
            shift 2
            ;;
        --seed)
            SEED_VAL="\$2"
            shift 2
            ;;
        --dataset-name)
            DATASET_NAME="\$2"
            shift 2
            ;;
        --num-episodes)
            NUM_EPISODES="\$2"
            shift 2
            ;;
        --selection-mode)
            SELECTION_MODE="\$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

EXP_NAME="\${SELECTION_MODE}_\${NUM_EPISODES}_seed\${SEED_VAL}_\${DATASET_NAME}_minivla_\${SELECTION_MODE}"
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/minivla/experiments/\${EXP_NAME}"
LOG_DIR="\$OUTPUT_BASE_DIR/logs"
TIME_FILE="\$LOG_DIR/\$EXP_NAME.time"
PID_FILE="\$LOG_DIR/\$EXP_NAME.pid"
DATASET_ROOT="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/\${DATASET_NAME}"

mkdir -p "\$LOG_DIR"

echo \$\$ > "\$PID_FILE"
echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" > "\$TIME_FILE"
echo "PID: \$\$" >> "\$TIME_FILE"

echo "========================================"
echo "MiniVLA Experiment: \$EXP_NAME"
echo "GPU: \$GPU_ID"
echo "Episodes: \$NUM_EPISODES"
echo "Seed: \$SEED_VAL"
echo "Selection Mode: \$SELECTION_MODE"
echo "Dataset: \$DATASET_NAME"
echo "Dataset Root: \$DATASET_ROOT"
echo "PID: \$\$"
echo "Started: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# Fix torchcodec/FFmpeg library loading issues
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
export LD_PRELOAD=\$CONDA_PREFIX/lib/libstdc++.so.6

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=\$GPU_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run MiniVLA experiment
python personal/work2/differentvlm/minivla/scripts/run_minivla.py \
    --gpu \$GPU_ID \
    --dataset \$DATASET_NAME \
    --num-episodes \$NUM_EPISODES \
    --seed \$SEED_VAL \
    --selection-mode \$SELECTION_MODE \
    --policy-type minivla_wrist \
    --train-steps 50000 \
    --train-save-freq 2000 \
    --env-eval-freq 2000 \
    --eval-n-episodes 10 \
    --final-eval-episodes 200 \
    --action-tokenizer-type extra_action_tokenizer \
    --scheduler-warmup-steps 500 \
    --projector-lr 1e-4 \
    --backbone-lr 2e-5 \
    2>&1 | tee -a "\$LOG_DIR/\$EXP_NAME.log"

echo "" >> "\$TIME_FILE"
echo "End time: \$(date '+%Y-%m-%d %H:%M:%S')" >> "\$TIME_FILE"
echo "Status: completed" >> "\$TIME_FILE"

echo ""
echo "========================================"
echo "MiniVLA Experiment completed: \$EXP_NAME"
echo "Finished: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

exec bash
RUNNER_EOF

chmod +x "$RUNNER_SCRIPT"

# Launch in tmux
tmux new-session -d -s $TMUX_SESSION "bash $RUNNER_SCRIPT --gpu-id $GPU_ID --seed $SEED --dataset-name $DATASET_NAME --num-episodes $NUM_EPISODES --selection-mode $SELECTION_MODE"

echo "Launched MiniVLA experiment: $EXP_NAME"
echo "tmux session: $TMUX_SESSION"
echo "Output dir: $OUTPUT_BASE_DIR"
echo "Dataset: $DATASET_NAME"
echo "Dataset root: $DATASET_ROOT"
echo "GPU: $GPU_ID"
echo "Episodes: $NUM_EPISODES"
echo "Selection mode: $SELECTION_MODE"
echo ""
echo "Monitor with: tmux attach -t $TMUX_SESSION"
echo "Check logs: tail -f $LOG_DIR/$EXP_NAME.log"