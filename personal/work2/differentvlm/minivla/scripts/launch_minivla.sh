#!/bin/bash
# Launch MiniVLA experiment in tmux
# Usage: bash launch_minivla.sh <gpu_id> <dataset_name> [dataset_path]
#
# This script mirrors the duibi/launch_grid_uniform.sh flow but for MiniVLA experiments.
# It creates a tmux session that runs the MiniVLA training pipeline.

set -e

GPU_ID=${1:-0}
DATASET_NAME=${2:-disassemble-v3_corner}
DATASET_PATH=${3:-/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/${DATASET_NAME}}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"

OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/minivla/experiments/minivla_${DATASET_NAME}"
LOG_DIR="$OUTPUT_BASE_DIR/logs"

NUM_EPISODES=112
SEED=42
EXP_NAME="grid_uniform_${NUM_EPISODES}_seed${SEED}"
TMUX_SESSION="minivla_${NUM_EPISODES}_s${SEED}_${DATASET_NAME}"

mkdir -p "$LOG_DIR"

# Save dataset path config for eval scripts
echo "$DATASET_PATH" > "$OUTPUT_BASE_DIR/dataset_path.txt"
echo "Dataset path saved: $OUTPUT_BASE_DIR/dataset_path.txt"

tmux kill-session -t $TMUX_SESSION 2>/dev/null || true

RUNNER_SCRIPT="$LOG_DIR/${EXP_NAME}_runner.sh"

cat > "$RUNNER_SCRIPT" << RUNNER_EOF
#!/bin/bash

eval "\$(conda shell.bash hook 2>/dev/null)"
conda activate lb_server

export PYTHONUNBUFFERED=1

cd $PROJECT_ROOT

NUM_EPISODES=112
SEED=42
EXP_NAME="grid_uniform_\${NUM_EPISODES}_seed\${SEED}"

GPU_ID=\$1
DATASET_NAME=\$2
DATASET_PATH=\$3

OUTPUT_BASE_DIR="$OUTPUT_BASE_DIR"

LOG_DIR="\$OUTPUT_BASE_DIR/logs"

TIME_FILE="\$LOG_DIR/\$EXP_NAME.time"
PID_FILE="\$LOG_DIR/\$EXP_NAME.pid"

TRAIN_SCRIPT="$PROJECT_ROOT/personal/work2/differentvlm/minivla/scripts/run_minivla.py"

mkdir -p "\$LOG_DIR"

# Save dataset path config
echo "\$DATASET_PATH" > "\$OUTPUT_BASE_DIR/dataset_path.txt"

echo \$\$ > "\$PID_FILE"

echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" > "\$TIME_FILE"
echo "PID: \$\$" >> "\$TIME_FILE"

echo "========================================"
echo "Experiment: \$EXP_NAME"
echo "Policy: MiniVLA (from scratch, VLM backbone only)"
echo "GPU: \$GPU_ID"
echo "Dataset: \$DATASET_NAME"
echo "Dataset Path: \$DATASET_PATH"
echo "PID: \$\$"
echo "Started: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

python "\$TRAIN_SCRIPT" \
    --gpu "\$GPU_ID" \
    --dataset "\$DATASET_NAME" \
    --num-episodes "\$NUM_EPISODES" \
    --seed "\$SEED" \
    --selection-mode "grid_uniform" \
    2>&1 | tee -a "\$LOG_DIR/\$EXP_NAME.log"


echo "" >> "\$TIME_FILE"
echo "End time: \$(date '+%Y-%m-%d %H:%M:%S')" >> "\$TIME_FILE"
echo "Status: completed" >> "\$TIME_FILE"


echo ""
echo "========================================"
echo "Experiment completed: \$EXP_NAME"
echo "Finished: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

exec bash

RUNNER_EOF

chmod +x "$RUNNER_SCRIPT"


tmux new-session -d -s $TMUX_SESSION "bash $RUNNER_SCRIPT $GPU_ID $DATASET_NAME $DATASET_PATH"


echo "Launched experiment: $EXP_NAME"
echo "tmux session: $TMUX_SESSION"
echo "Output dir: $OUTPUT_BASE_DIR"
echo "Dataset: $DATASET_NAME"
echo "Dataset Path: $DATASET_PATH"
echo ""
echo "Monitor with:"
echo "tmux attach -t $TMUX_SESSION"
echo ""
echo "Check logs:"
echo "tail -f $LOG_DIR/$EXP_NAME.log"