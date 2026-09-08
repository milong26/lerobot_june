#!/bin/bash
# Launch grid_uniform MiniVLA experiment in tmux
# Usage: bash launch_grid_uniform_minivla.sh <gpu_id> <dataset_name> [dataset_path]
#
# This script mirrors the duibi/train_and_eval_scripts/launch_grid_uniform.sh flow,
# but replaces the SmolVLA policy with MiniVLA policy.
#
# Flow: dataset preparation -> episode selection -> MiniVLA config -> LeRobot train
#       -> checkpoint save -> MiniVLA eval -> success/grasp_success stats -> summary

set -e

GPU_ID=${1:-0}
DATASET_NAME=${2:-disassemble-v3_corner}
DATASET_PATH=${3:-/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/${DATASET_NAME}}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/minivla/grid_uniform_42_${DATASET_NAME}"
LOG_DIR="$OUTPUT_BASE_DIR/logs"

NUM_EPISODES=112
EXP_NAME="grid_uniform_${NUM_EPISODES}_seed42"
TMUX_SESSION="minivla_grid_uniform_${NUM_EPISODES}_s42_${DATASET_NAME}"

mkdir -p "$LOG_DIR"

# Save dataset path config for run_minivla.py
echo "$DATASET_PATH" > "$OUTPUT_BASE_DIR/dataset_path.txt"
echo "Dataset path saved: $DATASET_PATH"

tmux kill-session -t $TMUX_SESSION 2>/dev/null || true

RUNNER_SCRIPT="$LOG_DIR/${EXP_NAME}_runner.sh"

cat > "$RUNNER_SCRIPT" << RUNNER_EOF
#!/bin/bash

eval "\$(conda shell.bash hook 2>/dev/null)"
conda activate lb_server

export PYTHONUNBUFFERED=1

cd /data/zhonglinye/jun/lerobot

NUM_EPISODES=112
EXP_NAME="grid_uniform_\${NUM_EPISODES}_seed42"

GPU_ID=\$1
DATASET_NAME=\$2
DATASET_PATH=\$3

OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/minivla/grid_uniform_42_\${DATASET_NAME}"

LOG_DIR="\$OUTPUT_BASE_DIR/logs"

TIME_FILE="\$LOG_DIR/\$EXP_NAME.time"
PID_FILE="\$LOG_DIR/\$EXP_NAME.pid"

mkdir -p "\$LOG_DIR"

# Save dataset path config
echo "\$DATASET_PATH" > "\$OUTPUT_BASE_DIR/dataset_path.txt"

echo \$\$ > "\$PID_FILE"

echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" > "\$TIME_FILE"
echo "PID: \$\$" >> "\$TIME_FILE"

echo "MiniVLA Experiment: \$EXP_NAME"
echo "Sampling: factorized obj-goal joint-region uniform"
echo "GPU: \$GPU_ID"
echo "Dataset: \$DATASET_NAME"
echo "Dataset Path: \$DATASET_PATH"
echo "PID: \$\$"
echo "Started: \$(date '+%Y-%m-%d %H:%M:%S')"
echo ""

python personal/work2/differentvlm/minivla/scripts/run_minivla.py \
    --gpu \$GPU_ID \
    --dataset \$DATASET_NAME \
    --num-episodes \$NUM_EPISODES \
    --seed 42 \
    --selection-mode grid_uniform \
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


tmux new-session -d -s $TMUX_SESSION "bash $RUNNER_SCRIPT $GPU_ID $DATASET_NAME $DATASET_PATH"


echo "Launched MiniVLA experiment: $EXP_NAME"
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