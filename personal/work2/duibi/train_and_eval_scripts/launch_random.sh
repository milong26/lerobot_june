#!/bin/bash
# Launch random experiment in tmux
# Usage: bash launch_random.sh <seed> <gpu_id> <dataset_name>

set -e

SEED=$1
GPU_ID=${2:-0}
DATASET_NAME=${3:-disassemble-v3_corner}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/random_${SEED}_${DATASET_NAME}"
LOG_DIR="$OUTPUT_BASE_DIR/logs"
EXP_NAME="random_112_seed${SEED}"
TMUX_SESSION="random_112_s${SEED}_${DATASET_NAME}"

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

EXP_NAME="random_112_seed${SEED}"
GPU_ID=\$1
SEED_VAL=\$2
DATASET_NAME=\$3
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/random_\${SEED_VAL}_\${DATASET_NAME}"
LOG_DIR="\$OUTPUT_BASE_DIR/logs"
TIME_FILE="\$LOG_DIR/\$EXP_NAME.time"
PID_FILE="\$LOG_DIR/\$EXP_NAME.pid"
TRAIN_SCRIPT="/data/zhonglinye/jun/lerobot/personal/work2/duibi/train_and_eval_scripts/train_and_eval.sh"
SELECT_SCRIPT="/data/zhonglinye/jun/lerobot/personal/work2/duibi/train_and_eval_scripts/select_random_episodes.py"
DATASET_ROOT="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/\${DATASET_NAME}"

mkdir -p "\$LOG_DIR"

echo \$\$ > "\$PID_FILE"
echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" > "\$TIME_FILE"
echo "PID: \$\$" >> "\$TIME_FILE"

echo "========================================"
echo "Experiment: \$EXP_NAME"
echo "GPU: \$GPU_ID"
echo "Seed: \$SEED_VAL"
echo "Dataset: \$DATASET_NAME"
echo "Dataset Root: \$DATASET_ROOT"
echo "PID: \$\$"
echo "Started: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# Step 1: Select random episodes
echo "=== Step 1: Selecting random episodes ==="
python "\$SELECT_SCRIPT" \
    --num-episodes 112 \
    --seed \${SEED_VAL} \
    --dataset-root "\$DATASET_ROOT" \
    --output-dir "\$OUTPUT_BASE_DIR/subsets"

if [ \$? -ne 0 ]; then
    echo "ERROR: select_random_episodes.py failed"
    exit 1
fi

echo ""
echo "=== Step 2: Training and evaluation ==="
bash "\$TRAIN_SCRIPT" random 112 \${SEED_VAL} \$GPU_ID "" "\$DATASET_NAME" 2>&1 | tee -a "\$LOG_DIR/\$EXP_NAME.log"

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

# Launch in tmux
tmux new-session -d -s $TMUX_SESSION "bash $RUNNER_SCRIPT $GPU_ID $SEED $DATASET_NAME"

echo "Launched experiment: $EXP_NAME"
echo "tmux session: $TMUX_SESSION"
echo "Output dir: $OUTPUT_BASE_DIR"
echo "Dataset: $DATASET_NAME"
echo "Dataset root: $DATASET_ROOT"
echo ""
echo "Monitor with: tmux attach -t $TMUX_SESSION"
echo "Check logs: tail -f $LOG_DIR/$EXP_NAME.log"