#!/bin/bash
# Launch random experiment in tmux
# Usage: bash launch_random.sh <seed> <gpu_id>

set -e

SEED=$1
GPU_ID=${2:-0}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/random_${SEED}"
LOG_DIR="$OUTPUT_BASE_DIR/logs"
EXP_NAME="random_112_seed${SEED}"
TMUX_SESSION="random_112_s${SEED}"

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
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/random_${SEED}"
LOG_DIR="\$OUTPUT_BASE_DIR/logs"
TIME_FILE="\$LOG_DIR/\$EXP_NAME.time"
PID_FILE="\$LOG_DIR/\$EXP_NAME.pid"
TRAIN_SCRIPT="/data/zhonglinye/jun/lerobot/personal/work2/duibi/train_and_eval_scripts/train_and_eval.sh"

mkdir -p "\$LOG_DIR"

echo \$\$ > "\$PID_FILE"
echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" > "\$TIME_FILE"
echo "PID: \$\$" >> "\$TIME_FILE"

echo "========================================"
echo "Experiment: \$EXP_NAME"
echo "GPU: \$GPU_ID"
echo "PID: \$\$"
echo "Started: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

bash "\$TRAIN_SCRIPT" random 112 ${SEED} \$GPU_ID "\$OUTPUT_BASE_DIR" 2>&1 | tee -a "\$LOG_DIR/\$EXP_NAME.log"

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
tmux new-session -d -s $TMUX_SESSION "bash $RUNNER_SCRIPT $GPU_ID"

echo "Launched experiment: $EXP_NAME"
echo "tmux session: $TMUX_SESSION"
echo "Output dir: $OUTPUT_BASE_DIR"
echo ""
echo "Monitor with: tmux attach -t $TMUX_SESSION"
echo "Check logs: tail -f $LOG_DIR/$EXP_NAME.log"