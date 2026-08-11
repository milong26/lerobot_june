#!/bin/bash
# Experiment: random_100_seed142
# GPU: 1
# tmux session: random_100_s142

set -e

EXP_NAME="random_100_seed142"
TMUX_SESSION="random_100_s142"
LOG_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/random/logs"
PID_FILE="$LOG_DIR/$EXP_NAME.pid"
TIME_FILE="$LOG_DIR/$EXP_NAME.time"
TRAIN_SCRIPT="/data/zhonglinye/jun/lerobot/personal/work2/duibi/random/scripts/train_random.sh"
NUM_EPISODES=100
SEED=142
GPU_ID=1

# Create log directory
mkdir -p "$LOG_DIR"

# Kill existing tmux session if exists
tmux kill-session -t $TMUX_SESSION 2>/dev/null || true

# Record start time
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" > "$TIME_FILE"

# Create a temporary runner script
RUNNER_SCRIPT="$LOG_DIR/$EXP_NAME"_runner.sh
cat > "$RUNNER_SCRIPT" << RUNNER_EOF
#!/bin/bash

# Initialize and activate conda
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate lb_server

# Record PID
echo $$ > "$PID_FILE"
echo "PID: $$" >> "$TIME_FILE"

echo "========================================"
echo "Experiment: $EXP_NAME"
echo "GPU: $GPU_ID"
echo "PID: $$"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# Run training
bash "$TRAIN_SCRIPT" $NUM_EPISODES $SEED $GPU_ID 2>&1 | tee -a "$LOG_DIR/$EXP_NAME.log"

# Record end time
echo "" >> "$TIME_FILE"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$TIME_FILE"
echo "Status: completed" >> "$TIME_FILE"

echo ""
echo "========================================"
echo "Experiment completed: $EXP_NAME"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# Keep tmux session open
exec bash
RUNNER_EOF

chmod +x "$RUNNER_SCRIPT"

# Launch in tmux
tmux new-session -d -s $TMUX_SESSION "bash $RUNNER_SCRIPT"

echo "Launched experiment: $EXP_NAME"
echo "tmux session: $TMUX_SESSION"
echo "PID file: $PID_FILE"
echo "Time file: $TIME_FILE"
echo ""
echo "Monitor with: tmux attach -t $TMUX_SESSION"
echo "Check logs: tail -f $LOG_DIR/$EXP_NAME.log"
