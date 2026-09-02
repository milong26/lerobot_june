#!/bin/bash
# Launch dynamic anchor (ours) experiment in tmux
# Usage: bash launch_ours.sh <num_episodes> <gpu_id> [seed] [dataset_name]

set -e

NUM_EPISODES=${1:-112}
GPU_ID=${2:-0}
SEED=${3:-42}
DATASET_NAME=${4:-corner3}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/ours_${NUM_EPISODES}_seed${SEED}_${DATASET_NAME}"
LOG_DIR="$OUTPUT_BASE_DIR/logs"
EXP_NAME="dynamicanchor_${NUM_EPISODES}_seed${SEED}"
TMUX_SESSION="ours_${NUM_EPISODES}_s${SEED}_${DATASET_NAME}"

mkdir -p "$LOG_DIR"

# Kill existing session if exists
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

# Create runner script
RUNNER_SCRIPT="$LOG_DIR/${EXP_NAME}_runner.sh"
cat > "$RUNNER_SCRIPT" << RUNNER_EOF
#!/bin/bash

# Initialize conda
eval "\$(conda shell.bash hook 2>/dev/null)"
conda activate lb_server

# Force unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

# Change to lerobot root directory for module imports
cd /data/zhonglinye/jun/lerobot

EXP_NAME="dynamicanchor_${NUM_EPISODES}_seed${SEED}"
GPU_ID=\$1
NUM_EPISODES=\$2
SEED=\$3
DATASET_NAME=\$4
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/ours_\${NUM_EPISODES}_seed\${SEED}_\${DATASET_NAME}"
LOG_DIR="\$OUTPUT_BASE_DIR/logs"
TIME_FILE="\$LOG_DIR/\$EXP_NAME.time"
PID_FILE="\$LOG_DIR/\$EXP_NAME.pid"
TRAIN_SCRIPT="/data/zhonglinye/jun/lerobot/personal/work2/duibi/train_and_eval_scripts/train_and_eval.sh"

mkdir -p "\$LOG_DIR"

# Initialize log file at the very beginning
LOG_FILE="\$LOG_DIR/\$EXP_NAME.log"
> "\$LOG_FILE"

# Redirect all stdout and stderr to log file from this point forward
exec > >(tee -a "\$LOG_FILE") 2>&1

echo \$\$ > "\$PID_FILE"
echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" > "\$TIME_FILE"
echo "PID: \$\$" >> "\$TIME_FILE"

echo "========================================"
echo "Experiment: \$EXP_NAME"
echo "GPU: \$GPU_ID"
echo "Num Episodes: \$NUM_EPISODES"
echo "Seed: \$SEED"
echo "Dataset: \$DATASET_NAME"
echo "PID: \$\$"
echo "Started: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# Ensure shared embedding cache before calling train_and_eval.sh
export CUDA_VISIBLE_DEVICES=\$GPU_ID

EMBEDDING_UTIL="/data/zhonglinye/jun/lerobot/personal/work2/embedding_utils/ensure_embeddings.py"
EMBEDDING_PATH_FILE="\$OUTPUT_BASE_DIR/shared_embedding_path.txt"
DATASET_DIR="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_\${DATASET_NAME}"

python "\$EMBEDDING_UTIL" \\
    --dataset-root "\$DATASET_DIR" \\
    --dataset-name "\$DATASET_NAME" \\
    --gpu-id "\$GPU_ID" \\
    --pca-dim 32 \\
    --path-file "\$EMBEDDING_PATH_FILE"

if [ \$? -ne 0 ]; then
    echo "ERROR: ensure_embeddings.py failed"
    exit 1
fi

EMBEDDINGS_DIR="\$(cat "\$EMBEDDING_PATH_FILE")"
echo "Shared embedding directory: \$EMBEDDINGS_DIR"

if [ ! -d "\$EMBEDDINGS_DIR" ]; then
    echo "ERROR: Shared embedding directory does not exist: \$EMBEDDINGS_DIR"
    exit 1
fi

bash "\$TRAIN_SCRIPT" dynamicanchor \$NUM_EPISODES \$SEED \$GPU_ID "\$OUTPUT_BASE_DIR" "\$DATASET_NAME" 2>&1 | tee -a "\$LOG_DIR/\$EXP_NAME.log"

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
tmux new-session -d -s "$TMUX_SESSION" "bash $RUNNER_SCRIPT $GPU_ID $NUM_EPISODES $SEED $DATASET_NAME"

echo "Launched experiment: $EXP_NAME"
echo "tmux session: $TMUX_SESSION"
echo "Output dir: $OUTPUT_BASE_DIR"
echo "Dataset: $DATASET_NAME"
echo ""
echo "Monitor with: tmux attach -t $TMUX_SESSION"
echo "Check logs: tail -f $LOG_DIR/$EXP_NAME.log"