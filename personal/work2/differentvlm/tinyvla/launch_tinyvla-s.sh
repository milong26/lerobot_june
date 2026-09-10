#!/bin/bash
# Launch TinyVLA-S training experiment in tmux
# Usage: bash launch_tinyvla-s.sh --gpu-id <id> --num-episodes <num> --dataset-name <name> --method <method> [--seed <seed>]
#
# Supported methods:
#   random       - Random episode selection
#   grid_uniform - Uniform spacing across episodes
#   first_n      - First N episodes
#   last_n       - Last N episodes

set -e

# Default values
GPU_ID=""
NUM_EPISODES=""
DATASET_NAME="disassemble-v3_corner"
SEED=42
METHOD="random"
POLICY_TYPE="tinyvla_s"

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
        --method)
            METHOD="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash launch_tinyvla-s.sh --gpu-id <id> --num-episodes <num> --dataset-name <name> --method <method> [--seed <seed>]"
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

# Directories
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/tinyvla/${POLICY_TYPE}_${METHOD}_ep${NUM_EPISODES}_seed${SEED}_${DATASET_NAME}"
LOG_DIR="$OUTPUT_BASE_DIR/logs"
SUBSETS_DIR="$OUTPUT_BASE_DIR/subsets"
EXP_NAME="${POLICY_TYPE}_${METHOD}_ep${NUM_EPISODES}_seed${SEED}"
TMUX_SESSION="${POLICY_TYPE}_${METHOD}_ep${NUM_EPISODES}_s${SEED}_${DATASET_NAME}"

# Auto-construct dataset root path from dataset name
DATASET_ROOT="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/${DATASET_NAME}"

mkdir -p "$LOG_DIR" "$SUBSETS_DIR"

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
METHOD=""
POLICY_TYPE=""

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
        --method)
            METHOD="\$2"
            shift 2
            ;;
        --policy-type)
            POLICY_TYPE="\$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

EXP_NAME="\${POLICY_TYPE}_\${METHOD}_ep\${NUM_EPISODES}_seed\${SEED_VAL}"
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/tinyvla/\${EXP_NAME}_\${DATASET_NAME}"
LOG_DIR="\$OUTPUT_BASE_DIR/logs"
SUBSETS_DIR="\$OUTPUT_BASE_DIR/subsets"
TIME_FILE="\$LOG_DIR/\$EXP_NAME.time"
PID_FILE="\$LOG_DIR/\$EXP_NAME.pid"
TRAIN_SCRIPT="/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/tinyvla/train_tinyvla.sh"
SELECT_SCRIPT="/data/zhonglinye/jun/lerobot/personal/work2/differentvlm/tinyvla/select_episodes.py"
DATASET_ROOT="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/\${DATASET_NAME}"

mkdir -p "\$LOG_DIR" "\$SUBSETS_DIR"

echo \$\$ > "\$PID_FILE"
echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" > "\$TIME_FILE"
echo "PID: \$\$" >> "\$TIME_FILE"

echo "========================================"
echo "Experiment: \$EXP_NAME"
echo "Policy: \$POLICY_TYPE"
echo "GPU: \$GPU_ID"
echo "Episodes: \$NUM_EPISODES"
echo "Seed: \$SEED_VAL"
echo "Method: \$METHOD"
echo "Dataset: \$DATASET_NAME"
echo "Dataset Root: \$DATASET_ROOT"
echo "PID: \$\$"
echo "Started: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# Step 1: Select episodes
echo "=== Step 1: Selecting episodes using \$METHOD ==="
python "\$SELECT_SCRIPT" \
    --method \$METHOD \
    --num-episodes \${NUM_EPISODES} \
    --seed \${SEED_VAL} \
    --dataset-root "\$DATASET_ROOT" \
    --output-dir "\$SUBSETS_DIR"

if [ \$? -ne 0 ]; then
    echo "ERROR: select_episodes.py failed"
    exit 1
fi

SUBSET_FILE="\$SUBSETS_DIR/selected_episodes.json"

echo ""
echo "=== Step 2: Training ==="
bash "\$TRAIN_SCRIPT" \
    "\$POLICY_TYPE" \
    "\${NUM_EPISODES}" \
    "\${SEED_VAL}" \
    "\$GPU_ID" \
    "\$OUTPUT_BASE_DIR" \
    "\$DATASET_NAME" \
    "\$METHOD" \
    "\$SUBSET_FILE" \
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

# Launch in tmux
tmux new-session -d -s $TMUX_SESSION "bash $RUNNER_SCRIPT --gpu-id $GPU_ID --seed $SEED --dataset-name $DATASET_NAME --num-episodes $NUM_EPISODES --method $METHOD --policy-type $POLICY_TYPE"

echo "Launched experiment: $EXP_NAME"
echo "tmux session: $TMUX_SESSION"
echo "Output dir: $OUTPUT_BASE_DIR"
echo "Dataset: $DATASET_NAME"
echo "Dataset root: $DATASET_ROOT"
echo "GPU: $GPU_ID"
echo "Episodes: $NUM_EPISODES"
echo "Method: $METHOD"
echo ""
echo "Monitor with: tmux attach -t $TMUX_SESSION"
echo "Check logs: tail -f $LOG_DIR/$EXP_NAME.log"