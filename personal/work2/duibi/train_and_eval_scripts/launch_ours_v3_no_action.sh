#!/bin/bash
# Launch dynamic anchor v3 no-action experiment in tmux
# Usage: bash launch_ours_v3_no_action.sh <num_episodes> <gpu_id> [seed] [dataset_name] [mode]
#   mode: "full" (default) or "selection-only"

set -e

NUM_EPISODES=${1:-112}
GPU_ID=${2:-0}
SEED=${3:-42}
DATASET_NAME=${4:-corner3}
MODE=${5:-full}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/ours_v3_no_action_${NUM_EPISODES}_seed${SEED}_${DATASET_NAME}"
LOG_DIR="$OUTPUT_BASE_DIR/logs"
EXP_NAME="dynamicanchor_v3_no_action_${NUM_EPISODES}_seed${SEED}"
TMUX_SESSION="ours_v3_no_action_${NUM_EPISODES}_s${SEED}_${DATASET_NAME}"

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

# Force unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

# Change to lerobot root directory for module imports
cd /data/zhonglinye/jun/lerobot

EXP_NAME="dynamicanchor_v3_no_action_${NUM_EPISODES}_seed${SEED}"
GPU_ID=\$1
NUM_EPISODES=\$2
SEED=\$3
DATASET_NAME=\$4
MODE=\$5
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/ours_v3_no_action_\${NUM_EPISODES}_seed\${SEED}_\${DATASET_NAME}"
LOG_DIR="\$OUTPUT_BASE_DIR/logs"
TIME_FILE="\$LOG_DIR/\$EXP_NAME.time"
PID_FILE="\$LOG_DIR/\$EXP_NAME.pid"
OUR_V3_DIR="/data/zhonglinye/jun/lerobot/personal/work2/our_v3_no_action"
OUR_V2_DIR="/data/zhonglinye/jun/lerobot/personal/work2/our_v2"
OUR_DIR="/data/zhonglinye/jun/lerobot/personal/work2/our"

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
echo "Mode: \$MODE"
echo "PID: \$\$"
echo "Started: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

DATASET_DIR="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_\${DATASET_NAME}"
RESULTS_DIR="\$OUTPUT_BASE_DIR/results"
SUBSET_DIR="\$OUTPUT_BASE_DIR/subsets"

mkdir -p "\$RESULTS_DIR" "\$SUBSET_DIR"

# Step 1: Find existing visual embedding (reuse, do not re-extract)
echo "=== Step 1: Looking for existing visual embeddings ==="

V3_EMBEDDINGS_DIR="\$OUTPUT_BASE_DIR/embeddings"
V2_EMBEDDINGS_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/ours_v2_\${NUM_EPISODES}_seed\${SEED}_\${DATASET_NAME}/embeddings"
V1_EMBEDDINGS_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/ours_112_seed42_\${DATASET_NAME}/embeddings"

EMBEDDINGS_DIR=""

# Priority 1: Check v3 experiment directory
if [ -d "\$V3_EMBEDDINGS_DIR" ] && [ "\$(ls -A \$V3_EMBEDDINGS_DIR/*.npy 2>/dev/null)" ]; then
    EMBEDDINGS_DIR="\$V3_EMBEDDINGS_DIR"
    echo "Embedding source: v3 directory"
    echo "Embedding path: \$EMBEDDINGS_DIR"
# Priority 2: Check v2 experiment directory
elif [ -d "\$V2_EMBEDDINGS_DIR" ] && [ "\$(ls -A \$V2_EMBEDDINGS_DIR/*.npy 2>/dev/null)" ]; then
    EMBEDDINGS_DIR="\$V2_EMBEDDINGS_DIR"
    echo "Embedding source: v2 directory (reuse)"
    echo "Embedding path: \$EMBEDDINGS_DIR"
# Priority 3: Check v1 ours directory
elif [ -d "\$V1_EMBEDDINGS_DIR" ] && [ "\$(ls -A \$V1_EMBEDDINGS_DIR/*.npy 2>/dev/null)" ]; then
    EMBEDDINGS_DIR="\$V1_EMBEDDINGS_DIR"
    echo "Embedding source: v1 directory (reuse)"
    echo "Embedding path: \$EMBEDDINGS_DIR"
else
    echo "ERROR: No existing visual embedding found."
    echo "  v3 directory: \$V3_EMBEDDINGS_DIR"
    echo "  v2 directory: \$V2_EMBEDDINGS_DIR"
    echo "  v1 directory: \$V1_EMBEDDINGS_DIR"
    echo "Please run embedding extraction first:"
    echo "  python \$OUR_DIR/embeddings/extract_embeddings.py \\"
    echo "      --dataset-dir \$DATASET_DIR \\"
    echo "      --output-dir \$V3_EMBEDDINGS_DIR \\"
    echo "      --n-components 32 \\"
    echo "      --device cuda"
    exit 1
fi

# Copy embeddings to v3 directory for future reuse
if [ "\$EMBEDDINGS_DIR" != "\$V3_EMBEDDINGS_DIR" ]; then
    echo "Copying embeddings to v3 directory for future reuse..."
    mkdir -p "\$V3_EMBEDDINGS_DIR"
    cp "\$EMBEDDINGS_DIR"/*.npy "\$V3_EMBEDDINGS_DIR/"
    EMBEDDINGS_DIR="\$V3_EMBEDDINGS_DIR"
    echo "Embedding source: copied to v3"
    echo "Embedding path: \$EMBEDDINGS_DIR"
fi

echo "Final embedding directory: \$EMBEDDINGS_DIR"

# Step 2: Run v3 no-action episode selection
echo ""
echo "=== Step 2: Running V3 No-Action episode selection ==="
python "\$OUR_V3_DIR/experiments/select_episodes_v3_no_action.py" \\
    --dataset-root "\$DATASET_DIR" \\
    --embedding-dir "\$EMBEDDINGS_DIR" \\
    --output-dir "\$OUTPUT_BASE_DIR" \\
    --num-selected "\$NUM_EPISODES" \\
    --seed "\$SEED"

# The selection script already generates the subset file at the correct location.
# Do NOT copy it (source and destination may be the same file).
SUBSET_FILE="\$SUBSET_DIR/dynamicgrid_v3_no_action_\${NUM_EPISODES}_seed\${SEED}.json"
if [ ! -f "\$SUBSET_FILE" ]; then
    echo "ERROR: Subset file not found at \$SUBSET_FILE"
    echo "Selection may have failed. Check logs above."
    exit 1
fi
echo "=== V3 No-Action selection complete ==="
echo "Subset file: \$SUBSET_FILE"

# If mode is selection-only, exit after selection and validation
if [ "\$MODE" = "selection-only" ]; then
    echo ""
    echo "========================================"
    echo "Selection-only mode: stopping after selection"
    echo "========================================"
    echo "" >> "\$TIME_FILE"
    echo "End time: \$(date '+%Y-%m-%d %H:%M:%S')" >> "\$TIME_FILE"
    echo "Status: selection-only completed" >> "\$TIME_FILE"
    exit 0
fi

# Load episode indices
EPISODES=\$(python -c "import json; data=json.load(open('\$SUBSET_FILE')); print('[' + ','.join(str(x) for x in data['selected_episode_indices']) + ']')")

echo "=== Training with \$NUM_EPISODES v3_no_action episodes (seed=\$SEED) on GPU \$GPU_ID ==="
echo "Episodes: \$EPISODES"

# Fix torchcodec/FFmpeg library loading issues
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
export LD_PRELOAD=\$CONDA_PREFIX/lib/libstdc++.so.6

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=\$GPU_ID

# Set camera names based on dataset name
if [[ "\$DATASET_NAME" == *"corner"* ]]; then
    CAMERA_NAMES="corner,gripperPOV"
elif [[ "\$DATASET_NAME" == *"top"* ]]; then
    CAMERA_NAMES="top,gripperPOV"
else
    CAMERA_NAMES="corner,gripperPOV"
fi

echo "Camera names: \$CAMERA_NAMES"

lerobot-train \\
    --policy.path=lerobot/smolvla_base \\
    --policy.device=cuda \\
    --policy.push_to_hub=false \\
    --dataset.repo_id=lerobot/metaworld_pick_place \\
    --dataset.root=\$DATASET_DIR \\
    --dataset.episodes="\$EPISODES" \\
    --dataset.eval_split=0.0 \\
    --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \\
    --env.type=metaworld \\
    --env.task=pick-place-v3 \\
    --env.camera_name="\$CAMERA_NAMES" \\
    --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \\
    --policy.freeze_vision_encoder=true \\
    --policy.train_expert_only=true \\
    --policy.train_state_proj=false \\
    --policy.optimizer_lr=1e-4 \\
    --save_freq=2000 \\
    --steps=12000 \\
    --batch_size=64 \\
    --num_workers=16 \\
    --eval.n_episodes=16 \\
    --eval.batch_size=16 \\
    --env_eval_freq=12000 \\
    --seed=\$SEED \\
    --job_name=smolvla_\$EXP_NAME \\
    --output_dir="\$OUTPUT_BASE_DIR/\$EXP_NAME" \\
    --remove_features='["observation.environment_state"]' \\
    --wandb.enable=true

echo ""
echo "Training steps: 12000"

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
tmux new-session -d -s $TMUX_SESSION "bash $RUNNER_SCRIPT $GPU_ID $NUM_EPISODES $SEED $DATASET_NAME $MODE"

echo "Launched experiment: $EXP_NAME"
echo "tmux session: $TMUX_SESSION"
echo "Output dir: $OUTPUT_BASE_DIR"
echo "Dataset: $DATASET_NAME"
echo "Mode: $MODE"
echo ""
echo "Monitor with: tmux attach -t $TMUX_SESSION"
echo "Check logs: tail -f $LOG_DIR/$EXP_NAME.log"