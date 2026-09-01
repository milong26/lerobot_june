#!/bin/bash
# Launch dynamic anchor v2 experiment in tmux
# Usage: bash launch_ours_v2.sh <num_episodes> <gpu_id> [seed] [dataset_name]

set -e

NUM_EPISODES=${1:-112}
GPU_ID=${2:-0}
SEED=${3:-42}
DATASET_NAME=${4:-corner3}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/ours_v2_${NUM_EPISODES}_seed${SEED}_${DATASET_NAME}"
LOG_DIR="$OUTPUT_BASE_DIR/logs"
EXP_NAME="dynamicanchor_v2_${NUM_EPISODES}_seed${SEED}"
TMUX_SESSION="ours_v2_${NUM_EPISODES}_s${SEED}_${DATASET_NAME}"

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

EXP_NAME="dynamicanchor_v2_${NUM_EPISODES}_seed${SEED}"
GPU_ID=\$1
NUM_EPISODES=\$2
SEED=\$3
DATASET_NAME=\$4
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/ours_v2_\${NUM_EPISODES}_seed\${SEED}_\${DATASET_NAME}"
LOG_DIR="\$OUTPUT_BASE_DIR/logs"
TIME_FILE="\$LOG_DIR/\$EXP_NAME.time"
PID_FILE="\$LOG_DIR/\$EXP_NAME.pid"
TRAIN_SCRIPT="/data/zhonglinye/jun/lerobot/personal/work2/duibi/train_and_eval_scripts/train_and_eval.sh"
OUR_V2_DIR="/data/zhonglinye/jun/lerobot/personal/work2/our_v2"
OUR_DIR="/data/zhonglinye/jun/lerobot/personal/work2/our"

mkdir -p "\$LOG_DIR"

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

DATASET_DIR="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_\${DATASET_NAME}"
RESULTS_DIR="\$OUTPUT_BASE_DIR/results"
SUBSET_DIR="\$OUTPUT_BASE_DIR/subsets"

mkdir -p "\$RESULTS_DIR" "\$SUBSET_DIR"

# Step 1: Find existing visual embedding (reuse, do not re-extract)
echo "=== Step 1: Looking for existing visual embeddings ==="

V2_EMBEDDINGS_DIR="\$OUTPUT_BASE_DIR/embeddings"
V1_EMBEDDINGS_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/ours_112_seed42_\${DATASET_NAME}/embeddings"

EMBEDDINGS_DIR=""

# Priority 1: Check v2 experiment directory
if [ -d "\$V2_EMBEDDINGS_DIR" ] && [ "\$(ls -A \$V2_EMBEDDINGS_DIR/*.npy 2>/dev/null)" ]; then
    EMBEDDINGS_DIR="\$V2_EMBEDDINGS_DIR"
    echo "Embedding source: v2 directory"
    echo "Embedding path: \$EMBEDDINGS_DIR"
# Priority 2: Check v1 ours directory
elif [ -d "\$V1_EMBEDDINGS_DIR" ] && [ "\$(ls -A \$V1_EMBEDDINGS_DIR/*.npy 2>/dev/null)" ]; then
    EMBEDDINGS_DIR="\$V1_EMBEDDINGS_DIR"
    echo "Embedding source: v1 directory (reuse)"
    echo "Embedding path: \$EMBEDDINGS_DIR"
else
    echo "ERROR: No existing visual embedding found for v2 or v1 directory."
    echo "  v2 directory: \$V2_EMBEDDINGS_DIR"
    echo "  v1 directory: \$V1_EMBEDDINGS_DIR"
    echo "Please run embedding extraction first:"
    echo "  python \$OUR_DIR/embeddings/extract_embeddings.py \\"
    echo "      --dataset-dir \$DATASET_DIR \\"
    echo "      --output-dir \$V2_EMBEDDINGS_DIR \\"
    echo "      --n-components 32 \\"
    echo "      --device cuda"
    exit 1
fi

# Copy embeddings to v2 directory for future reuse (if using v1)
if [ "\$EMBEDDINGS_DIR" = "\$V1_EMBEDDINGS_DIR" ]; then
    echo "Copying v1 embeddings to v2 directory for future reuse..."
    mkdir -p "\$V2_EMBEDDINGS_DIR"
    cp "\$V1_EMBEDDINGS_DIR"/*.npy "\$V2_EMBEDDINGS_DIR/"
    EMBEDDINGS_DIR="\$V2_EMBEDDINGS_DIR"
    echo "Embedding source: v1 directory (copied to v2)"
    echo "Embedding path: \$EMBEDDINGS_DIR"
fi

echo "Final embedding directory: \$EMBEDDINGS_DIR"

# Step 2: Run v2 sequential greedy selection with action embedding
echo ""
echo "=== Step 2: Running Dynamic Anchor v2 episode selection ==="
python "\$OUR_V2_DIR/experiments/select_episodes_v2.py" \\
    --dataset-root "\$DATASET_DIR" \\
    --embedding-dir "\$EMBEDDINGS_DIR" \\
    --output-dir "\$RESULTS_DIR" \\
    --num-selected "\$NUM_EPISODES" \\
    --seed "\$SEED" \\
    --use-action-embedding

# Copy subset file to expected location
SUBSET_FILE="\$SUBSET_DIR/dynamicanchor_v2_\${NUM_EPISODES}_seed\${SEED}.json"
cp "\$RESULTS_DIR"/dynamicanchor_v2_*.json "\$SUBSET_FILE"
echo "=== V2 selection complete ==="

# Load episode indices
EPISODES=\$(python -c "import json; data=json.load(open('\$SUBSET_FILE')); print('[' + ','.join(str(x) for x in data['selected_episode_indices']) + ']')")

echo "=== Training with \$NUM_EPISODES v2 episodes (seed=\$SEED) on GPU \$GPU_ID ==="
echo "Episodes: \$EPISODES"

# Fix torchcodec/FFmpeg library loading issues
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
export LD_PRELOAD=\$CONDA_PREFIX/lib/libstdc++.so.6

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=\$GPU_ID

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
    --env.camera_name="corner,gripperPOV" \\
    --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \\
    --policy.freeze_vision_encoder=true \\
    --policy.train_expert_only=true \\
    --policy.train_state_proj=false \\
    --policy.optimizer_lr=1e-4 \\
    --save_freq=2000 \\
    --steps=16000 \\
    --batch_size=64 \\
    --num_workers=16 \\
    --eval.n_episodes=16 \\
    --eval.batch_size=16 \\
    --env_eval_freq=16000 \\
    --seed=\$SEED \\
    --job_name=smolvla_\$EXP_NAME \\
    --output_dir="\$OUTPUT_BASE_DIR/\$EXP_NAME" \\
    --remove_features='["observation.environment_state"]' \\
    --wandb.enable=true \\
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
tmux new-session -d -s $TMUX_SESSION "bash $RUNNER_SCRIPT $GPU_ID $NUM_EPISODES $SEED $DATASET_NAME"

echo "Launched experiment: $EXP_NAME"
echo "tmux session: $TMUX_SESSION"
echo "Output dir: $OUTPUT_BASE_DIR"
echo "Dataset: $DATASET_NAME"
echo ""
echo "Monitor with: tmux attach -t $TMUX_SESSION"
echo "Check logs: tail -f $LOG_DIR/$EXP_NAME.log"