#!/bin/bash
# Launch our_v6 (AdaptiveGrid + V5 Action Descriptors) multi-dataset experiment in tmux
# Combines V4's from-scratch acquisition with V5's action descriptor approach
# Usage: bash launch_ours_v6.sh <num_episodes_per_dataset> <gpu_id> [seed]
#   Example: bash launch_ours_v6.sh 112 0 42

set -e

NUM_EPISODES=${1:-112}
GPU_ID=${2:-0}
SEED=${3:-42}

# Three Meta-World datasets for joint training (using cached datasets)
DATASET_NAMES=("coffee-button-v3_corner" "disassemble-v3_corner" "pick_place-v3_corner")
DATASET_DIRS=(
    "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/coffee-button-v3_corner"
    "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/disassemble-v3_corner"
    "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place-v3_corner"
)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/our_v6_${NUM_EPISODES}x3_seed${SEED}"
LOG_DIR="$OUTPUT_BASE_DIR/logs"
EXP_NAME="our_v6_${NUM_EPISODES}x3_seed${SEED}"
TMUX_SESSION="our_v6_${NUM_EPISODES}x3_s${SEED}"

mkdir -p "$LOG_DIR"

# Kill existing session if exists
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

# Create runner script
RUNNER_SCRIPT="$LOG_DIR/${EXP_NAME}_runner.sh"
cat > "$RUNNER_SCRIPT" << 'RUNNER_EOF'
#!/bin/bash

# Initialize conda
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate lb_server

# Force unbuffered Python output for real-time logging
export PYTHONUNBUFFERED=1

# Change to lerobot root directory for module imports
cd /data/zhonglinye/jun/lerobot

EXP_NAME="$1"
GPU_ID="$2"
NUM_EPISODES="$3"
SEED="$4"
OUTPUT_BASE_DIR="$5"
LOG_DIR="$6"
OUR_V6_DIR="/data/zhonglinye/jun/lerobot/personal/work2/our_v6"

# Dataset configuration
DATASET_NAMES=("$7" "$8" "$9")
DATASET_DIRS=("${10}" "${11}" "${12}")

TIME_FILE="$LOG_DIR/$EXP_NAME.time"
PID_FILE="$LOG_DIR/$EXP_NAME.pid"

mkdir -p "$LOG_DIR"

# Redirect all stdout and stderr to log file from this point forward
LOG_FILE="$LOG_DIR/$EXP_NAME.log"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo $$ > "$PID_FILE"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" > "$TIME_FILE"
echo "PID: $$" >> "$TIME_FILE"

echo "========================================"
echo "Experiment: $EXP_NAME"
echo "GPU: $GPU_ID"
echo "Num Episodes per dataset: $NUM_EPISODES"
echo "Seed: $SEED"
echo "Datasets: ${DATASET_NAMES[*]}"
echo "PID: $$"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

export CUDA_VISIBLE_DEVICES=$GPU_ID
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=$GPU_ID

# Fix torchcodec/FFmpeg library loading issues
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# ============================================================================
# Step 1: Select episodes for each dataset using V6
# ============================================================================
echo ""
echo "============================================================================"
echo "Step 1: Select episodes using V6 (AdaptiveGrid + V5 Action)"
echo "============================================================================"

ALL_EPISODES_JSON="["
FIRST_DATASET=true

for i in 0 1 2; do
    DATASET_NAME="${DATASET_NAMES[$i]}"
    DATASET_DIR="${DATASET_DIRS[$i]}"
    
    echo ""
    echo "------------------------------------------------------------------------"
    echo "Selecting episodes for: ${DATASET_NAME}"
    echo "------------------------------------------------------------------------"
    
    DATASET_OUTPUT_DIR="$OUTPUT_BASE_DIR/selection/${DATASET_NAME}"
    mkdir -p "$DATASET_OUTPUT_DIR"
    
    # Use existing embedding caches (no auto-generation)
    # Auto-find the visual embedding directory
    VISUAL_EMBEDDINGS_BASE="/data/zhonglinye/jun/lerobot/personal/work2/shared_embeddings/${DATASET_NAME}"
    if [ ! -d "$VISUAL_EMBEDDINGS_BASE" ]; then
        echo "ERROR: Visual embedding base directory not found: $VISUAL_EMBEDDINGS_BASE"
        exit 1
    fi
    
    # Find the first subdirectory (should be the embedding method directory)
    VISUAL_EMBEDDINGS_DIR=$(find "$VISUAL_EMBEDDINGS_BASE" -mindepth 1 -maxdepth 1 -type d | head -1)
    if [ -z "$VISUAL_EMBEDDINGS_DIR" ]; then
        echo "ERROR: No visual embedding subdirectory found in: $VISUAL_EMBEDDINGS_BASE"
        exit 1
    fi
    
    # Auto-find the action descriptor directory
    ACTION_DESCRIPTOR_BASE="/data/zhonglinye/jun/lerobot/personal/work2/action_descriptors/${DATASET_NAME}"
    if [ ! -d "$ACTION_DESCRIPTOR_BASE" ]; then
        echo "ERROR: Action descriptor base directory not found: $ACTION_DESCRIPTOR_BASE"
        exit 1
    fi
    
    # Find the first subdirectory (should be the action descriptor directory)
    ACTION_DESCRIPTOR_DIR=$(find "$ACTION_DESCRIPTOR_BASE" -mindepth 1 -maxdepth 1 -type d | head -1)
    if [ -z "$ACTION_DESCRIPTOR_DIR" ]; then
        echo "ERROR: No action descriptor subdirectory found in: $ACTION_DESCRIPTOR_BASE"
        exit 1
    fi
    
    echo "Using existing visual embedding cache: $VISUAL_EMBEDDINGS_DIR"
    echo "Using existing action descriptor cache: $ACTION_DESCRIPTOR_DIR"
    
    # Run V6 episode selection
    python "$OUR_V6_DIR/experiments/select_episodes_v6.py" \
        --dataset_root "$DATASET_DIR" \
        --embedding_dir "$VISUAL_EMBEDDINGS_DIR" \
        --action_descriptor_dir "$ACTION_DESCRIPTOR_DIR" \
        --total_budget "$NUM_EPISODES" \
        --initial_grid_x 7 \
        --initial_grid_y 4 \
        --max_depth 3 \
        --spatial_weight 1.0 \
        --visual_weight 1.0 \
        --action_weight 0.5 \
        --seed "$SEED" \
        --output_dir "$DATASET_OUTPUT_DIR"
    
    if [ $? -ne 0 ]; then
        echo "ERROR: V6 selection failed for ${DATASET_NAME}"
        exit 1
    fi
    
    # Extract episode indices
    EPISODES=$(python -c "
import json
with open('$DATASET_OUTPUT_DIR/selected_episodes_v6.json', 'r') as f:
    data = json.load(f)
print('[' + ','.join(str(x) for x in data['selected_episode_indices']) + ']')
")
    
    echo "Selected ${NUM_EPISODES} episodes for ${DATASET_NAME}"
    
    # Build combined episodes JSON
    if [ "$FIRST_DATASET" = true ]; then
        ALL_EPISODES_JSON="${ALL_EPISODES_JSON}{\"dataset\": \"${DATASET_NAME}\", \"episodes\": ${EPISODES}}"
        FIRST_DATASET=false
    else
        ALL_EPISODES_JSON="${ALL_EPISODES_JSON}, {\"dataset\": \"${DATASET_NAME}\", \"episodes\": ${EPISODES}}"
    fi
done

ALL_EPISODES_JSON="${ALL_EPISODES_JSON}]"

# Save combined selection results
echo "$ALL_EPISODES_JSON" | python -m json.tool > "$OUTPUT_BASE_DIR/combined_selected_episodes_v6.json"

echo ""
echo "============================================================================"
echo "Episode selection complete for all datasets"
echo "============================================================================"

# ============================================================================
# Step 2: Train SmolVLA model with combined dataset
# ============================================================================
echo ""
echo "============================================================================"
echo "Step 2: Training SmolVLA model with combined dataset"
echo "============================================================================"

# Combine all episode indices for training
ALL_EPISODES_COMBINED=$(python -c "
import json
with open('$OUTPUT_BASE_DIR/combined_selected_episodes_v6.json', 'r') as f:
    data = json.load(f)
all_eps = []
for d in data:
    all_eps.extend(d['episodes'])
print('[' + ','.join(str(x) for x in all_eps) + ']')
")

echo "Total episodes for training: $(echo "$ALL_EPISODES_COMBINED" | python -c "import json,sys; print(len(json.load(sys.stdin)))")"

# Use the first dataset as the primary dataset for training
# (multi-dataset training requires specifying one repo_id/root)
PRIMARY_DATASET_DIR="${DATASET_DIRS[0]}"
PRIMARY_DATASET_NAME="${DATASET_NAMES[0]}"

# Extract task name
TASK_NAME="${PRIMARY_DATASET_NAME}"
if [[ "$TASK_NAME" != *"v3"* ]] && [[ "$TASK_NAME" != *"v2"* ]]; then
    TASK_NAME="${TASK_NAME}-v3"
fi

echo "Primary dataset: $PRIMARY_DATASET_NAME"
echo "Task name: $TASK_NAME"
echo "Episodes: $ALL_EPISODES_COMBINED"

lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/metaworld_assembly \
    --dataset.root="$PRIMARY_DATASET_DIR" \
    --dataset.episodes="$ALL_EPISODES_COMBINED" \
    --dataset.eval_split=0.0 \
    --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
    --env.type=metaworld \
    --env.task="$TASK_NAME" \
    --env.camera_name="corner,gripperPOV" \
    --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.train_state_proj=false \
    --policy.optimizer_lr=1e-4 \
    --save_freq=2000 \
    --steps=12000 \
    --batch_size=64 \
    --num_workers=16 \
    --eval.n_episodes=200 \
    --eval.batch_size=16 \
    --env_eval_freq=12000 \
    --seed="$SEED" \
    --job_name="smolvla_$EXP_NAME" \
    --output_dir="$OUTPUT_BASE_DIR/$EXP_NAME" \
    --remove_features='["observation.environment_state"]' \
    --wandb.enable=true

echo ""
echo "Training complete!"

# ============================================================================
# Step 3: Extract evaluation results
# ============================================================================
echo ""
echo "=== Extracting evaluation results ==="
EVAL_RESULTS_FILE="$OUTPUT_BASE_DIR/eval_results/$EXP_NAME\_eval.json"
mkdir -p "$OUTPUT_BASE_DIR/eval_results"

python -c "
import json
import re
import os

log_file = '$LOG_FILE'
eval_results_file = '$EVAL_RESULTS_FILE'

success_rates = []
grasp_rates = []
eval_details = []

with open(log_file, 'r') as f:
    for line in f:
        if 'Suite overall aggregated' in line:
            match = re.search(r'\{.*\}', line)
            if match:
                try:
                    data = json.loads(match.group())
                    eval_details.append(data)
                    if 'pc_success' in data:
                        success_rates.append(data['pc_success'])
                    if 'pc_grasp_success' in data:
                        grasp_rates.append(data['pc_grasp_success'])
                except json.JSONDecodeError:
                    pass

result = {
    'experiment': '$EXP_NAME',
    'num_episodes_per_dataset': $NUM_EPISODES,
    'total_episodes': $NUM_EPISODES * 3,
    'seed': $SEED,
    'gpu_id': $GPU_ID,
    'datasets': ['${DATASET_NAMES[0]}', '${DATASET_NAMES[1]}', '${DATASET_NAMES[2]}'],
    'success_rates': success_rates,
    'grasp_success_rates': grasp_rates,
    'final_success_rate': success_rates[-1] if success_rates else None,
    'final_grasp_success_rate': grasp_rates[-1] if grasp_rates else None,
    'max_success_rate': max(success_rates) if success_rates else None,
    'avg_success_rate': sum(success_rates) / len(success_rates) if success_rates else None,
    'eval_details': eval_details,
    'n_eval_runs': len(success_rates)
}

os.makedirs(os.path.dirname(eval_results_file), exist_ok=True)
with open(eval_results_file, 'w') as f:
    json.dump(result, f, indent=2)

print(f'Evaluation results saved to: {eval_results_file}')
print(f'Final success rate: {result[\"final_success_rate\"]}')
print(f'Max success rate: {result[\"max_success_rate\"]}')
"

echo ""
echo "========================================"
echo "Experiment complete: $EXP_NAME"
echo "Results directory: $OUTPUT_BASE_DIR"
echo "========================================"

echo "" >> "$TIME_FILE"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" >> "$TIME_FILE"
echo "Status: completed" >> "$TIME_FILE"
RUNNER_EOF

chmod +x "$RUNNER_SCRIPT"

# Start tmux session
tmux new-session -d -s "$TMUX_SESSION"
tmux send-keys -t "$TMUX_SESSION" "bash $RUNNER_SCRIPT $EXP_NAME $GPU_ID $NUM_EPISODES $SEED $OUTPUT_BASE_DIR $LOG_DIR ${DATASET_NAMES[0]} ${DATASET_NAMES[1]} ${DATASET_NAMES[2]} ${DATASET_DIRS[0]} ${DATASET_DIRS[1]} ${DATASET_DIRS[2]}" C-m

echo "========================================"
echo "Launched tmux session: $TMUX_SESSION"
echo "Experiment: $EXP_NAME"
echo "Datasets: ${DATASET_NAMES[*]}"
echo "Episodes per dataset: $NUM_EPISODES"
echo "GPU: $GPU_ID"
echo "Seed: $SEED"
echo "Output: $OUTPUT_BASE_DIR"
echo "========================================"
echo ""
echo "To attach to the session:"
echo "  tmux attach -t $TMUX_SESSION"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_DIR/$EXP_NAME.log"