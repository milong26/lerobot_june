#!/bin/bash
# Launch Visual-FPS (Farthest Point Sampling) experiment in tmux
# Usage: bash launch_fps.sh --gpu-id <id> --num-episodes <num> --dataset-name <name> [--seed <seed>] [--mode <mode>]
#   mode: "full" (default) or "selection-only"

set -e

# Default values
GPU_ID=""
NUM_EPISODES=""
DATASET_NAME="pick_place-v3_corner"
SEED=42
MODE="full"

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
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash launch_fps.sh --gpu-id <id> --num-episodes <num> --dataset-name <name> [--seed <seed>] [--mode <mode>]"
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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/fps_ep${NUM_EPISODES}_seed${SEED}_${DATASET_NAME}"
LOG_DIR="$OUTPUT_BASE_DIR/logs"
EXP_NAME="fps_${NUM_EPISODES}_seed${SEED}"
TMUX_SESSION="fps_ep${NUM_EPISODES}_s${SEED}_${DATASET_NAME}"

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

# Parse named arguments
GPU_ID=""
SEED_VAL=""
DATASET_NAME=""
NUM_EPISODES=""
MODE_VAL=""

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
        --mode)
            MODE_VAL="\$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

EXP_NAME="fps_\${NUM_EPISODES}_seed\${SEED_VAL}"
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/fps_ep\${NUM_EPISODES}_seed\${SEED_VAL}_\${DATASET_NAME}"
LOG_DIR="\$OUTPUT_BASE_DIR/logs"
TIME_FILE="\$LOG_DIR/\$EXP_NAME.time"
PID_FILE="\$LOG_DIR/\$EXP_NAME.pid"
FPS_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/fps"
DATASET_ROOT="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/\${DATASET_NAME}"

mkdir -p "\$LOG_DIR"

echo \$\$ > "\$PID_FILE"
echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" > "\$TIME_FILE"
echo "PID: \$\$" >> "\$TIME_FILE"

echo "========================================"
echo "Experiment: \$EXP_NAME"
echo "GPU: \$GPU_ID"
echo "Episodes: \$NUM_EPISODES"
echo "Seed: \$SEED_VAL"
echo "Dataset: \$DATASET_NAME"
echo "Dataset Root: \$DATASET_ROOT"
echo "Mode: \$MODE_VAL"
echo "PID: \$\$"
echo "Started: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# Resolve canonical visual embedding cache path
# Note: shared_embeddings uses names like 'pick_place_corner' while dataset_view
# uses 'pick_place-v3_corner'. Strip '-v3' suffix for shared embedding lookup.
SHARED_EMBEDDING_DATASET_NAME=\$(echo "\$DATASET_NAME" | sed 's/-v3//g')
echo "=== Resolving visual embedding cache path ==="
echo "Dataset name (dataset_view): \$DATASET_NAME"
echo "Dataset name (shared_embeddings): \$SHARED_EMBEDDING_DATASET_NAME"
VISUAL_CACHE_DIR=\$(python -c "
import sys
sys.path.insert(0, '/data/zhonglinye/jun/lerobot/personal/work2')
from embedding_utils.cache import get_shared_embedding_dir
from embedding_utils.config import build_extraction_method_name, DEFAULT_PCA_DIM
pca_dim = DEFAULT_PCA_DIM
cache_dir = get_shared_embedding_dir('\$SHARED_EMBEDDING_DATASET_NAME', pca_dim)
method = build_extraction_method_name(pca_dim)
print(cache_dir)
print(method)
")
VISUAL_CACHE_PATH=\$(echo "\$VISUAL_CACHE_DIR" | head -1)
EMBEDDING_METHOD=\$(echo "\$VISUAL_CACHE_DIR" | tail -1)

echo "Visual embedding cache: \$VISUAL_CACHE_PATH"
echo "Embedding method: \$EMBEDDING_METHOD"

if [ ! -d "\$VISUAL_CACHE_PATH" ]; then
    echo "ERROR: Visual embedding cache does not exist: \$VISUAL_CACHE_PATH"
    exit 1
fi

# Step 1: Run Visual-FPS episode selection
echo ""
echo "=== Step 1: Running Visual-FPS episode selection ==="
export CUDA_VISIBLE_DEVICES=\$GPU_ID
export MUJOCO_EGL_DEVICE_ID=\$GPU_ID

python "\$FPS_DIR/select_visual_fps.py" \\
    --dataset-dir "\$DATASET_ROOT" \\
    --dataset-name "\$DATASET_NAME" \\
    --output-dir "\$OUTPUT_BASE_DIR" \\
    --num-selected "\$NUM_EPISODES" \\
    --seed "\$SEED_VAL" \\
    --pca-dim 32 2>&1 | tee -a "\$LOG_DIR/\$EXP_NAME.log"

if [ \${PIPESTATUS[0]} -ne 0 ]; then
    echo "ERROR: select_visual_fps.py failed"
    exit 1
fi

SUBSET_FILE="\$OUTPUT_BASE_DIR/subsets/fps_\${NUM_EPISODES}_seed\${SEED_VAL}.json"
if [ ! -f "\$SUBSET_FILE" ]; then
    echo "ERROR: Subset file not found at \$SUBSET_FILE"
    echo "Selection may have failed. Check logs above."
    exit 1
fi
echo "=== FPS selection complete ==="
echo "Subset file: \$SUBSET_FILE"

# Validate subset JSON
echo ""
echo "=== Validating subset JSON ==="
python -c "
import json
import sys

subset_file = '\$SUBSET_FILE'
with open(subset_file) as f:
    data = json.load(f)

selected = data['selected_episode_indices']
num_episodes = \$NUM_EPISODES

assert len(selected) == num_episodes, f'Expected {num_episodes} episodes, got {len(selected)}'
assert len(set(selected)) == num_episodes, 'Duplicate episodes found'

candidate_set = set(range(400))
for ep in selected:
    assert ep in candidate_set, f'Episode {ep} not in candidate set'

print(f'Validation passed: {len(selected)} episodes, no duplicates, all in candidate set')
print(f'Method: {data[\"method\"]}')
print(f'Selection method: {data[\"selection_method\"]}')
print(f'Visual embedding dir: {data[\"visual_embedding_dir\"]}')
print(f'Feature dim: {data[\"feature_dim\"]}')
print(f'First 10 selected episodes: {selected[:10]}')
" 2>&1 | tee -a "\$LOG_DIR/\$EXP_NAME.log"

# If mode is selection-only, exit after selection and validation
if [ "\$MODE_VAL" = "selection-only" ]; then
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

echo ""
echo "=== Step 2: Training and evaluation ==="
echo "Training with \$NUM_EPISODES FPS episodes (seed=\$SEED_VAL) on GPU \$GPU_ID"
echo "Episodes: \$EPISODES"

# Fix torchcodec/FFmpeg library loading issues
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
export LD_PRELOAD=\$CONDA_PREFIX/lib/libstdc++.so.6

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES=\$GPU_ID
export MUJOCO_EGL_DEVICE_ID=\$GPU_ID

# Set camera names based on dataset name
if [[ "\$DATASET_NAME" == *"corner"* ]]; then
    CAMERA_NAMES="corner,gripperPOV"
elif [[ "\$DATASET_NAME" == *"top"* ]]; then
    CAMERA_NAMES="top,gripperPOV"
else
    CAMERA_NAMES="corner,gripperPOV"
fi

# Extract task name from dataset name
# e.g., "pick_place_corner" -> "pick-place-v3"
# Remove common view suffixes: _corner, _top, _corner3, _gripper, etc.
DATASET_BASE=\$(echo "\$DATASET_NAME" | sed -E 's/_(corner|top|gripper|left|right|front|back|view)[0-9]*$//')
# Convert underscores to hyphens
TASK_BASE=\$(echo "\$DATASET_BASE" | tr '_' '-')
# Add -v3 if not already present
if [[ "\$TASK_BASE" != *"v3"* ]] && [[ "\$TASK_BASE" != *"v2"* ]]; then
    TASK_NAME="\${TASK_BASE}-v3"
else
    TASK_NAME="\$TASK_BASE"
fi

echo "Dataset name: \$DATASET_NAME"
echo "Extracted task name: \$TASK_NAME"
echo "Camera names: \$CAMERA_NAMES"

lerobot-train \\
    --policy.path=lerobot/smolvla_base \\
    --policy.device=cuda \\
    --policy.push_to_hub=false \\
    --dataset.repo_id=lerobot/metaworld_pick_place \\
    --dataset.root=\$DATASET_ROOT \\
    --dataset.episodes="\$EPISODES" \\
    --dataset.eval_split=0.0 \\
    --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \\
    --env.type=metaworld \\
    --env.task=\$TASK_NAME \\
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
    --eval.n_episodes=200 \\
    --eval.batch_size=16 \\
    --env_eval_freq=12000 \\
    --seed=\$SEED_VAL \\
    --job_name=smolvla_\$EXP_NAME \\
    --output_dir="\$OUTPUT_BASE_DIR/\$EXP_NAME" \\
    --remove_features='["observation.environment_state"]' \\
    --wandb.enable=true 2>&1 | tee -a "\$LOG_DIR/\$EXP_NAME.log"

echo ""
echo "Training steps: 12000"

# Extract eval results from log
echo ""
echo "=== Extracting evaluation results ==="
EVAL_RESULTS_FILE="\$OUTPUT_BASE_DIR/eval_results/\$EXP_NAME\_eval.json"
mkdir -p "\$OUTPUT_BASE_DIR/eval_results"

python -c "
import json
import re
import os

log_file = '\$LOG_DIR/\$EXP_NAME.log'
eval_results_file = '\$EVAL_RESULTS_FILE'

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
    'experiment': '\$EXP_NAME',
    'selection_method': 'visual_farthest_point_sampling',
    'dataset_name': '\$DATASET_NAME',
    'num_episodes': \$NUM_EPISODES,
    'seed': \$SEED_VAL,
    'gpu_id': \$GPU_ID,
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
if success_rates:
    print(f'Final success rate: {success_rates[-1]:.2f}%')
    print(f'Max success rate: {max(success_rates):.2f}%')
    print(f'Avg success rate: {sum(success_rates)/len(success_rates):.2f}%')
else:
    print('No eval results found in log')
" 2>&1 | tee -a "\$LOG_DIR/\$EXP_NAME.log"

echo "=== Evaluation results extraction complete ==="

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
tmux new-session -d -s $TMUX_SESSION "bash $RUNNER_SCRIPT --gpu-id $GPU_ID --seed $SEED --dataset-name $DATASET_NAME --num-episodes $NUM_EPISODES --mode $MODE"

echo "Launched experiment: $EXP_NAME"
echo "tmux session: $TMUX_SESSION"
echo "Output dir: $OUTPUT_BASE_DIR"
echo "Dataset: $DATASET_NAME"
echo "Dataset root: $DATASET_ROOT"
echo "GPU: $GPU_ID"
echo "Episodes: $NUM_EPISODES"
echo "Seed: $SEED"
echo "Mode: $MODE"
echo ""
echo "Monitor with: tmux attach -t $TMUX_SESSION"
echo "Check logs: tail -f $LOG_DIR/$EXP_NAME.log"