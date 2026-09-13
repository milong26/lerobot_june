#!/bin/bash
# Launch our_v5 multi-dataset experiment in tmux
# Usage: bash launch_ours_v5_multi.sh <num_episodes_per_dataset> <gpu_id> [seed] [mode]
#   mode: "full" (default) or "selection-only"
#
# This script selects episodes from 3 datasets and trains on the combined set:
#   - coffee-button-v3_corner
#   - disassemble-v3_corner
#   - pick_place-v3_corner
#
# Total episodes = num_episodes_per_dataset * 3

set -e

NUM_EPISODES_PER_DATASET=${1:-28}
GPU_ID=${2:-0}
SEED=${3:-42}
MODE=${4:-full}

# Define the 3 datasets
DATASETS=("coffee-button-v3_corner" "disassemble-v3_corner" "pick_place-v3_corner")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOTAL_EPISODES=$((NUM_EPISODES_PER_DATASET * 3))
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/our_v5_multi_${TOTAL_EPISODES}_seed${SEED}"
LOG_DIR="$OUTPUT_BASE_DIR/logs"
EXP_NAME="our_v5_multi_${TOTAL_EPISODES}_seed${SEED}"
TMUX_SESSION="our_v5_multi_${TOTAL_EPISODES}_s${SEED}"

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

GPU_ID=\$1
NUM_EPISODES_PER_DATASET=\$2
SEED=\$3
MODE=\$4
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/our_v5_multi_\$((NUM_EPISODES_PER_DATASET * 3))_seed\${SEED}"
LOG_DIR="\$OUTPUT_BASE_DIR/logs"
TIME_FILE="\$LOG_DIR/our_v5_multi_\$((NUM_EPISODES_PER_DATASET * 3))_seed\${SEED}.time"
PID_FILE="\$LOG_DIR/our_v5_multi_\$((NUM_EPISODES_PER_DATASET * 3))_seed\${SEED}.pid"
OUR_V5_DIR="/data/zhonglinye/jun/lerobot/personal/work2/our_v5"
EMBEDDING_UTIL="/data/zhonglinye/jun/lerobot/personal/work2/embedding_utils/ensure_embeddings_v5.py"

# Define the 3 datasets
DATASETS=("coffee-button-v3_corner" "disassemble-v3_corner" "pick_place-v3_corner")

mkdir -p "\$LOG_DIR"

# Append to log file (don't overwrite)
LOG_FILE="\$LOG_DIR/our_v5_multi_\$((NUM_EPISODES_PER_DATASET * 3))_seed\${SEED}.log"
touch "\$LOG_FILE"

# Redirect all stdout and stderr to log file from this point forward
exec > >(tee -a "\$LOG_FILE") 2>&1

echo \$\$ > "\$PID_FILE"
echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" > "\$TIME_FILE"
echo "PID: \$\$" >> "\$TIME_FILE"

echo "========================================"
echo "Experiment: Multi-dataset our_v5"
echo "GPU: \$GPU_ID"
echo "Episodes per dataset: \$NUM_EPISODES_PER_DATASET"
echo "Total episodes: \$((NUM_EPISODES_PER_DATASET * 3))"
echo "Seed: \$SEED"
echo "Datasets: \${DATASETS[*]}"
echo "Mode: \$MODE"
echo "PID: \$\$"
echo "Started: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

SUBSET_DIR="\$OUTPUT_BASE_DIR/subsets"
RESULTS_DIR="\$OUTPUT_BASE_DIR/results"
mkdir -p "\$SUBSET_DIR" "\$RESULTS_DIR"

# Step 1: Process each dataset - ensure embeddings and select episodes
ALL_EPISODE_INDICES=()

for DATASET_NAME in "\${DATASETS[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing dataset: \$DATASET_NAME"
    echo "========================================"
    
    DATASET_DIR="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/\${DATASET_NAME}"
    
    # Step 1.1: Ensure visual embeddings and action descriptors exist
    echo "=== Step 1.1: Ensuring visual embeddings and action descriptors ==="
    export CUDA_VISIBLE_DEVICES=\$GPU_ID
    
    EMBEDDING_PATH_FILE="\$OUTPUT_BASE_DIR/v5_embedding_paths_\${DATASET_NAME}.json"
    
    python "\$EMBEDDING_UTIL" \\
        --dataset-root "\$DATASET_DIR" \\
        --dataset-name "\$DATASET_NAME" \\
        --gpu-id "\$GPU_ID" \\
        --pca-dim 32 \\
        --path-file "\$EMBEDDING_PATH_FILE"
    
    if [ \$? -ne 0 ]; then
        echo "ERROR: ensure_embeddings_v5.py failed for \$DATASET_NAME"
        exit 1
    fi
    
    VISUAL_EMBEDDINGS_DIR=\$(python -c "import json; d=json.load(open('\$EMBEDDING_PATH_FILE')); print(d['visual_embedding_dir'])")
    ACTION_DESCRIPTOR_DIR=\$(python -c "import json; d=json.load(open('\$EMBEDDING_PATH_FILE')); print(d['action_descriptor_dir'])")
    
    echo "Visual embedding directory: \$VISUAL_EMBEDDINGS_DIR"
    echo "Action descriptor directory: \$ACTION_DESCRIPTOR_DIR"
    
    if [ ! -d "\$VISUAL_EMBEDDINGS_DIR" ]; then
        echo "ERROR: Visual embedding directory does not exist: \$VISUAL_EMBEDDINGS_DIR"
        exit 1
    fi
    
    if [ ! -d "\$ACTION_DESCRIPTOR_DIR" ]; then
        echo "ERROR: Action descriptor directory does not exist: \$ACTION_DESCRIPTOR_DIR"
        exit 1
    fi
    
    # Step 1.2: Run V5 episode selection
    echo ""
    echo "=== Step 1.2: Running V5 episode selection for \$DATASET_NAME ==="
    python "\$OUR_V5_DIR/select_our_v5.py" \\
        --visual-embedding-dir "\$VISUAL_EMBEDDINGS_DIR" \\
        --action-descriptor-dir "\$ACTION_DESCRIPTOR_DIR" \\
        --dataset-dir "\$DATASET_DIR" \\
        --output-dir "\$OUTPUT_BASE_DIR" \\
        --num-selected "\$NUM_EPISODES_PER_DATASET" \\
        --seed "\$SEED" \\
        --visual-weight 0.5 \\
        --action-weight 0.5
    
    # select_our_v5.py outputs to: our_v5_{num}_seed{seed}.json (without dataset name)
    # We need to rename it to include dataset name to avoid overwriting
    GENERIC_SUBSET_FILE="\$SUBSET_DIR/our_v5_\${NUM_EPISODES_PER_DATASET}_seed\${SEED}.json"
    SUBSET_FILE="\$SUBSET_DIR/our_v5_\${NUM_EPISODES_PER_DATASET}_seed\${SEED}_\${DATASET_NAME}.json"
    
    if [ ! -f "\$GENERIC_SUBSET_FILE" ]; then
        echo "ERROR: Generic subset file not found at \$GENERIC_SUBSET_FILE for \$DATASET_NAME"
        echo "Selection may have failed. Check logs above."
        exit 1
    fi
    
    # Rename to dataset-specific file
    mv "\$GENERIC_SUBSET_FILE" "\$SUBSET_FILE"
    echo "Renamed subset file to: \$SUBSET_FILE"
    
    echo "=== Selection complete for \$DATASET_NAME ==="
    echo "Subset file: \$SUBSET_FILE"
    
    # Extract episode indices and store them
    EPISODE_INDICES=\$(python -c "import json; data=json.load(open('\$SUBSET_FILE')); print(' '.join(str(x) for x in data['selected_episode_indices']))")
    echo "Selected episodes for \$DATASET_NAME: \$EPISODE_INDICES"
    
    # Store in array
    ALL_EPISODE_INDICES+=("\$EPISODE_INDICES")
done

# Step 2: Merge all episode indices
echo ""
echo "========================================"
echo "Merging episode indices from all datasets"
echo "========================================"

# Create a merged subset file
MERGED_SUBSET_FILE="\$SUBSET_DIR/our_v5_multi_\$((NUM_EPISODES_PER_DATASET * 3))_seed\${SEED}.json"

export SUBSET_DIR="\$SUBSET_DIR"
export MERGED_SUBSET_FILE="\$MERGED_SUBSET_FILE"
export NUM_EPISODES="\$NUM_EPISODES_PER_DATASET"
export SEED="\$SEED"

python << 'PYEOF'
import json
import os

datasets = ['coffee-button-v3_corner', 'disassemble-v3_corner', 'pick_place-v3_corner']
num_episodes = int(os.environ['NUM_EPISODES'])
seed = int(os.environ['SEED'])
subset_dir = os.environ['SUBSET_DIR']
merged_file = os.environ['MERGED_SUBSET_FILE']

all_indices = []
for ds in datasets:
    subset_file = f'{subset_dir}/our_v5_{num_episodes}_seed{seed}_{ds}.json'
    with open(subset_file, 'r') as f:
        data = json.load(f)
        indices = data['selected_episode_indices']
        all_indices.extend(indices)
        print(f'{ds}: {len(indices)} episodes selected')

merged_data = {
    'selected_episode_indices': all_indices,
    'num_selected': len(all_indices),
    'datasets': datasets,
    'episodes_per_dataset': num_episodes,
    'seed': seed
}

with open(merged_file, 'w') as f:
    json.dump(merged_data, f, indent=2)

print(f'Merged subset saved to: {merged_file}')
print(f'Total episodes: {len(all_indices)}')
PYEOF

echo "=== Merged subset file: \$MERGED_SUBSET_FILE ==="

# If mode is selection-only, exit after selection
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

# Step 3: Merge datasets into a single training dataset
echo ""
echo "=== Step 3: Merging selected episodes into a single dataset ==="
MERGED_DATASET_DIR="\$OUTPUT_BASE_DIR/merged_dataset"

python /data/zhonglinye/jun/lerobot/personal/work2/duibi/train_and_eval_scripts/merge_selected_episodes.py \\
    --subset-file "\$MERGED_SUBSET_FILE" \\
    --output-dir "\$MERGED_DATASET_DIR" \\
    --dataset-base-dir "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view"

if [ ! -d "\$MERGED_DATASET_DIR" ]; then
    echo "ERROR: Merged dataset directory not created at \$MERGED_DATASET_DIR"
    exit 1
fi

echo "=== Merged dataset created at: \$MERGED_DATASET_DIR ==="

# Step 4: Training with merged dataset
echo ""
echo "=== Step 4: Training with \$((NUM_EPISODES_PER_DATASET * 3)) episodes (seed=\$SEED) on GPU \$GPU_ID ==="

# Fix torchcodec/FFmpeg library loading issues
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
export LD_PRELOAD=\$CONDA_PREFIX/lib/libstdc++.so.6

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID=\$GPU_ID
export CUDA_VISIBLE_DEVICES=\$GPU_ID

CAMERA_NAMES="corner,gripperPOV"
TASK_NAME="disassemble-v3"

echo "Merged dataset: \$MERGED_DATASET_DIR"
echo "Camera names: \$CAMERA_NAMES"
echo "Task name: \$TASK_NAME"

lerobot-train \\
    --policy.path=lerobot/smolvla_base \\
    --policy.device=cuda \\
    --policy.push_to_hub=false \\
    --dataset.repo_id=lerobot/metaworld_pick_place \\
    --dataset.root=\$MERGED_DATASET_DIR \\
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
    --seed=\$SEED \\
    --job_name=smolvla_multi_\$((NUM_EPISODES_PER_DATASET * 3)) \\
    --output_dir="\$OUTPUT_BASE_DIR/our_v5_multi_\$((NUM_EPISODES_PER_DATASET * 3))_seed\${SEED}" \\
    --remove_features='["observation.environment_state"]' \\
    --wandb.enable=true

echo ""
echo "Training steps: 12000"

# Extract eval results from log
echo ""
echo "=== Extracting evaluation results ==="
EVAL_RESULTS_FILE="\$OUTPUT_BASE_DIR/eval_results/our_v5_multi_\$((NUM_EPISODES_PER_DATASET * 3))_seed\${SEED}_eval.json"
mkdir -p "\$OUTPUT_BASE_DIR/eval_results"

python -c "
import json
import re
import os

log_file = '\$LOG_FILE'
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
    'experiment': 'our_v5_multi_\$((NUM_EPISODES_PER_DATASET * 3))_seed\${SEED}',
    'num_episodes_per_dataset': $NUM_EPISODES_PER_DATASET,
    'total_episodes': \$((NUM_EPISODES_PER_DATASET * 3)),
    'seed': \$SEED,
    'gpu_id': \$GPU_ID,
    'datasets': ['coffee-button-v3_corner', 'disassemble-v3_corner', 'pick_place-v3_corner'],
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
"

echo "=== Evaluation results extraction complete ==="

echo "" >> "\$TIME_FILE"
echo "End time: \$(date '+%Y-%m-%d %H:%M:%S')" >> "\$TIME_FILE"
echo "Status: completed" >> "\$TIME_FILE"

echo ""
echo "========================================"
echo "Experiment completed: \$EXP_NAME"
echo "Finished: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""
echo "Results:"
echo "  - Log: \$LOG_FILE"
echo "  - Eval results: \$EVAL_RESULTS_FILE"
echo "  - Output dir: \$OUTPUT_BASE_DIR"

exec bash
RUNNER_EOF

chmod +x "$RUNNER_SCRIPT"

# Launch in tmux
tmux new-session -d -s "$TMUX_SESSION" "bash $RUNNER_SCRIPT $GPU_ID $NUM_EPISODES_PER_DATASET $SEED $MODE"

echo "Launched multi-dataset experiment: $EXP_NAME"
echo "tmux session: $TMUX_SESSION"
echo "Output dir: $OUTPUT_BASE_DIR"
echo "Datasets: ${DATASETS[*]}"
echo "Episodes per dataset: $NUM_EPISODES_PER_DATASET"
echo "Total episodes: $TOTAL_EPISODES"
echo "Mode: $MODE"
echo ""
echo "Monitor with: tmux attach -t $TMUX_SESSION"
echo "Check logs: tail -f $LOG_DIR/$EXP_NAME.log"