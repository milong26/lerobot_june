#!/bin/bash
# Launch our_v5 ablation experiment in tmux
# Usage: bash launch_ours_v5_ablation.sh <num_episodes> <gpu_id> <seed> <dataset_name> <ablation_mode>
#   ablation_mode: "full", "wo_action", "wo_region"

set -e

NUM_EPISODES=${1:-112}
GPU_ID=${2:-0}
SEED=${3:-42}
DATASET_NAME=${4:-pick_place_corner}
ABLATION_MODE=${5:-full}

if [[ "$ABLATION_MODE" != "full" && "$ABLATION_MODE" != "wo_action" && "$ABLATION_MODE" != "wo_region" ]]; then
    echo "Error: ablation_mode must be one of: full, wo_action, wo_region"
    echo "Usage: bash launch_ours_v5_ablation.sh <num_episodes> <gpu_id> <seed> <dataset_name> <ablation_mode>"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/ablation/our_v5_${ABLATION_MODE}_${NUM_EPISODES}_seed${SEED}_${DATASET_NAME}"
LOG_DIR="$OUTPUT_BASE_DIR/logs"
EXP_NAME="our_v5_${ABLATION_MODE}_${NUM_EPISODES}_seed${SEED}"
TMUX_SESSION="our_v5_${ABLATION_MODE}_${NUM_EPISODES}_s${SEED}_${DATASET_NAME}"

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

EXP_NAME="our_v5_${ABLATION_MODE}_${NUM_EPISODES}_seed${SEED}"
GPU_ID=\$1
NUM_EPISODES=\$2
SEED=\$3
DATASET_NAME=\$4
ABLATION_MODE=\$5
OUTPUT_BASE_DIR="/data/zhonglinye/jun/lerobot/personal/work2/duibi/ablation/our_v5_\${ABLATION_MODE}_\${NUM_EPISODES}_seed\${SEED}_\${DATASET_NAME}"
LOG_DIR="\$OUTPUT_BASE_DIR/logs"
TIME_FILE="\$LOG_DIR/\$EXP_NAME.time"
PID_FILE="\$LOG_DIR/\$EXP_NAME.pid"
ABLATION_SELECT_DIR="/data/zhonglinye/jun/lerobot/personal/work2/our_v5_ablation"

mkdir -p "\$LOG_DIR"

# Define log file path (MUST be before exec redirect)
LOG_FILE="\$LOG_DIR/\$EXP_NAME.log"
touch "\$LOG_FILE"

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
echo "Ablation Mode: \$ABLATION_MODE"
echo "PID: \$\$"
echo "Started: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

DATASET_DIR="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/\${DATASET_NAME}"
RESULTS_DIR="\$OUTPUT_BASE_DIR/results"
SUBSET_DIR="\$OUTPUT_BASE_DIR/subsets"
TRAIN_OUTPUT_DIR="\$OUTPUT_BASE_DIR/\$EXP_NAME"
CHECKPOINTS_DIR="\$TRAIN_OUTPUT_DIR/checkpoints"

mkdir -p "\$RESULTS_DIR" "\$SUBSET_DIR"

# Fix torchcodec/FFmpeg library loading issues
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
export LD_PRELOAD=\$CONDA_PREFIX/lib/libstdc++.so.6

export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=\$GPU_ID
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

# Extract task name from dataset name
DATASET_BASE=\$(echo "\$DATASET_NAME" | sed -E 's/_(corner|top|gripper|left|right|front|back|view)[0-9]*$//')
TASK_BASE=\$(echo "\$DATASET_BASE" | tr '_' '-')
if [[ "\$TASK_BASE" != *"v3"* ]] && [[ "\$TASK_BASE" != *"v2"* ]]; then
    TASK_NAME="\${TASK_BASE}-v3"
else
    TASK_NAME="\$TASK_BASE"
fi

echo "Dataset name: \$DATASET_NAME"
echo "Extracted task name: \$TASK_NAME"
echo "Camera names: \$CAMERA_NAMES"

# ---- Step 0: Check for existing checkpoint and resume if found ----
LATEST_CHECKPOINT=""
RESUME_FLAG=false
CONFIG_PATH=""

if [ -d "\$CHECKPOINTS_DIR" ]; then
    # Find the latest checkpoint (highest step number, excluding 'last' symlink)
    LATEST_CHECKPOINT=\$(find "\$CHECKPOINTS_DIR" -maxdepth 1 -type d -name "[0-9]*" | sort -V | tail -1)
fi

if [ -n "\$LATEST_CHECKPOINT" ] && [ -f "\$LATEST_CHECKPOINT/pretrained_model/train_config.json" ]; then
    echo "=== Step 0: Found existing checkpoint at \$LATEST_CHECKPOINT ==="
    echo "Will resume training from this checkpoint."
    RESUME_FLAG=true
    CONFIG_PATH="\$LATEST_CHECKPOINT/pretrained_model/train_config.json"

    # Extract episode indices from the checkpoint's train_config.json
    echo "Extracting episode indices from checkpoint config..."
    EPISODES=\$(python -c "
import json
with open('\$CONFIG_PATH', 'r') as f:
    cfg = json.load(f)
# Try to find episodes in dataset config
dataset = cfg.get('dataset', {})
episodes = dataset.get('episodes', '[]')
print(episodes if isinstance(episodes, str) else json.dumps(episodes))
")
    echo "Episodes from checkpoint config: \$EPISODES"
else
    echo "=== Step 0: No existing checkpoint found, will start fresh training ==="

    # Step 1: Ensure visual embeddings and action descriptors exist
    echo ""
    echo "=== Step 1: Ensuring visual embeddings and action descriptors ==="
    export CUDA_VISIBLE_DEVICES=\$GPU_ID

    EMBEDDING_UTIL="/data/zhonglinye/jun/lerobot/personal/work2/embedding_utils/ensure_embeddings_v5.py"
    EMBEDDING_PATH_FILE="\$OUTPUT_BASE_DIR/v5_embedding_paths.json"

    python "\$EMBEDDING_UTIL" \\
        --dataset-root "\$DATASET_DIR" \\
        --dataset-name "\$DATASET_NAME" \\
        --gpu-id "\$GPU_ID" \\
        --pca-dim 32 \\
        --path-file "\$EMBEDDING_PATH_FILE"

    if [ \$? -ne 0 ]; then
        echo "ERROR: ensure_embeddings_v5.py failed"
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

    if [ "\$ABLATION_MODE" != "wo_action" ] && [ ! -d "\$ACTION_DESCRIPTOR_DIR" ]; then
        echo "ERROR: Action descriptor directory does not exist: \$ACTION_DESCRIPTOR_DIR"
        exit 1
    fi

    # Step 2: Run V5 ablation episode selection (skip if subset file already exists)
    SUBSET_FILE="\$SUBSET_DIR/our_v5_\${ABLATION_MODE}_\${NUM_EPISODES}_seed\${SEED}.json"
    if [ -f "\$SUBSET_FILE" ]; then
        echo ""
        echo "=== Step 2: Subset file already exists, skipping selection ==="
        echo "Using existing subset file: \$SUBSET_FILE"
    else
        echo ""
        echo "=== Step 2: Running V5 ablation episode selection (mode=\$ABLATION_MODE) ==="
        python "\$ABLATION_SELECT_DIR/select_our_v5_ablation.py" \\
            --visual-embedding-dir "\$VISUAL_EMBEDDINGS_DIR" \\
            --action-descriptor-dir "\$ACTION_DESCRIPTOR_DIR" \\
            --dataset-dir "\$DATASET_DIR" \\
            --output-dir "\$OUTPUT_BASE_DIR" \\
            --num-selected "\$NUM_EPISODES" \\
            --seed "\$SEED" \\
            --ablation-mode "\$ABLATION_MODE" \\
            --visual-weight 0.5 \\
            --action-weight 0.5

        if [ ! -f "\$SUBSET_FILE" ]; then
            echo "ERROR: Subset file not found at \$SUBSET_FILE"
            echo "Selection may have failed. Check logs above."
            exit 1
        fi
        echo "=== V5 ablation selection complete ==="
    fi
    echo "Subset file: \$SUBSET_FILE"

    # Load episode indices
    EPISODES=\$(python -c "import json; data=json.load(open('\$SUBSET_FILE')); print('[' + ','.join(str(x) for x in data['selected_episode_indices']) + ']')")
fi

echo "=== Training with \$NUM_EPISODES our_v5_\$ABLATION_MODE episodes (seed=\$SEED) on GPU \$GPU_ID ==="
echo "Episodes: \$EPISODES"

# ---- Step 3: Pre-flight eval test (only if checkpoint exists and is at eval step) ----
if [ "\$RESUME_FLAG" = true ] && [ -n "\$LATEST_CHECKPOINT" ]; then
    CHECKPOINT_STEP=\$(basename "\$LATEST_CHECKPOINT")
    echo ""
    echo "=== Step 3: Pre-flight eval test at checkpoint \$CHECKPOINT_STEP ==="
    echo "Testing if eval can run successfully before resuming training..."

    # Run a quick eval test (50 episodes) to verify configuration
    EVAL_TEST_DIR="\$OUTPUT_BASE_DIR/eval_test_\${CHECKPOINT_STEP}"
    mkdir -p "\$EVAL_TEST_DIR"

    lerobot-eval \\
        --policy.path="\$LATEST_CHECKPOINT/pretrained_model" \\
        --env.type=metaworld \\
        --env.task=\$TASK_NAME \\
        --env.camera_name="\$CAMERA_NAMES" \\
        --env.use_self_mw=true \\
        --eval.batch_size=8 \\
        --eval.n_episodes=50 \\
        --policy.device=cuda \\
        --policy.use_amp=true \\
        --rename_map='{"observation.images.camera1":"observation.images.top","observation.images.camera2":"observation.images.wrist"}' \\
        --output_dir="\$EVAL_TEST_DIR" \\
        2>&1 | tee "\$EVAL_TEST_DIR/eval_test.log"

    EVAL_TEST_EXIT_code=\${PIPESTATUS[0]}
    if [ \$EVAL_TEST_exit_code -eq 0 ]; then
        echo "=== Pre-flight eval test PASSED ==="
        echo "Eval configuration is correct. Proceeding with training resume."
    else
        echo "=== WARNING: Pre-flight eval test FAILED (exit code: \$EVAL_TEST_exit_code) ==="
        echo "Check the eval test log: \$EVAL_TEST_DIR/eval_test.log"
        echo "Training will still resume, but eval may fail again."
        echo "Waiting 10 seconds before continuing..."
        sleep 10
    fi
else
    echo ""
    echo "=== Step 3: Skipping pre-flight eval test (no checkpoint to test) ==="
fi

# ---- Step 4: Train or Resume Training ----
echo ""
if [ "\$RESUME_FLAG" = true ]; then
    echo "=== Step 4: Resuming training from checkpoint \$LATEST_CHECKPOINT ==="
    lerobot-train \\
        --config_path="\$CONFIG_PATH" \\
        --resume=true \\
        --env.use_self_mw=true \\
        --env_eval_freq=2000 \\
        --save_freq=2000 \\
        --steps=12000 \\
        --eval.n_episodes=200 \\
        --eval.batch_size=16 \\
        --output_dir="\$TRAIN_OUTPUT_DIR" \\
        --wandb.enable=true
else
    echo "=== Step 4: Starting fresh training ==="
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
        --env.task=\$TASK_NAME \\
        --env.camera_name="\$CAMERA_NAMES" \\
        --env.use_self_mw=true \\
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
        --env_eval_freq=2000 \\
        --seed=\$SEED \\
        --job_name=smolvla_our_v5_\$ABLATION_MODE \\
        --output_dir="\$TRAIN_OUTPUT_DIR" \\
        --remove_features='["observation.environment_state"]' \\
        --wandb.enable=true
fi

echo ""
echo "Training steps: 12000"

# Extract eval results from log
echo ""
echo "=== Extracting evaluation results ==="
EVAL_DIR="\$OUTPUT_BASE_DIR/\$EXP_NAME/eval"
EVAL_RESULTS_FILE="\$OUTPUT_BASE_DIR/eval_results/\$EXP_NAME\_eval.json"
mkdir -p "\$OUTPUT_BASE_DIR/eval_results"

python -c "
import json
import re
import glob
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
    'experiment': '\$EXP_NAME',
    'ablation_mode': '\$ABLATION_MODE',
    'selection_method': 'our_v5_ablation_\$ABLATION_MODE',
    'dataset_name': '\$DATASET_NAME',
    'num_episodes': \$NUM_EPISODES,
    'seed': \$SEED,
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
tmux new-session -d -s "$TMUX_SESSION" "bash $RUNNER_SCRIPT $GPU_ID $NUM_EPISODES $SEED $DATASET_NAME $ABLATION_MODE"

echo "Launched experiment: $EXP_NAME"
echo "tmux session: $TMUX_SESSION"
echo "Output dir: $OUTPUT_BASE_DIR"
echo "Dataset: $DATASET_NAME"
echo "Ablation Mode: $ABLATION_MODE"
echo ""
echo "Monitor with: tmux attach -t $TMUX_SESSION"
echo "Check logs: tail -f $LOG_DIR/$EXP_NAME.log"