#!/bin/bash
# Random multi-task MetaWorld -> merge selected episodes -> jointly train SmolVLA.
#
# Usage:
#   bash personal/work2/duibi/train_and_eval_scripts/run_random_multi.sh \
#     [episodes_per_dataset] [gpu_id] [seed] [train_steps]
#
# Defaults:
#   episodes_per_dataset=25
#   gpu_id=0
#   seed=42
#   train_steps=10000
#
# Datasets:
#   coffee-button-v3_corner
#   disassemble-v3_corner
#   pick_place-v3_corner

set -euo pipefail

NUM_EPISODES_PER_DATASET=${1:-25}
GPU_ID=${2:-0}
SEED=${3:-42}
TRAIN_STEPS=${4:-10000}

if ! [[ "$NUM_EPISODES_PER_DATASET" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: episodes_per_dataset must be a positive integer"
    exit 2
fi
if ! [[ "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "ERROR: gpu_id must be a non-negative integer"
    exit 2
fi
if ! [[ "$SEED" =~ ^[0-9]+$ ]]; then
    echo "ERROR: seed must be a non-negative integer"
    exit 2
fi
if ! [[ "$TRAIN_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: train_steps must be a positive integer"
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
WORK2_ROOT="$REPO_ROOT/personal/work2"
SCRIPT_DIR="$WORK2_ROOT/duibi/train_and_eval_scripts"
DATASET_BASE="$WORK2_ROOT/dataset_view"
SELECT_SCRIPT="$SCRIPT_DIR/select_random_episodes.py"
MERGE_SCRIPT="$SCRIPT_DIR/merge_selected_episodes_safe.py"
DATASETS=("coffee-button-v3_corner" "disassemble-v3_corner" "pick_place-v3_corner")

TOTAL_EPISODES=$((NUM_EPISODES_PER_DATASET * ${#DATASETS[@]}))
EXP_TAG="random_multi_${NUM_EPISODES_PER_DATASET}x3_seed${SEED}_steps${TRAIN_STEPS}"
OUTPUT_BASE="$WORK2_ROOT/duibi/$EXP_TAG"
SUBSET_DIR="$OUTPUT_BASE/subsets"
MERGED_DATASET_DIR="$OUTPUT_BASE/merged_dataset"
TRAIN_OUTPUT="$OUTPUT_BASE/train"
LOG_DIR="$OUTPUT_BASE/logs"
LOG_FILE="$LOG_DIR/pipeline.log"
MERGED_SUBSET_FILE="$SUBSET_DIR/${EXP_TAG}.json"

mkdir -p "$SUBSET_DIR" "$LOG_DIR"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

cd "$REPO_ROOT"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate lb_server || true
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="$GPU_ID"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
        export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
fi

echo "============================================================"
echo "Random multi-task SmolVLA training"
echo "episodes/dataset:  $NUM_EPISODES_PER_DATASET"
echo "total episodes:    $TOTAL_EPISODES"
echo "gpu:               $GPU_ID"
echo "seed:              $SEED"
echo "train steps:       $TRAIN_STEPS"
echo "datasets:          ${DATASETS[*]}"
echo "output:            $OUTPUT_BASE"
echo "============================================================"

# Select the same deterministic Random baseline used by the existing
# select_random_episodes.py pipeline, once for each task.
for DATASET_NAME in "${DATASETS[@]}"; do
    DATASET_ROOT="$DATASET_BASE/$DATASET_NAME"
    TASK_SUBSET_DIR="$SUBSET_DIR/$DATASET_NAME"
    TASK_SUBSET_FILE="$TASK_SUBSET_DIR/random_${NUM_EPISODES_PER_DATASET}_seed${SEED}.json"

    if [[ ! -d "$DATASET_ROOT" ]]; then
        echo "ERROR: dataset not found: $DATASET_ROOT"
        exit 1
    fi

    if [[ -f "$TASK_SUBSET_FILE" ]]; then
        if python - "$TASK_SUBSET_FILE" "$NUM_EPISODES_PER_DATASET" "$SEED" <<'PYCHECK'
import json
import sys
path, expected_n, expected_seed = sys.argv[1:]
try:
    data = json.load(open(path, "r"))
    ids = [int(x) for x in data.get("selected_episode_indices", [])]
    ok = (
        data.get("method") == "random"
        and int(data.get("num_episodes", -1)) == int(expected_n)
        and int(data.get("seed", -1)) == int(expected_seed)
        and len(ids) == int(expected_n)
        and len(ids) == len(set(ids))
    )
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PYCHECK
        then
            echo "=== Reusing cached Random selection for $DATASET_NAME ==="
            continue
        fi
        echo "Cached subset is incompatible; regenerating: $TASK_SUBSET_FILE"
        rm -f "$TASK_SUBSET_FILE"
    fi

    echo "=== Random selecting $NUM_EPISODES_PER_DATASET episodes: $DATASET_NAME ==="
    python "$SELECT_SCRIPT" \
        --num-episodes "$NUM_EPISODES_PER_DATASET" \
        --seed "$SEED" \
        --dataset-root "$DATASET_ROOT" \
        --output-dir "$TASK_SUBSET_DIR"
done

# Build the merged manifest in dataset order. merge_selected_episodes_safe.py
# splits this flat list back into NUM_EPISODES_PER_DATASET entries per task.
export SUBSET_DIR MERGED_SUBSET_FILE NUM_EPISODES_PER_DATASET SEED
python <<'PYEOF'
import json
import os
from pathlib import Path

datasets = ["coffee-button-v3_corner", "disassemble-v3_corner", "pick_place-v3_corner"]
subset_dir = Path(os.environ["SUBSET_DIR"])
out = Path(os.environ["MERGED_SUBSET_FILE"])
n = int(os.environ["NUM_EPISODES_PER_DATASET"])
seed = int(os.environ["SEED"])

all_indices = []
selected_by_dataset = {}
for ds in datasets:
    path = subset_dir / ds / f"random_{n}_seed{seed}.json"
    data = json.loads(path.read_text())
    indices = [int(x) for x in data["selected_episode_indices"]]
    if len(indices) != n or len(indices) != len(set(indices)):
        raise RuntimeError(f"{ds}: invalid Random subset: {len(indices)} episodes")
    selected_by_dataset[ds] = indices
    all_indices.extend(indices)
    print(f"{ds}: {len(indices)} episodes")

payload = {
    "method": "random",
    "datasets": datasets,
    "episodes_per_dataset": n,
    "seed": seed,
    "selected_episode_indices": all_indices,
    "selected_by_dataset": selected_by_dataset,
    "num_selected": len(all_indices),
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, indent=2))
print(f"Merged Random manifest: {out}")
print(f"Total selected episodes: {len(all_indices)}")
PYEOF

# Merge only when needed. Keeping the merged dataset allows a failed training run
# to restart without re-splitting/re-merging all source videos.
if [[ -d "$MERGED_DATASET_DIR" ]]; then
    echo "=== Reusing existing merged dataset: $MERGED_DATASET_DIR ==="
else
    echo "=== Merging selected episodes into one LeRobot dataset ==="
    python "$MERGE_SCRIPT" \
        --subset-file "$MERGED_SUBSET_FILE" \
        --output-dir "$MERGED_DATASET_DIR" \
        --dataset-base-dir "$DATASET_BASE"
fi

if [[ ! -d "$MERGED_DATASET_DIR" ]]; then
    echo "ERROR: merged dataset was not created: $MERGED_DATASET_DIR"
    exit 1
fi

# Resume a partially completed run if a valid checkpoint is already present.
LATEST_CKPT=$(ls -d "$TRAIN_OUTPUT"/checkpoints/*/ 2>/dev/null | sort -V | tail -n1 || true)
RESUME_ARGS=()
if [[ -n "$LATEST_CKPT" && -f "$LATEST_CKPT/pretrained_model/train_config.json" ]]; then
    echo "=== Resuming from checkpoint: $LATEST_CKPT ==="
    RESUME_ARGS=(--resume=true "--config_path=$LATEST_CKPT/pretrained_model")
fi

echo "=== Training SmolVLA on $TOTAL_EPISODES Random demonstrations ==="
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/metaworld_three_tasks_random \
    --dataset.root="$MERGED_DATASET_DIR" \
    --dataset.eval_split=0.0 \
    --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
    --env.type=metaworld \
    --env.task=disassemble-v3 \
    --env.camera_name="corner,gripperPOV" \
    --env.use_self_mw=true \
    --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.train_state_proj=false \
    --policy.optimizer_lr=1e-4 \
    --save_freq=2000 \
    --steps="$TRAIN_STEPS" \
    --batch_size=64 \
    --num_workers=16 \
    --env_eval_freq=0 \
    --seed="$SEED" \
    --job_name="smolvla_${EXP_TAG}" \
    --output_dir="$TRAIN_OUTPUT" \
    --remove_features='["observation.environment_state"]' \
    --wandb.enable=true \
    "${RESUME_ARGS[@]}"

echo ""
echo "============================================================"
echo "Random multi-task training complete"
echo "manifest:        $MERGED_SUBSET_FILE"
echo "merged dataset:  $MERGED_DATASET_DIR"
echo "training output: $TRAIN_OUTPUT"
echo "log:             $LOG_FILE"
echo "============================================================"
