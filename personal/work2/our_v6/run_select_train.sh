#!/bin/bash
# Our-V6 end-to-end pipeline: causal selection on three Meta-World datasets,
# merge selected episodes, then train SmolVLA on the merged dataset.
#
# Usage:
#   bash personal/work2/our_v6/run_select_train.sh [episodes_per_dataset] [gpu_id] [seed] [visual_variant] [mode] [ablation]
# Example:
#   bash personal/work2/our_v6/run_select_train.sh 112 0 42 l2 full full
#   bash personal/work2/our_v6/run_select_train.sh 112 0 42 online_pca selection-only full
#   bash personal/work2/our_v6/run_select_train.sh 112 0 42 l2 full wo_action

set -euo pipefail

NUM_EPISODES=${1:-112}
GPU_ID=${2:-0}
SEED=${3:-42}
VISUAL_VARIANT=${4:-l2}
MODE=${5:-full}
ABLATION=${6:-full}

if [[ "$VISUAL_VARIANT" != "l2" && "$VISUAL_VARIANT" != "online_pca" ]]; then
    echo "ERROR: visual_variant must be l2 or online_pca"
    exit 2
fi
if [[ "$MODE" != "full" && "$MODE" != "selection-only" ]]; then
    echo "ERROR: mode must be full or selection-only"
    exit 2
fi
if [[ "$ABLATION" != "full" && "$ABLATION" != "wo_action" && "$ABLATION" != "wo_adaptive_priority" ]]; then
    echo "ERROR: ablation must be full, wo_action, or wo_adaptive_priority"
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WORK2_ROOT="$REPO_ROOT/personal/work2"
DATASET_BASE="$WORK2_ROOT/dataset_view"
MERGE_SCRIPT="$WORK2_ROOT/duibi/train_and_eval_scripts/merge_selected_episodes.py"
DATASETS=("coffee-button-v3_corner" "disassemble-v3_corner" "pick_place-v3_corner")
TOTAL_EPISODES=$((NUM_EPISODES * ${#DATASETS[@]}))
EXP_TAG="our_v6_${VISUAL_VARIANT}_${ABLATION}_${NUM_EPISODES}x3_seed${SEED}"
OUTPUT_BASE="$WORK2_ROOT/our_v6/outputs/$EXP_TAG"
SELECTION_DIR="$OUTPUT_BASE/selection"
SUBSET_DIR="$OUTPUT_BASE/subsets"
MERGED_DATASET_DIR="$OUTPUT_BASE/merged_dataset"
TRAIN_OUTPUT="$OUTPUT_BASE/train"
LOG_DIR="$OUTPUT_BASE/logs"
LOG_FILE="$LOG_DIR/pipeline.log"

mkdir -p "$SELECTION_DIR" "$SUBSET_DIR" "$LOG_DIR"
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
echo "Our-V6 end-to-end pipeline"
echo "episodes/dataset: $NUM_EPISODES"
echo "total selected:   $TOTAL_EPISODES"
echo "gpu:              $GPU_ID"
echo "seed:             $SEED"
echo "visual variant:   $VISUAL_VARIANT"
echo "ablation:         $ABLATION"
echo "mode:             $MODE"
echo "output:           $OUTPUT_BASE"
echo "============================================================"

for DATASET_NAME in "${DATASETS[@]}"; do
    DATASET_ROOT="$DATASET_BASE/$DATASET_NAME"
    OUT="$SELECTION_DIR/$DATASET_NAME"
    if [[ ! -d "$DATASET_ROOT" ]]; then
        echo "ERROR: dataset not found: $DATASET_ROOT"
        exit 1
    fi
    if [[ ! -f "$DATASET_ROOT/episode_initial_states.json" ]]; then
        echo "ERROR: rand_vec metadata not found: $DATASET_ROOT/episode_initial_states.json"
        exit 1
    fi

    echo ""
    echo "=== Selecting $DATASET_NAME ==="
    python "$WORK2_ROOT/our_v6/experiments/select_episodes_v6.py" \
        --dataset-root "$DATASET_ROOT" \
        --dataset-name "$DATASET_NAME" \
        --output-dir "$OUT" \
        --total-budget "$NUM_EPISODES" \
        --visual-variant "$VISUAL_VARIANT" \
        --device cuda \
        --seed "$SEED" \
        --ablation "$ABLATION"

done

MERGED_SUBSET_FILE="$SUBSET_DIR/${EXP_TAG}.json"
export SELECTION_DIR MERGED_SUBSET_FILE NUM_EPISODES SEED VISUAL_VARIANT ABLATION
python <<'PYEOF'
import json
import os
from pathlib import Path

datasets = ["coffee-button-v3_corner", "disassemble-v3_corner", "pick_place-v3_corner"]
selection_dir = Path(os.environ["SELECTION_DIR"])
all_indices = []
per_dataset = {}
for ds in datasets:
    path = selection_dir / ds / "selected_episodes_v6.json"
    with path.open("r") as f:
        result = json.load(f)
    indices = result["selected_episode_indices"]
    expected = int(os.environ["NUM_EPISODES"])
    if len(indices) != expected:
        raise RuntimeError(f"{ds}: selected {len(indices)} != requested {expected}")
    if len(indices) != len(set(indices)):
        raise RuntimeError(f"{ds}: duplicate selected episode")
    all_indices.extend(indices)
    per_dataset[ds] = indices

payload = {
    "method": "our_v6",
    "visual_variant": os.environ["VISUAL_VARIANT"],
    "ablation": os.environ["ABLATION"],
    "seed": int(os.environ["SEED"]),
    "datasets": datasets,
    "episodes_per_dataset": int(os.environ["NUM_EPISODES"]),
    "selected_episode_indices": all_indices,
    "selected_by_dataset": per_dataset,
}
out = Path(os.environ["MERGED_SUBSET_FILE"])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    json.dump(payload, f, indent=2)
print(f"Merged selection manifest: {out}")
print(f"Total selected episodes: {len(all_indices)}")
PYEOF

if [[ "$MODE" == "selection-only" ]]; then
    echo "Selection-only complete: $MERGED_SUBSET_FILE"
    exit 0
fi

echo ""
echo "=== Merging selected episodes into one LeRobot dataset ==="
rm -rf "$MERGED_DATASET_DIR"
python "$MERGE_SCRIPT" \
    --subset-file "$MERGED_SUBSET_FILE" \
    --output-dir "$MERGED_DATASET_DIR" \
    --dataset-base-dir "$DATASET_BASE"

if [[ ! -d "$MERGED_DATASET_DIR" ]]; then
    echo "ERROR: merged dataset was not created: $MERGED_DATASET_DIR"
    exit 1
fi

echo ""
echo "=== Training SmolVLA on merged Our-V6 dataset ==="
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/metaworld_pick_place \
    --dataset.root="$MERGED_DATASET_DIR" \
    --dataset.eval_split=0.0 \
    --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
    --env.type=metaworld \
    --env.task=disassemble-v3 \
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
    --job_name="smolvla_${EXP_TAG}" \
    --output_dir="$TRAIN_OUTPUT" \
    --remove_features='["observation.environment_state"]' \
    --wandb.enable=true

echo ""
echo "============================================================"
echo "Our-V6 pipeline complete"
echo "selection manifest: $MERGED_SUBSET_FILE"
echo "merged dataset:     $MERGED_DATASET_DIR"
echo "training output:    $TRAIN_OUTPUT"
echo "log:                $LOG_FILE"
echo "============================================================"
