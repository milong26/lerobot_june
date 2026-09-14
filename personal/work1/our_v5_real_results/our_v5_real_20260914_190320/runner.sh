#!/usr/bin/env bash
set -euo pipefail

# ---- Arguments passed from launcher ----
NUM_EPISODES_PER_DATASET="$1"
GPU_ID="$2"
SEED="$3"
MODE="$4"
DATASET_ROOT_CSV="$5"
DATASET_NAMES_CSV="$6"
OUTPUT_DIR="$7"
LOG_DIR="$8"

LOG_FILE="${LOG_DIR}/runner.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================"
echo "Runner started at $(date)"
echo "============================================"

# ---- Setup environment ----
eval "$(conda shell.bash hook)"
conda activate lb_server
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
cd /data/zhonglinye/jun/lerobot

echo "Python: $(which python3)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# ---- Parse dataset lists ----
IFS=',' read -ra DATASET_ROOTS <<< "$DATASET_ROOT_CSV"
IFS=',' read -ra DATASET_NAMES <<< "$DATASET_NAMES_CSV"

if [ "${#DATASET_ROOTS[@]}" -ne "${#DATASET_NAMES[@]}" ]; then
    echo "ERROR: Number of dataset roots (${#DATASET_ROOTS[@]}) does not match number of dataset names (${#DATASET_NAMES[@]})"
    exit 1
fi

NUM_DATASETS="${#DATASET_ROOTS[@]}"
echo "Datasets: $NUM_DATASETS"
for i in "${!DATASET_ROOTS[@]}"; do
    echo "  [$i] ${DATASET_NAMES[$i]}: ${DATASET_ROOTS[$i]}"
done
echo ""

# ---- Phase 1: Selection ----
echo "============================================"
echo "Phase 1: Episode Selection"
echo "============================================"

SELECTOR="personal/work1/select_our_v5_real.py"

for i in "${!DATASET_ROOTS[@]}"; do
    DROOT="${DATASET_ROOTS[$i]}"
    DNAME="${DATASET_NAMES[$i]}"
    echo ""
    echo "--------------------------------------------"
    echo "Selecting $NUM_EPISODES_PER_DATASET episodes from [$i] $DNAME"
    echo "--------------------------------------------"

    python3 "$SELECTOR" \
        --dataset-path "$DROOT" \
        --dataset-name "$DNAME" \
        --output-dir "$OUTPUT_DIR" \
        --num-selected "$NUM_EPISODES_PER_DATASET" \
        --seed "$SEED" \
        --pixel-camera-key "observation.images.top" \
        --pixel-size 64 \
        --region-ratio 0.1 \
        --min-regions 4 \
        --max-regions 16 \
        --b0-size 4 \
        --b0-region-ratio 0.5 \
        --coverage-weight 0.5 \
        --region-visual-weight 0.3 \
        --region-action-weight 0.2 \
        --candidate-visual-weight 0.5 \
        --candidate-action-weight 0.5

    echo "Selection complete for $DNAME"
done

echo ""
echo "============================================"
echo "All selections complete"
echo "============================================"

# Validate selection manifest
MANIFEST="${OUTPUT_DIR}/selection_manifest.json"
if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Selection manifest not found at $MANIFEST"
    exit 1
fi

echo "Selection manifest: $MANIFEST"
python3 -c "
import json
with open('$MANIFEST') as f:
    manifest = json.load(f)
print(f'Datasets in manifest: {len(manifest[\"datasets\"])}')
for ds in manifest['datasets']:
    print(f'  {ds[\"name\"]}: {ds[\"num_selected\"]} episodes')
"

if [ "$MODE" = "selection-only" ]; then
    echo ""
    echo "============================================"
    echo "Selection-only mode: stopping here"
    echo "============================================"
    echo "Manifest: $MANIFEST"
    echo "Subsets: ${OUTPUT_DIR}/subsets/"
    exit 0
fi

# ---- Phase 2: Merge datasets ----
echo ""
echo "============================================"
echo "Phase 2: Merge selected episodes"
echo "============================================"

MERGE_SCRIPT="personal/work1/merge_dataset.py"
MERGED_OUTPUT="${OUTPUT_DIR}/merged_dataset"

python3 "$MERGE_SCRIPT" \
    --manifest "$MANIFEST" \
    --output-dir "$MERGED_OUTPUT" \
    --merged-name "our_v5_real_merged"

echo ""
echo "Merged dataset: $MERGED_OUTPUT"

# ---- Phase 3: Training ----
echo ""
echo "============================================"
echo "Phase 3: Training"
echo "============================================"

TRAIN_OUTPUT="${OUTPUT_DIR}/training"
mkdir -p "$TRAIN_OUTPUT"

python3 train.py \
    --policy.path=lerobot/smolvla_base \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id=our_v5_real_merged \
    --dataset.root="$MERGED_OUTPUT" \
    --dataset.eval_split=0.0 \
    --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
    --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=true \
    --policy.train_state_proj=false \
    --policy.optimizer_lr=1e-4 \
    --save_freq=2000 \
    --steps=12000 \
    --batch_size=64 \
    --num_workers=16 \
    --seed="$SEED" \
    --remove_features='["observation.environment_state"]' \
    --output.dir="$TRAIN_OUTPUT" \
    --job_name="our_v5_real_${SEED}"

echo ""
echo "============================================"
echo "Training complete at $(date)"
echo "============================================"
echo "Output: $TRAIN_OUTPUT"
