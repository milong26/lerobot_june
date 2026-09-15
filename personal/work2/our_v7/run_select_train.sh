#!/bin/bash
# Our-V7 end-to-end pipeline: paper-faithful causal selection on three tasks,
# merge selected episodes, then jointly train SmolVLA.
#
# Usage:
#   bash personal/work2/our_v7/run_select_train.sh \
#     [episodes_per_dataset] [gpu_id] [seed] [visual_variant] [mode] [ablation] [benchmark]
#
# visual_variant: l2 | online_pca
# mode: full | selection-only
# ablation: full | wo_action | wo_adaptive_priority
# benchmark: metaworld | robomme (default: metaworld, preserving the original V7 command)

set -euo pipefail

NUM_EPISODES=${1:-112}
GPU_ID=${2:-0}
SEED=${3:-42}
VISUAL_VARIANT=${4:-l2}
MODE=${5:-full}
ABLATION=${6:-full}
BENCHMARK=${7:-metaworld}

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
if [[ "$BENCHMARK" != "metaworld" && "$BENCHMARK" != "robomme" ]]; then
    echo "ERROR: benchmark must be metaworld or robomme"
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WORK2_ROOT="$REPO_ROOT/personal/work2"
MERGE_SCRIPT="$WORK2_ROOT/duibi/train_and_eval_scripts/merge_selected_episodes_safe.py"
RAW_VISUAL_CACHE_ROOT="$WORK2_ROOT/our_v7/shared_raw_embeddings"

if [[ "$BENCHMARK" == "metaworld" ]]; then
    DATASET_BASE="$WORK2_ROOT/dataset_view"
    DATASETS=("coffee-button-v3_corner" "disassemble-v3_corner" "pick_place-v3_corner")
    # Preserve the original output location so existing MetaWorld V7 selections
    # remain reusable when the 7th argument is omitted or explicitly metaworld.
    EXP_TAG="our_v7_${VISUAL_VARIANT}_${ABLATION}_${NUM_EPISODES}x3_seed${SEED}"
else
    DATASET_BASE="$WORK2_ROOT/dataset_view_robomme"
    DATASETS=("MoveCube_easy" "PatternLock_medium" "RouteStick_hard")
    EXP_TAG="our_v7_robomme_${VISUAL_VARIANT}_${ABLATION}_${NUM_EPISODES}x3_seed${SEED}"
fi

TOTAL_EPISODES=$((NUM_EPISODES * ${#DATASETS[@]}))
OUTPUT_BASE="$WORK2_ROOT/our_v7/outputs/$EXP_TAG"
SELECTION_DIR="$OUTPUT_BASE/selection"
SUBSET_DIR="$OUTPUT_BASE/subsets"
MERGED_DATASET_DIR="$OUTPUT_BASE/merged_dataset"
TRAIN_OUTPUT="$OUTPUT_BASE/train"
LOG_DIR="$OUTPUT_BASE/logs"
LOG_FILE="$LOG_DIR/pipeline.log"

mkdir -p "$SELECTION_DIR" "$SUBSET_DIR" "$LOG_DIR" "$RAW_VISUAL_CACHE_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1

cd "$REPO_ROOT"
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate lb_server || true
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$WORK2_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$BENCHMARK" == "metaworld" ]]; then
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    export MUJOCO_EGL_DEVICE_ID="$GPU_ID"
else
    # Keep the same device convention as the existing working RoboMME pipeline.
    export SAPIEN_VULKAN_DEVICE_INDEX="$GPU_ID"
fi
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    if [[ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
        export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
fi

echo "============================================================"
echo "Our-V7 paper-faithful end-to-end pipeline"
echo "benchmark:         $BENCHMARK"
echo "episodes/dataset:  $NUM_EPISODES"
echo "total selected:    $TOTAL_EPISODES"
echo "gpu:               $GPU_ID"
echo "seed:              $SEED"
echo "visual variant:    $VISUAL_VARIANT"
echo "ablation:          $ABLATION"
echo "mode:              $MODE"
echo "dataset base:      $DATASET_BASE"
echo "raw visual cache:  $RAW_VISUAL_CACHE_ROOT"
echo "output:            $OUTPUT_BASE"
echo "============================================================"

for DATASET_NAME in "${DATASETS[@]}"; do
    DATASET_ROOT="$DATASET_BASE/$DATASET_NAME"
    OUT="$SELECTION_DIR/$DATASET_NAME"
    SELECTION_FILE="$OUT/selected_episodes_v7.json"

    if [[ ! -d "$DATASET_ROOT" ]]; then
        echo "ERROR: dataset not found: $DATASET_ROOT"
        exit 1
    fi
    if [[ ! -f "$DATASET_ROOT/episode_initial_states.json" ]]; then
        echo "ERROR: configuration metadata not found: $DATASET_ROOT/episode_initial_states.json"
        exit 1
    fi

    REUSE_SELECTION=false
    if [[ -f "$SELECTION_FILE" ]]; then
        if python - "$SELECTION_FILE" "$NUM_EPISODES" "$VISUAL_VARIANT" "$ABLATION" "$BENCHMARK" <<'PYCHECK'
import json
import sys
path, expected_n, expected_variant, expected_ablation, expected_benchmark = sys.argv[1:]
try:
    data = json.load(open(path, "r"))
    ids = data.get("selected_episode_indices", [])
    cached_benchmark = data.get("benchmark")
    # V7 MetaWorld selections created before benchmark support did not yet store
    # this field; they are still the same MetaWorld algorithm and remain valid.
    benchmark_ok = (
        cached_benchmark == expected_benchmark
        or (expected_benchmark == "metaworld" and cached_benchmark is None)
    )
    ok = (
        data.get("method") == "our_v7"
        and data.get("paper_faithful_selection") is True
        and len(ids) == int(expected_n)
        and len(ids) == len(set(ids))
        and data.get("visual_variant") == expected_variant
        and data.get("ablation") == expected_ablation
        and benchmark_ok
    )
except Exception:
    ok = False
sys.exit(0 if ok else 1)
PYCHECK
        then
            REUSE_SELECTION=true
        fi
    fi

    echo ""
    if [[ "$REUSE_SELECTION" == true ]]; then
        echo "=== Reusing cached V7 selection for $DATASET_NAME ==="
        echo "Selection file: $SELECTION_FILE"
    else
        echo "=== V7 selecting $DATASET_NAME ($BENCHMARK) ==="
        rm -f "$SELECTION_FILE"
        python "$WORK2_ROOT/our_v7/experiments/select_episodes_v7.py" \
            --benchmark "$BENCHMARK" \
            --dataset-root "$DATASET_ROOT" \
            --dataset-name "$DATASET_NAME" \
            --output-dir "$OUT" \
            --total-budget "$NUM_EPISODES" \
            --visual-variant "$VISUAL_VARIANT" \
            --device cuda \
            --seed "$SEED" \
            --ablation "$ABLATION" \
            --raw-visual-cache-root "$RAW_VISUAL_CACHE_ROOT"
    fi
done

MERGED_SUBSET_FILE="$SUBSET_DIR/${EXP_TAG}.json"
DATASETS_CSV=$(IFS=,; echo "${DATASETS[*]}")
export SELECTION_DIR MERGED_SUBSET_FILE NUM_EPISODES SEED VISUAL_VARIANT ABLATION BENCHMARK DATASETS_CSV
python <<'PYEOF'
import json
import os
from pathlib import Path

datasets = [x for x in os.environ["DATASETS_CSV"].split(",") if x]
selection_dir = Path(os.environ["SELECTION_DIR"])
all_indices = []
per_dataset = {}
for ds in datasets:
    path = selection_dir / ds / "selected_episodes_v7.json"
    result = json.loads(path.read_text())
    indices = [int(x) for x in result["selected_episode_indices"]]
    expected = int(os.environ["NUM_EPISODES"])
    cached_benchmark = result.get("benchmark")
    benchmark_ok = (
        cached_benchmark == os.environ["BENCHMARK"]
        or (os.environ["BENCHMARK"] == "metaworld" and cached_benchmark is None)
    )
    if result.get("method") != "our_v7" or result.get("paper_faithful_selection") is not True:
        raise RuntimeError(f"{ds}: not a paper-faithful V7 selection")
    if not benchmark_ok:
        raise RuntimeError(f"{ds}: cached benchmark mismatch: {cached_benchmark}")
    if len(indices) != expected or len(indices) != len(set(indices)):
        raise RuntimeError(f"{ds}: invalid selected episode count/duplicates")
    if result.get("visual_variant") != os.environ["VISUAL_VARIANT"]:
        raise RuntimeError(f"{ds}: cached visual_variant mismatch")
    if result.get("ablation") != os.environ["ABLATION"]:
        raise RuntimeError(f"{ds}: cached ablation mismatch")
    all_indices.extend(indices)
    per_dataset[ds] = indices

payload = {
    "method": "our_v7",
    "paper_faithful_selection": True,
    "benchmark": os.environ["BENCHMARK"],
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
out.write_text(json.dumps(payload, indent=2))
print(f"Merged V7 selection manifest: {out}")
print(f"Benchmark: {os.environ['BENCHMARK']}")
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
echo "=== Training SmolVLA on merged Our-V7 $BENCHMARK dataset ==="
if [[ "$BENCHMARK" == "metaworld" ]]; then
    lerobot-train \
        --policy.path=lerobot/smolvla_base \
        --policy.device=cuda \
        --policy.push_to_hub=false \
        --dataset.repo_id=lerobot/metaworld_three_tasks_our_v7 \
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
        --steps=12000 \
        --batch_size=64 \
        --num_workers=16 \
        --env_eval_freq=0 \
        --seed="$SEED" \
        --job_name="smolvla_${EXP_TAG}" \
        --output_dir="$TRAIN_OUTPUT" \
        --remove_features='["observation.environment_state"]' \
        --wandb.enable=true
else
    # Keep the same working RoboMME SmolVLA configuration as robomme_pipeline.
    # The native dataset keys are observation.images.image / wrist_image and the
    # native joint-angle action is 8-D, so no MetaWorld rename_map is applied.
    lerobot-train \
        --policy.path=lerobot/smolvla_base \
        --policy.device=cuda \
        --policy.push_to_hub=false \
        --dataset.repo_id=work2/robomme_three_tasks_our_v7 \
        --dataset.root="$MERGED_DATASET_DIR" \
        --dataset.eval_split=0.0 \
        --env.type=robomme \
        --env.task=MoveCube \
        --env.action_space=joint_angle \
        --env.dataset_split=test \
        --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
        --policy.freeze_vision_encoder=true \
        --policy.train_expert_only=true \
        --policy.train_state_proj=false \
        --policy.optimizer_lr=1e-4 \
        --save_freq=2000 \
        --steps=12000 \
        --batch_size=64 \
        --num_workers=16 \
        --env_eval_freq=0 \
        --seed="$SEED" \
        --job_name="smolvla_${EXP_TAG}" \
        --output_dir="$TRAIN_OUTPUT" \
        --wandb.enable=true
fi

echo ""
echo "============================================================"
echo "Our-V7 pipeline complete"
echo "benchmark:           $BENCHMARK"
echo "selection manifest:  $MERGED_SUBSET_FILE"
echo "merged dataset:      $MERGED_DATASET_DIR"
echo "training output:     $TRAIN_OUTPUT"
echo "raw visual cache:    $RAW_VISUAL_CACHE_ROOT"
echo "log:                 $LOG_FILE"
echo "============================================================"
