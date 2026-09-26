#!/bin/bash
# Run a single Robomme pipeline: selection -> merge -> train -> eval -> aggregate
# Usage:
#   bash run_robomme_pipeline.sh \
#       --method ours_v5|random|grid_uniform|deminf|fps \
#       --seed 42 \
#       --k 28 \
#       --move-cube-repo-id work2/robomme_MoveCube_easy \
#       --pattern-lock-repo-id work2/robomme_PatternLock_medium \
#       --route-stick-repo-id work2/robomme_RouteStick_hard \
#       [--gpu-id 0] [--steps 12000] [--batch-size 64] [--n-eval-episodes 20]
#
# All outputs go to: personal/work2/duibi_robomme/outputs/<method>/seed_<seed>/k_<K>/

set -euo pipefail

# ─── Default values ──────────────────────────────────────────────────
GPU_ID=0
STEPS=12000
BATCH_SIZE=64
NUM_WORKERS=16
LEARNING_RATE=1e-4
SAVE_FREQ=2000
N_EVAL_EPISODES=20
TRAIN_EVAL_EPISODES=5
ENV_EVAL_FREQ=2000
WANDB_ENABLE=true
FORCE=0
MAX_STEPS=300

# ─── Resolve repo root ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

# ─── Parse named arguments ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="$2"; shift 2 ;;
        --seed)
            SEED="$2"; shift 2 ;;
        --k)
            K="$2"; shift 2 ;;
        --move-cube-repo-id)
            MOVE_CUBE_REPO_ID="$2"; shift 2 ;;
        --pattern-lock-repo-id)
            PATTERN_LOCK_REPO_ID="$2"; shift 2 ;;
        --route-stick-repo-id)
            ROUTE_STICK_REPO_ID="$2"; shift 2 ;;
        --move-cube-dataset-dir)
            MOVE_CUBE_DATASET_DIR="$2"; shift 2 ;;
        --pattern-lock-dataset-dir)
            PATTERN_LOCK_DATASET_DIR="$2"; shift 2 ;;
        --route-stick-dataset-dir)
            ROUTE_STICK_DATASET_DIR="$2"; shift 2 ;;
        --gpu-id)
            GPU_ID="$2"; shift 2 ;;
        --steps)
            STEPS="$2"; shift 2 ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --num-workers)
            NUM_WORKERS="$2"; shift 2 ;;
        --lr)
            LEARNING_RATE="$2"; shift 2 ;;
        --save-freq)
            SAVE_FREQ="$2"; shift 2 ;;
        --n-eval-episodes)
            N_EVAL_EPISODES="$2"; shift 2 ;;
        --train-eval-episodes)
            TRAIN_EVAL_EPISODES="$2"; shift 2 ;;
        --env-eval-freq)
            ENV_EVAL_FREQ="$2"; shift 2 ;;
        --wandb-enable)
            WANDB_ENABLE="$2"; shift 2 ;;
        --no-wandb)
            WANDB_ENABLE=false; shift ;;
        --force)
            FORCE=1; shift ;;
        --max-steps)
            MAX_STEPS="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── Validate required arguments ────────────────────────────────────
if [ -z "${METHOD:-}" ]; then
    echo "Error: --method is required (ours_v5|random|grid_uniform|deminf|fps)"
    exit 1
fi
if [ -z "${SEED:-}" ]; then
    echo "Error: --seed is required"
    exit 1
fi
if [ -z "${K:-}" ]; then
    echo "Error: --k is required"
    exit 1
fi

# Normalize method name
METHOD=$(echo "$METHOD" | tr '-' '_')
case "$METHOD" in
    ours_v5|random|grid_uniform|deminf|fps) ;;
    *)
        echo "Error: Unknown method: $METHOD. Must be one of: ours_v5, random, grid_uniform, deminf, fps"
        exit 1
        ;;
esac

# ─── Setup paths ────────────────────────────────────────────────────
OUTPUT_BASE="$REPO_ROOT/personal/work2/duibi_robomme/outputs/$METHOD/seed_${SEED}/k_${K}"
SELECT_DIR="$OUTPUT_BASE/selected"
MERGED_DIR="$OUTPUT_BASE/merged_dataset"
TRAIN_DIR="$OUTPUT_BASE/train"
EVAL_DIR="$OUTPUT_BASE/eval"
LOG_DIR="$OUTPUT_BASE/logs"
RESULTS_DIR="$REPO_ROOT/personal/work2/duibi_robomme/results"
GLOBAL_RESULTS_CSV="$RESULTS_DIR/all_results.csv"
GLOBAL_RESULTS_JSON="$RESULTS_DIR/all_results.json"

mkdir -p "$SELECT_DIR" "$MERGED_DIR" "$TRAIN_DIR" "$EVAL_DIR" "$LOG_DIR" "$RESULTS_DIR"

# Initialize dataset dir variables from parsed arguments (avoid set -u errors)
MOVE_CUBE_DIR="${MOVE_CUBE_DATASET_DIR:-}"
PATTERN_LOCK_DIR="${PATTERN_LOCK_DATASET_DIR:-}"
ROUTE_STICK_DIR="${ROUTE_STICK_DATASET_DIR:-}"

# Build TASKS list from available datasets only
TASKS=()
if [ -n "$MOVE_CUBE_DIR" ] && [ -d "$MOVE_CUBE_DIR" ]; then
    TASKS+=("MoveCube_easy")
fi
if [ -n "$PATTERN_LOCK_DIR" ] && [ -d "$PATTERN_LOCK_DIR" ]; then
    TASKS+=("PatternLock_medium")
fi
if [ -n "$ROUTE_STICK_DIR" ] && [ -d "$ROUTE_STICK_DIR" ]; then
    TASKS+=("RouteStick_hard")
fi

if [ ${#TASKS[@]} -eq 0 ]; then
    echo "Error: No valid dataset directories found"
    exit 1
fi

echo "Active tasks: ${TASKS[*]}"
echo "Total tasks: ${#TASKS[@]}"

# Resolve dataset directories
resolve_dataset_dir() {
    local task_name="$1"
    local repo_id="${2:-}"
    local explicit_dir="${3:-}"

    if [ -n "$explicit_dir" ] && [ -d "$explicit_dir" ]; then
        echo "$explicit_dir"
        return
    fi

    if [ -n "$repo_id" ]; then
        local short_name="${repo_id#*/}"
        local candidate="$REPO_ROOT/personal/work2/dataset_view_robomme/${task_name}"
        if [ -d "$candidate" ]; then
            echo "$candidate"
            return
        fi
        candidate="$REPO_ROOT/personal/work2/dataset_view_robomme/${short_name}"
        if [ -d "$candidate" ]; then
            echo "$candidate"
            return
        fi
    fi

    echo ""
}

MOVE_CUBE_DIR=$(resolve_dataset_dir "MoveCube_easy" "${MOVE_CUBE_REPO_ID:-}" "${MOVE_CUBE_DATASET_DIR:-}")
PATTERN_LOCK_DIR=$(resolve_dataset_dir "PatternLock_medium" "${PATTERN_LOCK_REPO_ID:-}" "${PATTERN_LOCK_DATASET_DIR:-}")
ROUTE_STICK_DIR=$(resolve_dataset_dir "RouteStick_hard" "${ROUTE_STICK_REPO_ID:-}" "${ROUTE_STICK_DATASET_DIR:-}")

# Validate dataset directories
for task_dir_pair in "MoveCube_easy:$MOVE_CUBE_DIR" "PatternLock_medium:$PATTERN_LOCK_DIR" "RouteStick_hard:$ROUTE_STICK_DIR"; do
    task_name="${task_dir_pair%%:*}"
    task_dir="${task_dir_pair#*:}"
    if [ -z "$task_dir" ] || [ ! -d "$task_dir" ]; then
        echo "Error: Dataset directory for $task_name not found."
        echo "  Please provide --${task_name,,}-dataset-dir or ensure the dataset exists."
        exit 1
    fi
done

echo "========================================"
echo "Robomme Pipeline: $METHOD"
echo "Seed: $SEED"
echo "K per task: $K"
echo "GPU: $GPU_ID"
echo "========================================"
echo "MoveCube_easy: $MOVE_CUBE_DIR"
echo "PatternLock_medium: $PATTERN_LOCK_DIR"
echo "RouteStick_hard: $ROUTE_STICK_DIR"
echo "Output: $OUTPUT_BASE"
echo "========================================"

# ─── Stage completion markers ───────────────────────────────────────
MARKER_DIR="$OUTPUT_BASE/.markers"
mkdir -p "$MARKER_DIR"

# FORCE_ARGS: only add --force when FORCE==1
FORCE_ARGS=()
if [ "$FORCE" -eq 1 ]; then
    FORCE_ARGS+=(--force)
fi

check_stage() {
    local stage="$1"
    local marker="$MARKER_DIR/${stage}.done"
    if [ -f "$marker" ] && [ "${FORCE}" != "1" ]; then
        # Verify key artifact exists
        case "$stage" in
            select_MoveCube_easy|select_PatternLock_medium|select_RouteStick_hard)
                local task_name="${stage#select_}"
                local expected_file="$SELECT_DIR/$task_name/${METHOD}_${K}_seed${SEED}.json"
                if [ -f "$expected_file" ]; then
                    return 0
                fi
                echo "  [RE-EXEC] Selection marker exists but subset JSON missing for $task_name"
                ;;
            merge)
                if [ -d "$MERGED_DIR" ]; then
                    local ep_count
                    ep_count=$(python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id='1/2', root='$MERGED_DIR')
print(ds.num_episodes)
" 2>/dev/null || echo "0")
                    local expected_total=$((K * 3))
                    if [ "$ep_count" -eq "$expected_total" ]; then
                        return 0
                    fi
                    echo "  [RE-EXEC] Merge marker exists but episode count mismatch: got $ep_count, expected $expected_total"
                else
                    echo "  [RE-EXEC] Merge marker exists but merged dataset directory missing"
                fi
                ;;
            train)
                if [ -f "$TRAIN_DIR/final_checkpoint.txt" ]; then
                    local ckpt_path
                    ckpt_path=$(cat "$TRAIN_DIR/final_checkpoint.txt" 2>/dev/null || echo "")
                    if [ -n "$ckpt_path" ] && [ -d "$ckpt_path" ]; then
                        return 0
                    fi
                    echo "  [RE-EXEC] Train marker exists but checkpoint directory missing"
                else
                    echo "  [RE-EXEC] Train marker exists but final_checkpoint.txt missing"
                fi
                ;;
            eval_MoveCube_easy|eval_PatternLock_medium|eval_RouteStick_hard)
                local task_name="${stage#eval_}"
                local result_file="$EVAL_DIR/$task_name/eval_result.json"
                if [ -f "$result_file" ]; then
                    return 0
                fi
                echo "  [RE-EXEC] Eval marker exists but eval_result.json missing for $task_name"
                ;;
            *)
                return 0
                ;;
        esac
    fi
    return 1
}

mark_stage() {
    local stage="$1"
    echo "$(date '+%Y-%m-%d %H:%M:%S')" > "$MARKER_DIR/${stage}.done"
}

# ─── Stage 1: Selection ─────────────────────────────────────────────
echo ""
echo "========================================"
echo "Stage 1: Episode Selection ($METHOD)"
echo "========================================"

SELECTOR_SCRIPT="$REPO_ROOT/personal/work2/duibi_robomme/selectors/select_${METHOD}.py"
if [ ! -f "$SELECTOR_SCRIPT" ]; then
    echo "Error: Selector script not found: $SELECTOR_SCRIPT"
    exit 1
fi

SELECTED_FILES=()
for task_name in "${TASKS[@]}"; do
    case "$task_name" in
        MoveCube_easy) DATASET_DIR="$MOVE_CUBE_DIR" ;;
        PatternLock_medium) DATASET_DIR="$PATTERN_LOCK_DIR" ;;
        RouteStick_hard) DATASET_DIR="$ROUTE_STICK_DIR" ;;
    esac

    TASK_SELECT_DIR="$SELECT_DIR/$task_name"
    mkdir -p "$TASK_SELECT_DIR"

    STAGE_KEY="select_${task_name}"
    if check_stage "$STAGE_KEY"; then
        echo "  [SKIP] Selection for $task_name already complete"
    else
        echo "  Selecting $K episodes from $task_name..."
        python "$SELECTOR_SCRIPT" \
            --num-episodes "$K" \
            --seed "$SEED" \
            --dataset-root "$DATASET_DIR" \
            --task-name "$task_name" \
            --output-dir "$TASK_SELECT_DIR"

        # Verify selection using exact filename
        SUBSET_FILE="$TASK_SELECT_DIR/${METHOD}_${K}_seed${SEED}.json"
        if [ ! -f "$SUBSET_FILE" ]; then
            echo "Error: Expected subset file not found: $SUBSET_FILE"
            exit 1
        fi

        # Validate metadata matches current run
        python -c "
import json, sys
with open('$SUBSET_FILE') as f:
    data = json.load(f)
assert data.get('method') == '$METHOD', f\"method mismatch: {data.get('method')}\"
assert data.get('seed') == $SEED, f\"seed mismatch: {data.get('seed')}\"
assert data.get('num_episodes') == $K, f\"num_episodes mismatch: {data.get('num_episodes')}\"
assert len(data['selected_episode_indices']) == $K, f\"selected count mismatch\"
" || { echo "Error: Subset file metadata validation failed for $task_name"; exit 1; }

        N_SELECTED=$(python -c "import json; print(len(json.load(open('$SUBSET_FILE'))['selected_episode_indices']))")
        if [ "$N_SELECTED" -ne "$K" ]; then
            echo "Error: $task_name selected $N_SELECTED episodes, expected $K"
            exit 1
        fi
        echo "  $task_name: $N_SELECTED episodes selected -> $SUBSET_FILE"
        mark_stage "$STAGE_KEY"
    fi

    SUBSET_FILE="$TASK_SELECT_DIR/${METHOD}_${K}_seed${SEED}.json"
    SELECTED_FILES+=("$SUBSET_FILE")
done

# ─── Stage 2: Merge ─────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Stage 2: Merge Selected Datasets"
echo "========================================"

if check_stage "merge"; then
    echo "  [SKIP] Merge already complete"
else
    SOURCE_DATASETS=""
    for task_name in "${TASKS[@]}"; do
        case "$task_name" in
            MoveCube_easy) DATASET_DIR="$MOVE_CUBE_DIR" ;;
            PatternLock_medium) DATASET_DIR="$PATTERN_LOCK_DIR" ;;
            RouteStick_hard) DATASET_DIR="$ROUTE_STICK_DIR" ;;
        esac
        SOURCE_DATASETS="$SOURCE_DATASETS ${task_name}=${DATASET_DIR}"
    done

    SELECTED_FILES_STR=""
    for f in "${SELECTED_FILES[@]}"; do
        SELECTED_FILES_STR="$SELECTED_FILES_STR $f"
    done

    python "$REPO_ROOT/personal/work2/duibi_robomme/utils/merge_selected_episodes.py" \
        --source-datasets $SOURCE_DATASETS \
        --selected-episodes $SELECTED_FILES_STR \
        --method "$METHOD" \
        --seed "$SEED" \
        --k "$K" \
        --output-dir "$MERGED_DIR"

    if [ ! -d "$MERGED_DIR" ]; then
        echo "Error: Merged dataset directory not created"
        exit 1
    fi

    MERGED_EP_COUNT=$(python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id='1/2', root='$MERGED_DIR')
print(ds.num_episodes)
")
    EXPECTED_TOTAL=$((K * ${#TASKS[@]}))
    if [ "$MERGED_EP_COUNT" -ne "$EXPECTED_TOTAL" ]; then
        echo "Error: Merged dataset has $MERGED_EP_COUNT episodes, expected $EXPECTED_TOTAL (${#TASKS[@]} tasks * $K)"
        exit 1
    fi
    echo "  Merged dataset: $MERGED_EP_COUNT episodes (expected $EXPECTED_TOTAL)"
    mark_stage "merge"
fi

# ─── Stage 3: Training ──────────────────────────────────────────────
echo ""
echo "========================================"
echo "Stage 3: SmolVLA Training"
echo "========================================"

if check_stage "train"; then
    echo "  [SKIP] Training already complete"
else
    bash "$REPO_ROOT/personal/work2/duibi_robomme/train_and_eval_scripts/train_smolvla_robomme.sh" \
        --merged-dataset-dir "$MERGED_DIR" \
        --output-dir "$TRAIN_DIR" \
        --seed "$SEED" \
        --method "$METHOD" \
        --k "$K" \
        --gpu-id "$GPU_ID" \
        --steps "$STEPS" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --lr "$LEARNING_RATE" \
        --save-freq "$SAVE_FREQ" \
        --env-eval-freq "$ENV_EVAL_FREQ" \
        --eval-n-episodes "$TRAIN_EVAL_EPISODES" \
        --eval-batch-size "$TRAIN_EVAL_EPISODES" \
        --eval-tasks "MoveCube,PatternLock,RouteStick" \
        --wandb-enable "$WANDB_ENABLE" \
        "${FORCE_ARGS[@]}"

    if [ ! -f "$TRAIN_DIR/final_checkpoint.txt" ]; then
        echo "Error: Training did not produce final checkpoint"
        exit 1
    fi
    mark_stage "train"
fi

CHECKPOINT_PATH=$(cat "$TRAIN_DIR/final_checkpoint.txt" 2>/dev/null || true)
if [ -z "$CHECKPOINT_PATH" ] || [ ! -d "$CHECKPOINT_PATH" ]; then
    CHECKPOINT_PATH=$(ls -d "$TRAIN_DIR"/checkpoints/*/ 2>/dev/null | sort -V | tail -n1 || true)
fi
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Error: No checkpoint found after training"
    exit 1
fi
echo "  Checkpoint: $CHECKPOINT_PATH"

# ─── Stage 4: Evaluation ────────────────────────────────────────────
echo ""
echo "========================================"
echo "Stage 4: Evaluation on Three Robomme Tasks"
echo "========================================"

EVAL_SCRIPT="$REPO_ROOT/personal/work2/duibi_robomme/utils/eval_robomme.py"

# Evaluate the first N fixed benchmark test episodes for every task.
EVAL_TASK_IDS=$(python -c "print(','.join(str(i) for i in range($N_EVAL_EPISODES)))")

for task_name in "${TASKS[@]}"; do
    TASK_EVAL_DIR="$EVAL_DIR/$task_name"
    mkdir -p "$TASK_EVAL_DIR"

    STAGE_KEY="eval_${task_name}"
    if check_stage "$STAGE_KEY"; then
        echo "  [SKIP] Evaluation for $task_name already complete"
    else
        echo "  Evaluating $task_name..."
        python "$EVAL_SCRIPT" \
            --checkpoint-path "$CHECKPOINT_PATH/pretrained_model" \
            --task "$task_name" \
            --eval-task-ids "$EVAL_TASK_IDS" \
            --output-dir "$TASK_EVAL_DIR" \
            --gpu-id "$GPU_ID" \
            --episode-length "$MAX_STEPS"

        if [ ! -f "$TASK_EVAL_DIR/eval_result.json" ]; then
            echo "Error: Evaluation result not found for $task_name"
            exit 1
        fi

        SUCCESS_RATE=$(python -c "import json; print(json.load(open('$TASK_EVAL_DIR/eval_result.json'))['success_rate'])")
        echo "  $task_name success rate: $SUCCESS_RATE"
        mark_stage "$STAGE_KEY"
    fi
done

# ─── Stage 5: Aggregate Results ─────────────────────────────────────
echo ""
echo "========================================"
echo "Stage 5: Aggregate Results"
echo "========================================"

if check_stage "aggregate"; then
    echo "  [SKIP] Aggregation already complete"
else
    python "$REPO_ROOT/personal/work2/duibi_robomme/utils/aggregate_results.py" \
        --eval-dir "$EVAL_DIR" \
        --method "$METHOD" \
        --seed "$SEED" \
        --k "$K" \
        --output-dir "$OUTPUT_BASE" \
        --global-results "$GLOBAL_RESULTS_CSV" \
        --merged-dataset-path "$MERGED_DIR" \
        --checkpoint-path "$CHECKPOINT_PATH"

    mark_stage "aggregate"
fi

echo ""
echo "========================================"
echo "Pipeline Complete: $METHOD seed=$SEED k=$K"
echo "========================================"
echo "Output: $OUTPUT_BASE"
echo "Summary: $OUTPUT_BASE/summary.json"
echo "Global results: $GLOBAL_RESULTS_CSV"
echo "========================================"