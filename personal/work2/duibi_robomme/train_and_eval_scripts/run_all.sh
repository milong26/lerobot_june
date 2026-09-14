#!/bin/bash
# Run all methods and seeds for the Robomme comparison pipeline.
# Usage:
#   bash run_all.sh \
#       --k 28 \
#       --seeds 42,123,456 \
#       --move-cube-repo-id work2/robomme_MoveCube_easy \
#       --pattern-lock-repo-id work2/robomme_PatternLock_medium \
#       --route-stick-repo-id work2/robomme_RouteStick_hard \
#       [--gpu-id 0] [--steps 12000] [--batch-size 64] [--n-eval-episodes 20]
#
# Methods: ours_v5, random, grid_uniform, deminf, fps
# For S seeds, produces 5*S models and 15*S task-level evaluations.

set -euo pipefail

# ─── Default values ──────────────────────────────────────────────────
GPU_ID=0
STEPS=12000
BATCH_SIZE=64
NUM_WORKERS=16
LEARNING_RATE=1e-4
SAVE_FREQ=2000
N_EVAL_EPISODES=20
EVAL_SEEDS="0,1,2,3,4"
WANDB_ENABLE=true
FORCE=0
METHODS="ours_v5 random grid_uniform deminf fps"
SEEDS=""
K=""
MOVE_CUBE_REPO_ID=""
PATTERN_LOCK_REPO_ID=""
ROUTE_STICK_REPO_ID=""
MOVE_CUBE_DATASET_DIR=""
PATTERN_LOCK_DATASET_DIR=""
ROUTE_STICK_DATASET_DIR=""

# ─── Resolve repo root ──────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$REPO_ROOT"

# ─── Parse named arguments ──────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --k)
            K="$2"; shift 2 ;;
        --seeds)
            SEEDS="$2"; shift 2 ;;
        --methods)
            METHODS="$2"; shift 2 ;;
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
        --eval-seeds)
            EVAL_SEEDS="$2"; shift 2 ;;
        --wandb-enable)
            WANDB_ENABLE="$2"; shift 2 ;;
        --no-wandb)
            WANDB_ENABLE=false; shift ;;
        --force)
            FORCE=1; shift ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ─── Validate required arguments ────────────────────────────────────
if [ -z "$K" ]; then
    echo "Error: --k is required"
    exit 1
fi
if [ -z "$SEEDS" ]; then
    echo "Error: --seeds is required (comma-separated list)"
    exit 1
fi

echo "========================================"
echo "Robomme Full Experiment Launcher"
echo "========================================"
echo "Methods: $METHODS"
echo "Seeds: $SEEDS"
echo "K per task: $K"
echo "GPU: $GPU_ID"
echo "Steps: $STEPS"
echo "Batch size: $BATCH_SIZE"
echo "========================================"

# Parse seeds into array
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
IFS=' ' read -ra METHOD_ARRAY <<< "$METHODS"

TOTAL_RUNS=$(( ${#METHOD_ARRAY[@]} * ${#SEED_ARRAY[@]} ))
echo "Total runs: $TOTAL_RUNS (${#METHOD_ARRAY[@]} methods x ${#SEED_ARRAY[@]} seeds)"
echo ""

PIPELINE_SCRIPT="$REPO_ROOT/personal/work2/duibi_robomme/train_and_eval_scripts/run_robomme_pipeline.sh"

RUN_IDX=0
for METHOD in "${METHOD_ARRAY[@]}"; do
    for SEED in "${SEED_ARRAY[@]}"; do
        RUN_IDX=$((RUN_IDX + 1))
        echo ""
        echo "========================================"
        echo "Run $RUN_IDX / $TOTAL_RUNS"
        echo "Method: $METHOD"
        echo "Seed: $SEED"
        echo "K: $K"
        echo "========================================"

        CMD="bash $PIPELINE_SCRIPT"
        CMD="$CMD --method $METHOD"
        CMD="$CMD --seed $SEED"
        CMD="$CMD --k $K"
        CMD="$CMD --gpu-id $GPU_ID"
        CMD="$CMD --steps $STEPS"
        CMD="$CMD --batch-size $BATCH_SIZE"
        CMD="$CMD --num-workers $NUM_WORKERS"
        CMD="$CMD --lr $LEARNING_RATE"
        CMD="$CMD --save-freq $SAVE_FREQ"
        CMD="$CMD --n-eval-episodes $N_EVAL_EPISODES"
        CMD="$CMD --eval-seeds $EVAL_SEEDS"
        CMD="$CMD --wandb-enable $WANDB_ENABLE"

        if [ -n "$MOVE_CUBE_REPO_ID" ]; then
            CMD="$CMD --move-cube-repo-id $MOVE_CUBE_REPO_ID"
        fi
        if [ -n "$PATTERN_LOCK_REPO_ID" ]; then
            CMD="$CMD --pattern-lock-repo-id $PATTERN_LOCK_REPO_ID"
        fi
        if [ -n "$ROUTE_STICK_REPO_ID" ]; then
            CMD="$CMD --route-stick-repo-id $ROUTE_STICK_REPO_ID"
        fi
        if [ -n "$MOVE_CUBE_DATASET_DIR" ]; then
            CMD="$CMD --move-cube-dataset-dir $MOVE_CUBE_DATASET_DIR"
        fi
        if [ -n "$PATTERN_LOCK_DATASET_DIR" ]; then
            CMD="$CMD --pattern-lock-dataset-dir $PATTERN_LOCK_DATASET_DIR"
        fi
        if [ -n "$ROUTE_STICK_DATASET_DIR" ]; then
            CMD="$CMD --route-stick-dataset-dir $ROUTE_STICK_DATASET_DIR"
        fi
        if [ "${FORCE:-0}" == "1" ]; then
            CMD="$CMD --force"
        fi

        echo "Running: $CMD"
        eval "$CMD"

        if [ $? -ne 0 ]; then
            echo "Error: Run $RUN_IDX failed (method=$METHOD, seed=$SEED)"
            echo "Continuing to next run..."
        fi
    done
done

echo ""
echo "========================================"
echo "All runs complete!"
echo "========================================"
echo "Global results: $REPO_ROOT/personal/work2/duibi_robomme/results/all_results.csv"
echo "Global JSON: $REPO_ROOT/personal/work2/duibi_robomme/results/all_results.json"
echo "========================================"