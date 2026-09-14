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
#       [--fail-fast]
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
FAIL_FAST=true
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
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
        --fail-fast)
            FAIL_FAST=true; shift ;;
        --no-fail-fast)
            FAIL_FAST=false; shift ;;
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
echo "Fail-fast: $FAIL_FAST"
echo "========================================"

# Parse seeds into array
IFS=',' read -ra SEED_ARRAY <<< "$SEEDS"
IFS=' ' read -ra METHOD_ARRAY <<< "$METHODS"

TOTAL_RUNS=$(( ${#METHOD_ARRAY[@]} * ${#SEED_ARRAY[@]} ))
echo "Total runs: $TOTAL_RUNS (${#METHOD_ARRAY[@]} methods x ${#SEED_ARRAY[@]} seeds)"
echo ""

PIPELINE_SCRIPT="$REPO_ROOT/personal/work2/duibi_robomme/train_and_eval_scripts/run_robomme_pipeline.sh"
RESULTS_DIR="$REPO_ROOT/personal/work2/duibi_robomme/results"
mkdir -p "$RESULTS_DIR"
RUN_LOG="$RESULTS_DIR/run_log.txt"

RUN_IDX=0
FAILED_RUNS=()

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

        # Build command as a Bash array
        CMD=(
            bash "$PIPELINE_SCRIPT"
            --method "$METHOD"
            --seed "$SEED"
            --k "$K"
            --gpu-id "$GPU_ID"
            --steps "$STEPS"
            --batch-size "$BATCH_SIZE"
            --num-workers "$NUM_WORKERS"
            --lr "$LEARNING_RATE"
            --save-freq "$SAVE_FREQ"
            --n-eval-episodes "$N_EVAL_EPISODES"
            --eval-seeds "$EVAL_SEEDS"
            --wandb-enable "$WANDB_ENABLE"
        )

        if [ -n "$MOVE_CUBE_REPO_ID" ]; then
            CMD+=(--move-cube-repo-id "$MOVE_CUBE_REPO_ID")
        fi
        if [ -n "$PATTERN_LOCK_REPO_ID" ]; then
            CMD+=(--pattern-lock-repo-id "$PATTERN_LOCK_REPO_ID")
        fi
        if [ -n "$ROUTE_STICK_REPO_ID" ]; then
            CMD+=(--route-stick-repo-id "$ROUTE_STICK_REPO_ID")
        fi
        if [ -n "$MOVE_CUBE_DATASET_DIR" ]; then
            CMD+=(--move-cube-dataset-dir "$MOVE_CUBE_DATASET_DIR")
        fi
        if [ -n "$PATTERN_LOCK_DATASET_DIR" ]; then
            CMD+=(--pattern-lock-dataset-dir "$PATTERN_LOCK_DATASET_DIR")
        fi
        if [ -n "$ROUTE_STICK_DATASET_DIR" ]; then
            CMD+=(--route-stick-dataset-dir "$ROUTE_STICK_DATASET_DIR")
        fi
        if [ "$FORCE" -eq 1 ]; then
            CMD+=(--force)
        fi

        echo "Running: ${CMD[*]}"

        if ! "${CMD[@]}"; then
            echo "Error: Run $RUN_IDX failed (method=$METHOD, seed=$SEED)"
            FAILED_RUNS+=("method=$METHOD,seed=$SEED,k=$K")
            echo "$(date '+%Y-%m-%d %H:%M:%S') FAILED: method=$METHOD, seed=$SEED, k=$K" >> "$RUN_LOG"

            if [ "$FAIL_FAST" = true ]; then
                echo "Fail-fast enabled. Stopping."
                exit 1
            fi
            echo "Continuing to next run..."
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S') COMPLETED: method=$METHOD, seed=$SEED, k=$K" >> "$RUN_LOG"
        fi
    done
done

echo ""
echo "========================================"
if [ ${#FAILED_RUNS[@]} -gt 0 ]; then
    echo "Some runs failed:"
    for fail in "${FAILED_RUNS[@]}"; do
        echo "  - $fail"
    done
    echo "See run log: $RUN_LOG"
else
    echo "All runs complete!"
fi
echo "========================================"
echo "Global results: $REPO_ROOT/personal/work2/duibi_robomme/results/all_results.csv"
echo "Global JSON: $REPO_ROOT/personal/work2/duibi_robomme/results/all_results.json"
echo "========================================"