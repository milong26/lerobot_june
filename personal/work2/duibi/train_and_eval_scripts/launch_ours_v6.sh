#!/bin/bash
# Compatibility launcher for the rewritten causal Our-V6 pipeline.
# Usage:
#   bash personal/work2/duibi/train_and_eval_scripts/launch_ours_v6.sh \
#        [episodes_per_dataset] [gpu_id] [seed] [visual_variant] [mode] [ablation]
#
# Example:
#   bash personal/work2/duibi/train_and_eval_scripts/launch_ours_v6.sh 112 0 42 l2 full full

set -euo pipefail

NUM_EPISODES=${1:-112}
GPU_ID=${2:-0}
SEED=${3:-42}
VISUAL_VARIANT=${4:-l2}
MODE=${5:-full}
ABLATION=${6:-full}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
PIPELINE="$REPO_ROOT/personal/work2/our_v6/run_select_train.sh"

if [[ ! -f "$PIPELINE" ]]; then
    echo "ERROR: Our-V6 pipeline not found: $PIPELINE"
    exit 1
fi

SESSION="our_v6_${VISUAL_VARIANT}_${ABLATION}_${NUM_EPISODES}x3_s${SEED}"
OUTPUT_TAG="our_v6_${VISUAL_VARIANT}_${ABLATION}_${NUM_EPISODES}x3_seed${SEED}"
OUTPUT_BASE="$REPO_ROOT/personal/work2/our_v6/outputs/$OUTPUT_TAG"
mkdir -p "$OUTPUT_BASE/logs"

# Keep the old launch behavior: start in tmux and return immediately.
tmux kill-session -t "$SESSION" 2>/dev/null || true
CMD="cd '$REPO_ROOT' && bash '$PIPELINE' '$NUM_EPISODES' '$GPU_ID' '$SEED' '$VISUAL_VARIANT' '$MODE' '$ABLATION'"
tmux new-session -d -s "$SESSION" "$CMD"

echo "Launched causal Our-V6"
echo "tmux session: $SESSION"
echo "visual variant: $VISUAL_VARIANT"
echo "ablation: $ABLATION"
echo "mode: $MODE"
echo "output: $OUTPUT_BASE"
echo "attach: tmux attach -t $SESSION"
echo "log: tail -f $OUTPUT_BASE/logs/pipeline.log"
