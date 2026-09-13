#!/bin/bash
# Launch our_v5_real experiment
# 基于真实第一帧像素的Episode选择 + 微调训练
#
# Usage: bash launch_our_v5_real.sh [gpu_id] [seed]
#   gpu_id: GPU ID (default: 0)
#   seed: Random seed (default: 42)
#
# 在tmux中运行，日志保存到 output_dir/logs/

set -e

GPU_ID=${1:-0}
SEED=${2:-42}
NUM_EPISODES_PER_TASK=20

# 路径配置
DATASET_ROOT="/data/zhonglinye/jun/ep10_30episodes_standing40/ep10_30episodes_standing40"
WORK_DIR="/data/zhonglinye/jun/lerobot/personal/work1"
OUTPUT_BASE_DIR="${WORK_DIR}/our_v5_real_${NUM_EPISODES_PER_TASK}ep_seed${SEED}"
LOG_DIR="${OUTPUT_BASE_DIR}/logs"
SUBSET_DIR="${OUTPUT_BASE_DIR}/subsets"

mkdir -p "$LOG_DIR" "$SUBSET_DIR"

# 日志文件
SELECTION_LOG="${LOG_DIR}/selection.log"
TRAINING_LOG="${LOG_DIR}/training.log"
FULL_LOG="${LOG_DIR}/full_experiment.log"

echo "========================================"
echo "Our V5 Real Experiment"
echo "GPU: $GPU_ID"
echo "Seed: $SEED"
echo "Episodes per task: $NUM_EPISODES_PER_TASK"
echo "Output: $OUTPUT_BASE_DIR"
echo "Logs: $LOG_DIR"
echo "========================================"

# 如果已经在tmux中，直接运行
if [ -n "$TMUX" ]; then
    echo "Running inside tmux..."
    
    # Step 1: Episode Selection
    echo "" | tee -a "$FULL_LOG"
    echo "=== Step 1: Episode Selection ===" | tee -a "$FULL_LOG"

    SELECT_SCRIPT="${WORK_DIR}/select_our_v5_real.py"

    python "$SELECT_SCRIPT" \
        --dataset-root "$DATASET_ROOT" \
        --output-dir "$OUTPUT_BASE_DIR" \
        --num-selected "$NUM_EPISODES_PER_TASK" \
        --batch-size 4 \
        --seed "$SEED" \
        --gpu-id "$GPU_ID" \
        --pca-dim 32 \
        --region-ratio 0.3 \
        --min-regions 8 \
        --max-regions 32 \
        --coverage-weight 0.5 \
        --visual-weight 0.3 2>&1 | tee "$SELECTION_LOG" | tee -a "$FULL_LOG"

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "ERROR: Episode selection failed" | tee -a "$FULL_LOG"
        exit 1
    fi

    # 读取选中的episodes
    EPISODES_FILE="${OUTPUT_BASE_DIR}/episodes.txt"
    if [ ! -f "$EPISODES_FILE" ]; then
        echo "ERROR: Episodes file not found: $EPISODES_FILE" | tee -a "$FULL_LOG"
        exit 1
    fi

    EPISODES=$(cat "$EPISODES_FILE")
    echo "" | tee -a "$FULL_LOG"
    echo "Selected episodes: $EPISODES" | tee -a "$FULL_LOG"

    # Step 2: Training
    echo "" | tee -a "$FULL_LOG"
    echo "=== Step 2: Training ===" | tee -a "$FULL_LOG"

    export CUDA_VISIBLE_DEVICES=$GPU_ID
    export HF_HUB_OFFLINE=1

    cd /data/zhonglinye/jun/lerobot

    lerobot-train \
        --dataset.repo_id=ep10/standing40 \
        --dataset.root="$DATASET_ROOT" \
        --dataset.episodes="$EPISODES" \
        --policy.path=lerobot/smolvla_base \
        --output_dir="$OUTPUT_BASE_DIR/training" \
        --job_name=smolvla_our_v5_real \
        --policy.device=cuda \
        --policy.push_to_hub=false \
        --wandb.enable=false \
        --steps=16000 \
        --batch_size=32 \
        --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' 2>&1 | tee "$TRAINING_LOG" | tee -a "$FULL_LOG"

    echo "" | tee -a "$FULL_LOG"
    echo "========================================" | tee -a "$FULL_LOG"
    echo "Experiment completed!" | tee -a "$FULL_LOG"
    echo "Output: $OUTPUT_BASE_DIR" | tee -a "$FULL_LOG"
    echo "Logs: $LOG_DIR" | tee -a "$FULL_LOG"
    echo "========================================" | tee -a "$FULL_LOG"
else
    # 不在tmux中，创建tmux session
    SESSION_NAME="our_v5_real_seed${SEED}"
    
    # 如果session已存在，先删除
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    
    # 创建tmux session
    tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50
    
    # 发送命令到tmux
    tmux send-keys -t "$SESSION_NAME" "cd /data/zhonglinye/jun/lerobot" C-m
    tmux send-keys -t "$SESSION_NAME" "echo '========================================'" C-m
    tmux send-keys -t "$SESSION_NAME" "echo 'Our V5 Real Experiment'" C-m
    tmux send-keys -t "$SESSION_NAME" "echo 'GPU: $GPU_ID'" C-m
    tmux send-keys -t "$SESSION_NAME" "echo 'Seed: $SEED'" C-m
    tmux send-keys -t "$SESSION_NAME" "echo 'Output: $OUTPUT_BASE_DIR'" C-m
    tmux send-keys -t "$SESSION_NAME" "echo 'Logs: $LOG_DIR'" C-m
    tmux send-keys -t "$SESSION_NAME" "echo '========================================'" C-m
    tmux send-keys -t "$SESSION_NAME" "" C-m
    
    # Step 1: Episode Selection
    tmux send-keys -t "$SESSION_NAME" "echo '=== Step 1: Episode Selection ==='" C-m
    tmux send-keys -t "$SESSION_NAME" "SELECT_SCRIPT=\"${WORK_DIR}/select_our_v5_real.py\"" C-m
    tmux send-keys -t "$SESSION_NAME" "python \"\$SELECT_SCRIPT\" \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --dataset-root \"$DATASET_ROOT\" \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --output-dir \"$OUTPUT_BASE_DIR\" \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --num-selected \"$NUM_EPISODES_PER_TASK\" \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --batch-size 4 \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --seed \"$SEED\" \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --gpu-id \"$GPU_ID\" \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --pca-dim 32 \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --region-ratio 0.3 \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --min-regions 8 \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --max-regions 32 \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --coverage-weight 0.5 \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --visual-weight 0.3 2>&1 | tee \"${SELECTION_LOG}\" | tee -a \"${FULL_LOG}\"" C-m
    tmux send-keys -t "$SESSION_NAME" "" C-m
    
    # Step 2: Training
    tmux send-keys -t "$SESSION_NAME" "echo '=== Step 2: Training ==='" C-m
    tmux send-keys -t "$SESSION_NAME" "export CUDA_VISIBLE_DEVICES=$GPU_ID" C-m
    tmux send-keys -t "$SESSION_NAME" "export HF_HUB_OFFLINE=1" C-m
    tmux send-keys -t "$SESSION_NAME" "EPISODES=\$(cat \"${OUTPUT_BASE_DIR}/episodes.txt\")" C-m
    tmux send-keys -t "$SESSION_NAME" "echo \"Selected episodes: \$EPISODES\"" C-m
    tmux send-keys -t "$SESSION_NAME" "" C-m
    tmux send-keys -t "$SESSION_NAME" "cd /data/zhonglinye/jun/lerobot" C-m
    tmux send-keys -t "$SESSION_NAME" "lerobot-train \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --dataset.repo_id=ep10/standing40 \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --dataset.root=\"$DATASET_ROOT\" \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --dataset.episodes=\"\$EPISODES\" \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --policy.path=lerobot/smolvla_base \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --output_dir=\"$OUTPUT_BASE_DIR/training\" \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --job_name=smolvla_our_v5_real \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --policy.device=cuda \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --policy.push_to_hub=false \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --wandb.enable=false \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --steps=16000 \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --batch_size=32 \\" C-m
    tmux send-keys -t "$SESSION_NAME" "    --rename_map='{\"observation.images.top\":\"observation.images.camera1\",\"observation.images.wrist\":\"observation.images.camera2\"}' 2>&1 | tee \"${TRAINING_LOG}\" | tee -a \"${FULL_LOG}\"" C-m
    tmux send-keys -t "$SESSION_NAME" "" C-m
    tmux send-keys -t "$SESSION_NAME" "echo '========================================'" C-m
    tmux send-keys -t "$SESSION_NAME" "echo 'Experiment completed!'" C-m
    tmux send-keys -t "$SESSION_NAME" "echo 'Output: $OUTPUT_BASE_DIR'" C-m
    tmux send-keys -t "$SESSION_NAME" "echo 'Logs: $LOG_DIR'" C-m
    tmux send-keys -t "$SESSION_NAME" "echo '========================================'" C-m
    
    echo ""
    echo "========================================"
    echo "Tmux session created: $SESSION_NAME"
    echo "========================================"
    echo ""
    echo "查看tmux: tmux attach -t $SESSION_NAME"
    echo "查看日志: tail -f $FULL_LOG"
    echo ""
    echo "日志文件:"
    echo "  选择日志: $SELECTION_LOG"
    echo "  训练日志: $TRAINING_LOG"
    echo "  完整日志: $FULL_LOG"
    echo ""
fi