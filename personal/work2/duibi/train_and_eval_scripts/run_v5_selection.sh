#!/bin/bash
# 为三个数据集运行V5 episode选择，生成merge脚本需要的subset文件
# 注意：只运行V5选择，不启动训练

set -e

GPU_ID=${1:-0}
EPISODES_PER_DATASET=112
SEED=42
V5_OUTPUT_BASE="/data/zhonglinye/jun/lerobot/personal/work2/duibi/our_v5_multi_336_seed42/subsets"
DATASET_ROOT="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view"

echo "========================================"
echo "为三个数据集运行V5 Episode选择"
echo "GPU: $GPU_ID"
echo "Episodes per dataset: $EPISODES_PER_DATASET"
echo "Seed: $SEED"
echo "V5 Output Base: $V5_OUTPUT_BASE"
echo "========================================"

# 数据集列表
DATASETS=("disassemble-v3_corner" "pick_place-v3_corner" "coffee-button-v3_corner")

for DATASET_NAME in "${DATASETS[@]}"; do
    echo ""
    echo "========================================"
    echo "处理数据集: $DATASET_NAME"
    echo "========================================"
    
    # merge脚本期望的位置
    MERGE_EXPECTED="${V5_OUTPUT_BASE}/our_v5_${EPISODES_PER_DATASET}_seed${SEED}_${DATASET_NAME}/subsets/our_v5_${EPISODES_PER_DATASET}_seed${SEED}.json"
    
    # 如果subset文件已存在，跳过
    if [ -f "$MERGE_EXPECTED" ]; then
        echo "✓ Subset文件已存在，跳过: $MERGE_EXPECTED"
        continue
    fi
    
    # 使用专用的V5选择脚本（不训练）
    cd /data/zhonglinye/jun/lerobot
    
    echo "运行V5选择（不训练，不eval）..."
    CUDA_VISIBLE_DEVICES=$GPU_ID python personal/work2/duibi/train_and_eval_scripts/run_v5_selection_only.py \
        --dataset-name "$DATASET_NAME" \
        --dataset-root "${DATASET_ROOT}/${DATASET_NAME}" \
        --num-episodes $EPISODES_PER_DATASET \
        --seed $SEED \
        --gpu-id $GPU_ID \
        --output-dir "$V5_OUTPUT_BASE"
    
    # V5选择生成的subset文件位置
    V5_SUBSET="${V5_OUTPUT_BASE}/our_v5_${EPISODES_PER_DATASET}_seed${SEED}/subsets/our_v5_${EPISODES_PER_DATASET}_seed${SEED}.json"
    
    # 创建目标目录并复制文件
    mkdir -p "$(dirname "$MERGE_EXPECTED")"
    if [ -f "$V5_SUBSET" ]; then
        cp "$V5_SUBSET" "$MERGE_EXPECTED"
        echo "✓ 已复制subset文件到: $MERGE_EXPECTED"
    else
        echo "✗ 错误: V5选择未生成subset文件: $V5_SUBSET"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "所有V5选择完成！"
echo "========================================"
echo ""
echo "现在可以运行merge脚本:"
echo "cd /data/zhonglinye/jun/lerobot && python personal/work2/duibi/train_and_eval_scripts/merge_selected_episodes.py \\"
echo "    --dataset-root personal/work2/dataset_view \\"
echo "    --dataset-names disassemble-v3_corner pick_place-v3_corner coffee-button-v3_corner \\"
echo "    --episodes-per-dataset 112 \\"
echo "    --selection-mode ours_v5 \\"
echo "    --seed 42 \\"
echo "    --v5-output-dir personal/work2/duibi/our_v5_multi_336_seed42/subsets \\"
echo "    --output-dir personal/work2/dataset_view/merged_3tasks_v5_336_seed42"