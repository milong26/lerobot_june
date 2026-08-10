#!/bin/bash

# 批量实验脚本：运行 num_loop 从 2 到 10 的实验
# 每次运行的日志和结果保存到独立文件夹

# 基础配置
BASE_CMD="python replay_dataset_to_server.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=start_new_heihei_2 \
    --task=\"Grab the rectangular object\" \
    --server_url=ws://10.10.16.19:9001 \
    --fps=30 \
    --action_steps=35"

# 结果保存根目录
OUTPUT_ROOT="./batch_results"
mkdir -p "$OUTPUT_ROOT"

echo "========================================="
echo "批量实验开始"
echo "num_loop 范围: 2-10"
echo "结果保存目录: $OUTPUT_ROOT"
echo "========================================="

# 循环 num_loop 从 2 到 10
for NUM_LOOP in $(seq 2 2); do
    # 创建本次实验的输出文件夹
    EXPERIMENT_DIR="${OUTPUT_ROOT}/num_loop_${NUM_LOOP}"
    mkdir -p "$EXPERIMENT_DIR"
    
    echo ""
    echo "========================================="
    echo "开始实验: num_loop=${NUM_LOOP}"
    echo "输出目录: ${EXPERIMENT_DIR}"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================="
    
    # 使用 script 命令记录日志
    # script 会捕获所有终端输出（包括 stdout 和 stderr）
    script -q -c "$BASE_CMD --num_loop=${NUM_LOOP}" "${EXPERIMENT_DIR}/experiment.log"
    
    # 记录实验结束时间
    END_TIME=$(date '+%Y-%m-%d %H:%M:%S')
    echo "实验结束时间: ${END_TIME}" >> "${EXPERIMENT_DIR}/experiment.log"
    
    # 复制生成的图表到实验文件夹（如果有的话）
    if [ -d "./debug_output" ]; then
        # 获取实验开始前后的图表文件
        cp ./debug_output/*.png "${EXPERIMENT_DIR}/" 2>/dev/null || true
    fi
    
    echo "✅ 实验 num_loop=${NUM_LOOP} 完成"
    echo "   日志: ${EXPERIMENT_DIR}/experiment.log"
    echo "   图表: ${EXPERIMENT_DIR}/*.png"
    
    # 可选：在每次实验之间添加短暂暂停
    sleep 2
done

echo ""
echo "========================================="
echo "所有实验完成！"
echo "结果保存在: ${OUTPUT_ROOT}"
echo "========================================="

# 显示结果统计
echo ""
echo "实验结果统计:"
for NUM_LOOP in $(seq 2 10); do
    EXPERIMENT_DIR="${OUTPUT_ROOT}/num_loop_${NUM_LOOP}"
    if [ -d "$EXPERIMENT_DIR" ]; then
        LOG_SIZE=$(wc -l < "${EXPERIMENT_DIR}/experiment.log" 2>/dev/null || echo "0")
        PNG_COUNT=$(ls "${EXPERIMENT_DIR}"/*.png 2>/dev/null | wc -l || echo "0")
        echo "  num_loop=${NUM_LOOP}: 日志 ${LOG_SIZE} 行, 图表 ${PNG_COUNT} 个"
    fi
done