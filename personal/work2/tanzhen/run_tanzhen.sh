#!/bin/bash
# 探针实验一键运行脚本
# 用法: ./run_tanzhen.sh [--smoke-test] [--final] [--gpu <gpu_id>]

set -e

# TODO: 指定gpu id，默认用0
# 优先级：命令行参数 > 环境变量 > 默认值0
GPU_ID=${GPU_ID:-1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$SCRIPT_DIR/configs/probe_config.yaml"

# 解析参数
SMOKE_TEST=false
FINAL_TEST=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-test)
            SMOKE_TEST=true
            shift
            ;;
        --final)
            FINAL_TEST=true
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 设置CUDA_VISIBLE_DEVICES，使代码中的device_id直接对应该GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "========================================="
echo "  探针实验 (Tanzhen Probe Experiment)"
echo "========================================="
echo ""
echo "使用GPU ID: $GPU_ID (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo ""

# Step 1: 运行探针rollout
echo "Step 1: 运行探针rollout..."
if $SMOKE_TEST; then
    echo "  [冒烟测试模式]"
    python "$SCRIPT_DIR/probe/run_probe_rollout.py" \
        --config "$CONFIG_FILE" \
        --smoke-test
elif $FINAL_TEST; then
    echo "  [正式测试模式]"
    python "$SCRIPT_DIR/probe/run_probe_rollout.py" \
        --config "$CONFIG_FILE"
else
    echo "  请指定 --smoke-test 或 --final"
    exit 1
fi

echo ""

# Step 2: 计算覆盖空隙
echo "Step 2: 计算覆盖空隙..."

python "$SCRIPT_DIR/probe/coverage_gap.py" \
    --subset-file "$PROJECT_ROOT/personal/work2/duibi/random_42/subsets/random_112_seed42.json" \
    --dataset-metadata "$PROJECT_ROOT/personal/work2/dataset_view/pickplacev3/episode_initial_states.json" \
    --output "$SCRIPT_DIR/results/probe_raw/random_42_coverage_gaps.json"

python "$SCRIPT_DIR/probe/coverage_gap.py" \
    --subset-file "$PROJECT_ROOT/personal/work2/duibi/uniform_42/subsets/uniform_112_seed42.json" \
    --dataset-metadata "$PROJECT_ROOT/personal/work2/dataset_view/pickplacev3/episode_initial_states.json" \
    --output "$SCRIPT_DIR/results/probe_raw/uniform_42_coverage_gaps.json"

echo ""

# Step 3: 相关性分析与报告生成
echo "Step 3: 生成分析报告..."
python "$SCRIPT_DIR/analysis/correlate_and_report.py" \
    --config "$CONFIG_FILE"

echo ""
echo "========================================="
echo "  实验完成！"
echo "========================================="
echo ""
echo "查看报告: $SCRIPT_DIR/results/REPORT.md"
echo "查看权重: $SCRIPT_DIR/results/weakness_scores.json"
echo "查看图表: $SCRIPT_DIR/results/figures/"