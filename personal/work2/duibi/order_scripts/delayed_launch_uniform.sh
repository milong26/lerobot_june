#!/bin/bash
# 延迟执行任务：5小时后检查GPU1状态，如果空闲则启动uniform训练

DELAY_HOURS=5
GPU_ID=1

echo "=== 延迟执行任务 ==="
echo "等待 ${DELAY_HOURS} 小时后检查 GPU${GPU_ID} 状态"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "预计执行时间: $(date -d "+${DELAY_HOURS} hours" '+%Y-%m-%d %H:%M:%S')"

# 等待5小时
sleep ${DELAY_HOURS}h

echo ""
echo "=== 检查 GPU${GPU_ID} 状态 ==="
echo "检查时间: $(date '+%Y-%m-%d %H:%M:%S')"

# 检查GPU是否有运行进程
GPU_PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $GPU_ID 2>/dev/null | grep -v "No running" | wc -l)

echo "GPU${GPU_ID} 运行进程数: ${GPU_PROCS}"

# 判断GPU是否空闲（无进程）
if [ "$GPU_PROCS" -eq 0 ]; then
    echo ""
    echo "=== GPU${GPU_ID} 空闲，启动训练 ==="
    
    cd /data/zhonglinye/jun/lerobot/personal/work2/duibi/train_and_eval_scripts/
    eval "$(conda shell.bash hook)"
    conda activate lb_server
    bash launch_uniform.sh 1
    
    echo ""
    echo "=== 训练已启动 ==="
else
    echo ""
    echo "=== GPU${GPU_ID} 正忙，取消执行 ==="
    echo "运行进程数: ${GPU_PROCS}"
fi