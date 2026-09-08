#!/bin/bash

# 等待1小时（3600秒）
echo "[$(date)] 开始等待1小时..."
sleep 3600

echo "[$(date)] 等待结束，开始监控GPU0显存使用情况..."

# 循环检查GPU0显存
while true; do
    # 获取GPU0的memory usage (MiB)
    GPU0_MEMORY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | awk '{print $1}')
    
    echo "[$(date)] GPU0当前显存使用: ${GPU0_MEMORY}MiB"
    
    # 判断是否小于5000MiB
    if [ "$GPU0_MEMORY" -lt 5000 ] 2>/dev/null; then
        echo "[$(date)] GPU0显存使用(${GPU0_MEMORY}MiB)已小于5000MiB，开始执行任务..."
        
        # 执行任务
        cd /data/zhonglinye/jun/lerobot
        conda activate lb_server
        bash personal/work2/differentvlm/scripts/run_tinyvla.sh tinyvla-b --dataset disassemble-v3_corner --gpu 0 --num_episodes 300 --selection-mode random
        
        echo "[$(date)] 任务执行完毕，停止监控。"
        break
    fi
    
    echo "[$(date)] GPU0显存仍大于等于5000MiB，10分钟后再次检查..."
    sleep 600  # 等待10分钟（600秒）
done

echo "[$(date)] 监控脚本执行完成。"