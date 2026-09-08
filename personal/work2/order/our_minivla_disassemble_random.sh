#!/bin/bash

# 等待5小时（18000秒）
echo "[$(date)] 开始等待5小时..."
sleep 18000

echo "[$(date)] 等待结束，开始监控GPU3显存使用情况..."

# 循环检查GPU3显存
while true; do
    # 获取GPU3的memory usage (MiB)
    GPU3_MEMORY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | awk '{print $1}')
    
    echo "[$(date)] GPU3当前显存使用: ${GPU3_MEMORY}MiB"
    
    # 判断是否小于5000MiB
    if [ "$GPU3_MEMORY" -lt 5000 ] 2>/dev/null; then
        echo "[$(date)] GPU3显存使用(${GPU3_MEMORY}MiB)已小于5000MiB，开始执行任务..."
        
        # 执行任务
        cd /data/zhonglinye/jun/lerobot
        conda activate lb_server
        bash personal/work2/differentvlm/minivla/scripts/run_tmux_minivla.sh --gpu 3 --dataset disassemble-v3_corner --selection-mode random
        
        echo "[$(date)] 任务执行完毕，停止监控。"
        break
    fi
    
    echo "[$(date)] GPU3显存仍大于等于5000MiB，10分钟后再次检查..."
    sleep 600  # 等待10分钟（600秒）
done

echo "[$(date)] 监控脚本执行完成。"