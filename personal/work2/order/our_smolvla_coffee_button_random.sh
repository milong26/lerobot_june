#!/bin/bash

# 等待5小时（18000秒）
echo "[$(date)] 开始等待5小时..."
sleep 18000

echo "[$(date)] 等待结束，开始监控GPU1显存使用情况..."

# 循环检查GPU1显存
while true; do
    # 获取GPU1的memory usage (MiB)
    GPU1_MEMORY=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | awk '{print $1}')
    
    echo "[$(date)] GPU1当前显存使用: ${GPU1_MEMORY}MiB"
    
    # 判断是否小于5000MiB
    if [ "$GPU1_MEMORY" -lt 5000 ] 2>/dev/null; then
        echo "[$(date)] GPU1显存使用(${GPU1_MEMORY}MiB)已小于5000MiB，开始执行任务..."
        
        # 执行任务
        cd /data/zhonglinye/jun/lerobot
        conda activate lb_server
        bash personal/work2/duibi/train_and_eval_scripts/train_and_eval.sh random 50 42 1 "" coffee-button-v3_corner
        
        echo "[$(date)] 任务执行完毕，停止监控。"
        break
    fi
    
    echo "[$(date)] GPU1显存仍大于等于5000MiB，10分钟后再次检查..."
    sleep 600  # 等待10分钟（600秒）
done

echo "[$(date)] 监控脚本执行完成。"