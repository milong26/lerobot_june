#!/bin/bash
# Auto-launch grid_uniform training when GPU 1 memory usage is below 5000 MiB
# Check interval: 20 minutes

GPU_ID=1
MEM_THRESHOLD=4000
CHECK_INTERVAL=1200  # 20 minutes in seconds

LOG_DIR="/data/zhonglinye/jun/lerobot/personal/work2/order/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/auto_launch_grid_uniform.log"

echo "========================================" | tee -a "$LOG_FILE"
echo "Auto-launch monitor started: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "GPU ID: $GPU_ID" | tee -a "$LOG_FILE"
echo "Memory threshold: ${MEM_THRESHOLD} MiB" | tee -a "$LOG_FILE"
echo "Check interval: ${CHECK_INTERVAL}s (20 minutes)" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

while true; do
    # Get GPU memory usage in MiB
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID 2>/dev/null | tr -d ' ')
    
    if [ -z "$GPU_MEM" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: Failed to get GPU $GPU_ID memory usage" | tee -a "$LOG_FILE"
        sleep $CHECK_INTERVAL
        continue
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $GPU_ID memory usage: ${GPU_MEM} MiB (threshold: ${MEM_THRESHOLD} MiB)" | tee -a "$LOG_FILE"
    
    if [ "$GPU_MEM" -lt "$MEM_THRESHOLD" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $GPU_ID memory is low (${GPU_MEM} MiB < ${MEM_THRESHOLD} MiB), launching training..." | tee -a "$LOG_FILE"
        
        cd /data/zhonglinye/jun/lerobot
        
        # Initialize conda properly for non-interactive shell
        eval "$(conda shell.bash hook 2>/dev/null)"
        conda activate lb_server
        
        bash personal/work2/duibi/train_and_eval_scripts/launch_grid_uniform.sh 1 pick_place-v3_corner 2>&1 | tee -a "$LOG_FILE"
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training launched, exiting monitor." | tee -a "$LOG_FILE"
        exit 0
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $GPU_ID memory is still high (${GPU_MEM} MiB >= ${MEM_THRESHOLD} MiB), waiting..." | tee -a "$LOG_FILE"
    sleep $CHECK_INTERVAL
done