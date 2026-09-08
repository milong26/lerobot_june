#!/bin/bash
# Auto-launch DemInf experiment when GPU 1 is free
# Monitors GPU memory every 20 minutes
# Triggers when GPU memory < 5000 MiB

set -e

# Configuration
GPU_ID=1
MEMORY_THRESHOLD=4000
CHECK_INTERVAL=1200  # 20 minutes in seconds

# Log directory
LOG_DIR="/data/zhonglinye/jun/lerobot/personal/work2/order/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/auto_launch_deminf.log"

# Flag to track if training has been launched
LAUNCHED=false

echo "========================================" | tee -a "$LOG_FILE"
echo "Auto-launch DemInf Monitor Started" | tee -a "$LOG_FILE"
echo "GPU ID: $GPU_ID" | tee -a "$LOG_FILE"
echo "Memory Threshold: ${MEMORY_THRESHOLD} MiB" | tee -a "$LOG_FILE"
echo "Check Interval: ${CHECK_INTERVAL}s (20 minutes)" | tee -a "$LOG_FILE"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

while true; do
    # Get GPU memory usage
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID 2>/dev/null | tr -d '[:space:]')
    
    if [ -z "$GPU_MEM" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: Failed to read GPU $GPU_ID memory, retrying in 20 minutes..." | tee -a "$LOG_FILE"
        sleep $CHECK_INTERVAL
        continue
    fi
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $GPU_ID Memory: ${GPU_MEM} MiB (threshold: ${MEMORY_THRESHOLD} MiB)" | tee -a "$LOG_FILE"
    
    if [ "$GPU_MEM" -lt "$MEMORY_THRESHOLD" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $GPU_ID is free (${GPU_MEM} MiB < ${MEMORY_THRESHOLD} MiB)" | tee -a "$LOG_FILE"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching DemInf experiment..." | tee -a "$LOG_FILE"
        
        cd /data/zhonglinye/jun/lerobot
        
        # Initialize conda properly for non-interactive shell
        eval "$(conda shell.bash hook 2>/dev/null)"
        conda activate lb_server
        
        # Launch DemInf with 10 episodes
        bash personal/work2/duibi/train_and_eval_scripts/launch_deminf.sh 1 10 coffee-button-v3_corner 2>&1 | tee -a "$LOG_FILE"
        
        # Launch order_grid_uniform.sh
        bash personal/work2/order_grid_uniform.sh 2>&1 | tee -a "$LOG_FILE"
        
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] All experiments launched successfully. Exiting monitor." | tee -a "$LOG_FILE"
        exit 0
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU $GPU_ID is busy (${GPU_MEM} MiB >= ${MEMORY_THRESHOLD} MiB), waiting..." | tee -a "$LOG_FILE"
    fi
    
    sleep $CHECK_INTERVAL
done