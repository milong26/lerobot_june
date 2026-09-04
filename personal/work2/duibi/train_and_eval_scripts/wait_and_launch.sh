#!/bin/bash
# Wait 4 hours, then check GPU memory every 30 minutes
# If GPU 0 memory < 5000MiB, launch the experiment

set -e

WAIT_HOURS=4
CHECK_INTERVAL_MINUTES=30
MAX_MEMORY_MB=5000
GPU_ID=0

echo "========================================"
echo "Waiting ${WAIT_HOURS} hours before starting checks..."
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Will check GPU${GPU_ID} memory every ${CHECK_INTERVAL_MINUTES} minutes"
echo "Launch when memory < ${MAX_MEMORY_MB}MiB"
echo "========================================"

# Wait 4 hours
sleep $((WAIT_HOURS * 3600))

echo ""
echo "========================================"
echo "Starting GPU memory checks..."
echo "Check start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

while true; do
    # Get GPU memory usage for GPU 0
    MEMORY_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU_ID | tr -d ' ')
    
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU${GPU_ID} memory used: ${MEMORY_USED}MiB / ${MAX_MEMORY_MB}MiB threshold"
    
    if [ "$MEMORY_USED" -lt "$MAX_MEMORY_MB" ]; then
        echo ""
        echo "========================================"
        echo "GPU${GPU_ID} is available! Memory: ${MEMORY_USED}MiB < ${MAX_MEMORY_MB}MiB"
        echo "Launching experiment..."
        echo "========================================"
        
        cd /data/zhonglinye/jun/lerobot
        conda activate lb_server
        bash personal/work2/duibi/train_and_eval_scripts/launch_new_uniform.sh $GPU_ID disassemble-v3_corner
        
        echo ""
        echo "Experiment launched successfully!"
        echo "Launch time: $(date '+%Y-%m-%d %H:%M:%S')"
        exit 0
    else
        echo "GPU${GPU_ID} still in use. Will check again in ${CHECK_INTERVAL_MINUTES} minutes..."
        sleep $((CHECK_INTERVAL_MINUTES * 60))
    fi
done