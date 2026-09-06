#!/bin/bash
# GPU监控脚本：4小时后开始检查GPU 0显存，满足条件后自动启动训练任务
# 使用方法: bash monitor_gpu_and_launch.sh

set -e

# 配置参数
INITIAL_WAIT_HOURS=4          # 初始等待时间（小时）
CHECK_INTERVAL_MINUTES=20     # 检查间隔（分钟）
GPU_TARGET=0                  # 目标GPU ID
MEMORY_THRESHOLD=5000         # 显存阈值（MiB）
LAUNCH_SCRIPT="personal/work2/duibi/train_and_eval_scripts/launch_deminf.sh"
LAUNCH_ARGS="0 disassemble-v3_corner"
WORK_DIR="/data/zhonglinye/jun/lerobot"

# 日志配置
LOG_DIR="/data/zhonglinye/jun/lerobot/personal/work2/gpu_monitor_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="$LOG_DIR/monitor_${TIMESTAMP}.log"

# 日志函数
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

# 获取GPU显存使用量（MiB）
get_gpu_memory_usage() {
    local gpu_id=$1
    # 使用nvidia-smi查询指定GPU的显存使用量，返回数字（MiB）
    local memory_used
    memory_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id 2>/dev/null | tr -d '[:space:]')
    
    if [ -z "$memory_used" ]; then
        echo "-1"
        return 1
    fi
    
    echo "$memory_used"
    return 0
}

# 记录当前GPU状态
log_gpu_status() {
    log "=== 当前GPU状态 ==="
    nvidia-smi 2>&1 | tee -a "$LOG_FILE"
    log "==================="
}

# 启动训练任务
launch_training() {
    log "========================================="
    log "满足条件！开始启动训练任务..."
    log "GPU ${GPU_TARGET} 显存使用: ${1}MiB < ${MEMORY_THRESHOLD}MiB"
    log "执行命令: bash ${LAUNCH_SCRIPT} ${LAUNCH_ARGS}"
    log "========================================="
    
    cd "$WORK_DIR"
    
    # 执行启动脚本
    bash "$WORK_DIR/$LAUNCH_SCRIPT" $LAUNCH_ARGS 2>&1 | tee -a "$LOG_FILE"
    
    local exit_code=$?
    
    log "========================================="
    if [ $exit_code -eq 0 ]; then
        log "训练任务已成功启动！"
    else
        log "训练任务启动失败，退出码: $exit_code"
    fi
    log "========================================="
    
    return $exit_code
}

# 主流程
main() {
    log "========================================="
    log "GPU监控脚本启动"
    log "========================================="
    log "配置信息:"
    log "  - 初始等待时间: ${INITIAL_WAIT_HOURS} 小时"
    log "  - 检查间隔: ${CHECK_INTERVAL_MINUTES} 分钟"
    log "  - 目标GPU: ${GPU_TARGET}"
    log "  - 显存阈值: ${MEMORY_THRESHOLD}MiB"
    log "  - 日志文件: ${LOG_FILE}"
    log "========================================="
    
    # 记录初始GPU状态
    log_gpu_status
    
    # 计算等待时间（秒）
    local wait_seconds=$((INITIAL_WAIT_HOURS * 3600))
    local wait_end_time=$(date -d "+${INITIAL_WAIT_HOURS} hours" '+%Y-%m-%d %H:%M:%S')
    
    log "开始等待 ${INITIAL_WAIT_HOURS} 小时..."
    log "预计开始检查时间: ${wait_end_time}"
    log "等待期间脚本将持续运行，请勿关闭终端"
    log ""
    
    # 显示倒计时（每分钟更新一次）
    local remaining=$wait_seconds
    while [ $remaining -gt 0 ]; do
        local hours=$((remaining / 3600))
        local minutes=$(( (remaining % 3600) / 60 ))
        local seconds=$((remaining % 60))
        
        # 每5分钟记录一次日志（避免日志过多）
        if [ $((remaining % 300)) -eq 0 ] || [ $remaining -le 60 ]; then
            log "倒计时: ${hours}小时 ${minutes}分钟 ${seconds}秒"
        fi
        
        sleep 60
        remaining=$((remaining - 60))
    done
    
    log ""
    log "========================================="
    log "等待结束！开始检查GPU状态..."
    log "========================================="
    
    # 开始循环检查
    local check_count=0
    while true; do
        check_count=$((check_count + 1))
        log ""
        log "--- 第 ${check_count} 次检查 [$(date '+%Y-%m-%d %H:%M:%S')] ---"
        
        # 获取GPU显存使用量
        local memory_used
        memory_used=$(get_gpu_memory_usage $GPU_TARGET)
        
        if [ "$memory_used" = "-1" ]; then
            log "警告: 无法获取GPU ${GPU_TARGET} 的显存信息，将在 ${CHECK_INTERVAL_MINUTES} 分钟后重试"
            log "等待 ${CHECK_INTERVAL_MINUTES} 分钟后再次检查..."
            sleep $((CHECK_INTERVAL_MINUTES * 60))
            continue
        fi
        
        log "GPU ${GPU_TARGET} 当前显存使用: ${memory_used}MiB"
        log "阈值: ${MEMORY_THRESHOLD}MiB"
        
        # 记录完整GPU状态
        log_gpu_status
        
        # 检查是否满足条件
        if [ "$memory_used" -lt "$MEMORY_THRESHOLD" ]; then
            log ">>> 条件满足！显存使用 ${memory_used}MiB < ${MEMORY_THRESHOLD}MiB"
            
            # 启动训练任务
            launch_training $memory_used
            
            log ""
            log "========================================="
            log "监控任务完成！"
            log "训练任务已启动，监控脚本退出"
            log "完整日志保存在: ${LOG_FILE}"
            log "========================================="
            
            exit 0
        else
            log ">>> 条件不满足，显存使用 ${memory_used}MiB >= ${MEMORY_THRESHOLD}MiB"
            log "将在 ${CHECK_INTERVAL_MINUTES} 分钟后再次检查..."
        fi
        
        # 等待后继续检查
        sleep $((CHECK_INTERVAL_MINUTES * 60))
    done
}

# 捕获信号，优雅退出
cleanup() {
    log ""
    log "========================================="
    log "收到中断信号，监控脚本退出"
    log "日志已保存到: ${LOG_FILE}"
    log "========================================="
    exit 1
}

trap cleanup SIGINT SIGTERM

# 执行主函数
main "$@"