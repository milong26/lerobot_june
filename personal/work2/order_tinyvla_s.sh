#!/bin/bash
# 后台启动GPU监控脚本的便捷工具（TinyVLA-S任务）
# 使用方法: bash order_tinyvla_s.sh

WORK_DIR="/data/zhonglinye/jun/lerobot"
MONITOR_SCRIPT="$WORK_DIR/personal/work2/monitor_gpu_and_launch_tinyvla_s.sh"
LOG_DIR="$WORK_DIR/personal/work2/gpu_monitor_logs"
ORDER_LOG="$LOG_DIR/order_tinyvla_s_$(date '+%Y%m%d_%H%M%S').log"

mkdir -p "$LOG_DIR"

echo "========================================="
echo "GPU监控脚本 - order_tinyvla_s"
echo "========================================="
echo ""

# 检查监控脚本是否存在
if [ ! -f "$MONITOR_SCRIPT" ]; then
    echo "错误: 找不到监控脚本 $MONITOR_SCRIPT"
    exit 1
fi

echo "使用 nohup 后台启动..."
echo "日志文件: $ORDER_LOG"
echo ""

cd "$WORK_DIR"
nohup bash "$MONITOR_SCRIPT" >> "$ORDER_LOG" 2>&1 &
PID=$!

echo "✓ 监控脚本已在后台启动"
echo "  进程ID: $PID"
echo "  日志文件: $ORDER_LOG"
echo ""
echo "常用命令:"
echo "  查看日志: tail -f $ORDER_LOG"
echo "  查看进程: ps -p $PID"
echo "  停止进程: kill $PID"
echo ""
echo "========================================="
echo "启动完成！即使断开SSH连接，脚本也会继续运行"
echo "========================================="