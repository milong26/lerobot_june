#!/bin/bash
# DifferentVLM Experiment Launcher
# Usage: bash run_tmux.sh --vlm <vlm_name> --gpu <gpu_id> [--camera <camera>]
# Example:
#   bash run_tmux.sh --vlm llava_pythia400m --gpu 0 --camera corner
#   bash run_tmux.sh --vlm prismatic_qwen25_05b --gpu 1 --camera corner

set -e

VLM_NAME=""
GPU_ID=0
CAMERA="corner"

while [[ $# -gt 0 ]]; do
    case $1 in
        --vlm)
            VLM_NAME="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --camera)
            CAMERA="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$VLM_NAME" ]; then
    echo "Error: --vlm is required"
    echo "Usage: bash run_tmux.sh --vlm <vlm_name> --gpu <gpu_id> [--camera <camera>]"
    echo "Available VLMs: llava_pythia400m, prismatic_qwen25_05b"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK2_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$WORK2_ROOT/../../../.." && pwd)"

LOG_DIR="$WORK2_ROOT/differentvlm/logs/${VLM_NAME}_${CAMERA}"
mkdir -p "$LOG_DIR"

TMUX_SESSION="diffvlm_${VLM_NAME}_${CAMERA}_gpu${GPU_ID}"

tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

RUNNER_SCRIPT="$LOG_DIR/${VLM_NAME}_${CAMERA}_runner.sh"
cat > "$RUNNER_SCRIPT" << RUNNER_EOF
#!/bin/bash

eval "\$(conda shell.bash hook 2>/dev/null)"
conda activate lb_server

export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

LOG_FILE="$LOG_DIR/${VLM_NAME}_${CAMERA}.log"
TIME_FILE="$LOG_DIR/${VLM_NAME}_${CAMERA}.time"
PID_FILE="$LOG_DIR/${VLM_NAME}_${CAMERA}.pid"

mkdir -p "$LOG_DIR"

> "\$LOG_FILE"

exec > >(tee -a "\$LOG_FILE") 2>&1

echo \$\$ > "\$PID_FILE"
echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" > "\$TIME_FILE"
echo "PID: \$\$" >> "\$TIME_FILE"

echo "========================================"
echo "DifferentVLM Experiment"
echo "VLM: ${VLM_NAME}"
echo "GPU: ${GPU_ID}"
echo "Camera: ${CAMERA}"
echo "PID: \$\$"
echo "Started: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

python "$SCRIPT_DIR/run_different_vlm.py" \\
    --vlm ${VLM_NAME} \\
    --gpu ${GPU_ID} \\
    --camera ${CAMERA}

EXIT_CODE=\$?

echo "" >> "\$TIME_FILE"
echo "End time: \$(date '+%Y-%m-%d %H:%M:%S')" >> "\$TIME_FILE"
if [ \$EXIT_CODE -eq 0 ]; then
    echo "Status: completed successfully" >> "\$TIME_FILE"
else
    echo "Status: failed with exit code \$EXIT_CODE" >> "\$TIME_FILE"
fi

echo ""
echo "========================================"
echo "Experiment completed: ${VLM_NAME}"
echo "Exit code: \$EXIT_CODE"
echo "Finished: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

exec bash
RUNNER_EOF

chmod +x "$RUNNER_SCRIPT"

tmux new-session -d -s "$TMUX_SESSION" "bash $RUNNER_SCRIPT"

echo "Launched DifferentVLM experiment: $VLM_NAME"
echo "tmux session: $TMUX_SESSION"
echo "GPU: $GPU_ID"
echo "Camera: $CAMERA"
echo "Log dir: $LOG_DIR"
echo ""
echo "Monitor with: tmux attach -t $TMUX_SESSION"
echo "Check logs: tail -f $LOG_DIR/${VLM_NAME}_${CAMERA}.log"