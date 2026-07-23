#!/bin/bash
# =============================================================================
# SmolVLA 多GPU训练脚本
# =============================================================================
# 使用说明:
#   chmod +x train_smolvla_multigpu.sh
#   ./train_smolvla_multigpu.sh
#
# 安全措施:
#   1. LD_LIBRARY_PATH 确保子进程能找到 FFmpeg 库
#   2. CUDA_VISIBLE_DEVICES 限制使用 GPU 0,2,3（避开有问题的1和4）
#   3. HF_HUB_OFFLINE=1 避免网络请求卡住
#   4. 训练日志实时输出到 train_log/test_smolvla_ep10.log
#   5. 使用 accelerate 配置文件，避免临时文件散落在系统目录
# =============================================================================

# 安全设置：确保 conda 环境的动态库优先被加载
# 这解决了 torchcodec 找不到 FFmpeg 库的问题
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 训练命令
nohup env \
  CUDA_VISIBLE_DEVICES=0,2,3 \
  HF_HUB_OFFLINE=1 \
accelerate launch \
  --config_file=accelerate_tmp/accelerate_config.yaml \
  $(which lerobot-train) \
  --dataset.repo_id=ep10/all \
  --dataset.root=personal/work1/ep10_all \
  --policy.path=lerobot/smolvla_base \
  --output_dir=outputs/train/smolvla_ep10_multigpu \
  --job_name=smolvla_ep10 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --steps=1000 \
  --batch_size=32 \
  --rename_map='{"observation.images.side":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  > train_log/test_smolvla_ep10.log 2>&1 &

# 输出进程ID，方便后续管理
echo "训练进程已启动，PID: $!"
echo "日志文件: train_log/test_smolvla_ep10.log"
echo ""
echo "监控命令:"
echo "  tail -f train_log/test_smolvla_ep10.log"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "停止训练:"
echo "  kill $!"