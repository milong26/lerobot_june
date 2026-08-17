#!/bin/bash

# 运行方式：
# 1. 先赋予执行权限（仅首次需要）：
#    chmod +x /home/qwe/jun/lerobot/work/test_rollback/record_sh/ring.sh
#
# 2. 直接运行：
#    ./ring.sh
#    或者：
#    bash /home/qwe/jun/lerobot/work/test_rollback/record_sh/ring.sh
#
# 3. 如果需要修改参数，直接编辑本脚本中的对应行即可
# ============================================================

# 切换到脚本所在目录
cd /home/qwe/jun/lerobot/work/test_rollback

# 运行 lerobot-record 录制命令
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30},top: {type: intelrealsense, serial_number_or_name: 806312060427, width: 640, height: 480, fps: 30, use_depth: False}}" \
    --robot.id=start_new_heihei_2 \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM2 \
    --teleop.id=start_new_leader_heihei \
    --dataset.repo_id=ep10/ring \
    --dataset.num_episodes=10 \
    --dataset.single_task="Grab the circular ring part" \
    --dataset.streaming_encoding=true \
    --dataset.root=/home/qwe/.cache/huggingface/lerobot/ep10/ring \
    --wowskin.enabled=true \
    --wowskin.port=/dev/ttyACM0 \
    --dataset.encoder_threads=2 \
    --display_data=false \
    --resume=true
