#!/bin/bash

# 运行方式：
# 1. 先赋予执行权限（仅首次需要）：
#    chmod +x /home/qwe/jun/lerobot/work/test_evo1_ep10_half/eval_sh/ushape.sh
#
# 2. 直接运行：
#    ./ushape.sh
#    或者：
#    bash /home/qwe/jun/lerobot/work/test_evo1_ep10_half/eval_sh/ushape.sh
#
# 3. 如果需要修改参数，直接编辑本脚本中的对应行即可
# ============================================================

# 切换到脚本所在目录
cd /home/qwe/jun/lerobot/work/test_evo1_ep10_half

# 运行主控制程序
/home/qwe/anaconda3/envs/lb_local/bin/python evo1_main_controller.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=start_new_heihei_2 \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: MJPG},top: {type: intelrealsense, serial_number_or_name: 806312060427, width: 640, height: 480, fps: 30, use_depth: False}}" \
    --task="Grab the U-shape object" \
    --server_url=ws://10.10.16.19:8001 \
    --fps=30 \
    --recording.enable_recording=False \
    --recording.record_dir=./debug_recorded_data \
    --recording.save_images=False \
    --wowskin.enabled=True \
    --wowskin.port=/dev/ttyACM0 \
    --action_steps=35
