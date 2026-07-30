#!/bin/bash

# ============================================================
# SmolVLA 测试脚本：valve
# 功能：运行 SmolVLA 模型推理，执行 valve 相关任务
# ============================================================

# 运行方式：
# 1. 先赋予执行权限（仅首次需要）：
#    chmod +x /home/qwe/jun/lerobot/work/test_smolvla_ep10_half/smolvla_eval_sh/valve.sh
#
# 2. 直接运行：
#    ./valve.sh
#    或者：
#    bash /home/qwe/jun/lerobot/work/test_smolvla_ep10_half/smolvla_eval_sh/valve.sh
#
# 3. 如果需要修改参数，直接编辑本脚本中的对应行即可
# ============================================================

# 切换到 lerobot 项目目录
cd /home/qwe/jun/lerobot

# 运行 SmolVLA 推理客户端
/home/qwe/anaconda3/envs/lb_local/bin/python -m lerobot.async_inference.robot_client \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=start_new_heihei_2 \
    --robot.cameras="{ \
        camera2:{type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30}, \
        camera1: {type: intelrealsense, serial_number_or_name: 806312060427, width: 640, height: 480, fps: 30, use_depth: false} \
    }" \
    --task="Grab the cross-shape equipment." \
    --server_address=10.10.16.18:8080 \
    --policy_type=smolvla \
    --pretrained_name_or_path=outputs/smolvla_ep10_half/checkpoints/026000/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0 \
    --half_img_resolu=true
