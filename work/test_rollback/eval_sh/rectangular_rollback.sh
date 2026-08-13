#!/bin/bash

# 运行方式：
# 1. 先赋予执行权限（仅首次需要）：
#    chmod +x /home/qwe/jun/lerobot/work/test_rollback/eval_sh/rectangular_rollback.sh
#
# 2. 直接运行：
#    ./rectangular_rollback.sh
#    或者：
#    bash /home/qwe/jun/lerobot/work/test_rollback/eval_sh/rectangular_rollback.sh
#
# 3. 如果需要修改参数，直接编辑本脚本中的对应行即可
# ============================================================

# 切换到脚本所在目录
cd /home/qwe/jun/lerobot/work/test_rollback

# 运行主控制程序（带回滚重试机制 + 力传感器记录 + state 对比绘图）
/home/qwe/anaconda3/envs/lb_local/bin/python main_controller.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=start_new_heihei_2 \
    --robot.cameras="{wrist: {type: opencv, index_or_path: 6, width: 640, height: 480, fps: 30, fourcc: MJPG},top: {type: intelrealsense, serial_number_or_name: 806312060427, width: 640, height: 480, fps: 30, use_depth: False}}" \
    --task="Grab the rectangular object" \
    --server_url=ws://10.10.16.19:9001 \
    --fps=30 \
    --wowskin.enabled=True \
    --wowskin.port=/dev/ttyACM0 \
    --num_loop=2 \
    --action_steps=35 \
    --rollback_enabled=True \
    --force_ratio_threshold=10.0 \
    --max_consecutive_failures=3 \
    --max_rollback_count=10 \
    --reset_wait_time=2.0 \
    --use_force_check=False \
    --use_state_check=True \
    --record_force=True \
    --record_force_save_dir=./force_comparison \
    --record_gripper=True \
    --record_gripper_save_dir=./gripper_comparison
