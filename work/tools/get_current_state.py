"""
获取机械臂当前状态
用途：手动将机械臂调整到某个姿态（如夹爪完全张开），运行此脚本记录当前 state 值
"""

import sys
import os
import time
import numpy as np

# 添加 lerobot 项目路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig  # noqa: F401


def main():
    # 配置机器人（不连接相机，仅获取关节状态）
    robot_config = SOFollowerRobotConfig(
        port="/dev/ttyACM1",
        id="start_new_heihei_2",
        cameras={},  # 不连接相机
    )

    print("[INFO] 正在连接机器人...")
    robot = make_robot_from_config(robot_config)
    robot.connect()
    print("[INFO] 机器人已连接")

    # 等待稳定
    time.sleep(0.5)

    # 获取当前状态
    obs = robot.get_observation()

    # 提取关节状态
    motor_keys = [
        'shoulder_pan.pos',
        'shoulder_lift.pos',
        'elbow_flex.pos',
        'wrist_flex.pos',
        'wrist_roll.pos',
        'gripper.pos',
    ]
    state = np.array([obs[k] for k in motor_keys], dtype=np.float32)

    print("\n" + "=" * 60)
    print("当前机械臂状态 (state):")
    print("=" * 60)
    for i, key in enumerate(motor_keys):
        print(f"  [{i}] {key:25s} = {state[i]:.6f}")
    print("=" * 60)
    print(f"\n完整 state 数组 (可直接复制使用):")
    print(f"  {state.tolist()}")
    print(f"\n夹爪值 (gripper.pos): {state[5]:.6f}")
    print("=" * 60)

    # 断开
    robot.disconnect()
    print("\n[INFO] 机器人已断开")


if __name__ == "__main__":
    main()