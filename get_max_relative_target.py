"""
从已经采集的数据中得到max_relative_target
但只能当作一个勉强的解决方案
"""
import numpy as np
import torch
from lerobot.datasets import LeRobotDataset

# 加载你的数据集
dataset = LeRobotDataset(
    repo_id="ep10/ring",
    root="/home/qwe/.cache/huggingface/lerobot/ep10/ring",
)

# 获取所有动作数据
actions = dataset.hf_dataset["action"]

# 电机名称（按顺序对应 action tensor 的 6 个值）
motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

max_diffs = {motor: 0 for motor in motor_names}

for i in range(1, len(actions)):
    # action 是一个 tensor，按顺序包含 6 个关节的角度
    action_curr = actions[i]
    action_prev = actions[i-1]
    
    # 如果是 tensor，转换为 numpy 或者直接用 tensor 操作
    if isinstance(action_curr, torch.Tensor):
        action_curr = action_curr.numpy()
        action_prev = action_prev.numpy()
    
    # 按索引计算每个关节的差值
    for j, motor in enumerate(motor_names):
        diff = abs(action_curr[j] - action_prev[j])
        max_diffs[motor] = max(max_diffs[motor], diff)

print("每个关节的最大单帧变化量：")
for motor, max_diff in max_diffs.items():
    print(f"  {motor}: {max_diff:.2f}")

# 建议设置（留一些余量，比如乘以1.2）
print("\n建议的 max_relative_target 设置：")
suggested = {motor: round(float(diff) * 1.1, 2) for motor, diff in max_diffs.items()}
print(suggested)

# 输出可直接复制到 launch.json 的命令行参数
args_str = ", ".join(f"{motor}: {val}" for motor, val in suggested.items())
print(f'"--robot.max_relative_target={{{args_str}}}",')