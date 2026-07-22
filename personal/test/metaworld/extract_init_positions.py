#!/usr/bin/env python
"""
提取 pick-place-v3 所有 episode 的初始物体位置和目标位置
"""

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import metaworld

REPO_ID = "lerobot/metaworld_mt50"

print("=" * 80)
print("提取 pick-place-v3 所有 episode 的初始物体位置")
print("=" * 80)

# 1. 找到所有 pick-place-v3 的 episode
dataset_meta = LeRobotDatasetMetadata(REPO_ID)

pick_place_episodes = []
for i in range(len(dataset_meta.episodes)):
    ep = dataset_meta.episodes[i]
    # 使用 tasks 字段（文本描述）来匹配
    tasks = ep.get('tasks', [])
    if 'Pick and place a puck to a goal' in str(tasks):
        pick_place_episodes.append(i)

print(f"\n找到 {len(pick_place_episodes)} 个 pick-place-v3 episode")

# 2. 提取每个 episode 的初始物体位置
print("\n提取每个 episode 的初始状态 (第一帧)...")
print(f"{'Episode':<10} {'obj_x':<12} {'obj_y':<12} {'obj_z':<12} {'goal_err_x':<12} {'goal_err_y':<12} {'goal_err_z':<12}")
print("-" * 80)

all_obj_positions = []
all_goal_errors = []

# 提取所有 50 个 episode
for ep_idx in pick_place_episodes:
    dataset = LeRobotDataset(REPO_ID, episodes=[ep_idx])
    
    # 获取第一帧
    frame0 = dataset[0]
    env_state = frame0['observation.environment_state'].numpy()
    
    # 提取物体位置 [4:7]
    obj_pos = env_state[4:7]
    
    # 提取目标误差信息 [7:10]
    goal_err = env_state[7:10]
    
    all_obj_positions.append(obj_pos.copy())
    all_goal_errors.append(goal_err.copy())
    
    print(f"{ep_idx:<10} {obj_pos[0]:<12.6f} {obj_pos[1]:<12.6f} {obj_pos[2]:<12.6f} "
          f"{goal_err[0]:<12.6e} {goal_err[1]:<12.6e} {goal_err[2]:<12.6e}")

# 3. 统计分析
print("\n" + "=" * 80)
print("统计分析")
print("=" * 80)

all_obj_positions = np.array(all_obj_positions)
all_goal_errors = np.array(all_goal_errors)

print(f"\n物体位置统计 (obj_pose):")
print(f"  X: mean={all_obj_positions[:, 0].mean():.6f}, std={all_obj_positions[:, 0].std():.6f}, "
      f"min={all_obj_positions[:, 0].min():.6f}, max={all_obj_positions[:, 0].max():.6f}")
print(f"  Y: mean={all_obj_positions[:, 1].mean():.6f}, std={all_obj_positions[:, 1].std():.6f}, "
      f"min={all_obj_positions[:, 1].min():.6f}, max={all_obj_positions[:, 1].max():.6f}")
print(f"  Z: mean={all_obj_positions[:, 2].mean():.6f}, std={all_obj_positions[:, 2].std():.6f}, "
      f"min={all_obj_positions[:, 2].min():.6f}, max={all_obj_positions[:, 2].max():.6f}")

print(f"\n目标误差统计:")
print(f"  X: mean={all_goal_errors[:, 0].mean():.6e}, std={all_goal_errors[:, 0].std():.6e}")
print(f"  Y: mean={all_goal_errors[:, 1].mean():.6e}, std={all_goal_errors[:, 1].std():.6e}")
print(f"  Z: mean={all_goal_errors[:, 2].mean():.6e}, std={all_goal_errors[:, 2].std():.6e}")

# 4. 对比 Meta-World 环境的固定配置
print("\n" + "=" * 80)
print("对比 Meta-World 环境配置")
print("=" * 80)

# 尝试几个不同的 seed
for seed in [0, 1, 42, 100, 123]:
    mt1 = metaworld.MT1("pick-place-v3", seed=seed)
    env = mt1.train_classes["pick-place-v3"](render_mode="rgb_array")
    env.set_task(mt1.train_tasks[0])
    env.reset()
    
    print(f"\nSeed {seed}:")
    print(f"  obj_init_pos: {env.obj_init_pos}")
    print(f"  goal: {env.goal}")

# 5. 结论
print("\n" + "=" * 80)
print("结论")
print("=" * 80)

print("""
1. ✓ 可以从数据集中提取每个 episode 的初始物体位置！
   - 位置在 observation.environment_state[4:7]
   - 这是每个时间步的实际物体位置

2. 从统计结果可以看到：
   - 如果所有 episode 的 obj_pos 相同 → 数据集使用固定配置
   - 如果 obj_pos 不同 → 数据集使用了随机化

3. 如果你想研究物体位置对泛化性的影响：
   - 方案 A: 使用现有数据集，分析不同初始位置的 episode 的模型表现
   - 方案 B: 生成新数据集，显式随机化 obj_init_pos 和 goal

4. 目标位置 (goal) 的信息：
   - observation.environment_state[7:10] 可能是误差向量，不是绝对目标位置
   - 绝对目标位置需要从环境中获取 (env.goal)
""")

print("\n" + "=" * 80)
print("完成！")
print("=" * 80)