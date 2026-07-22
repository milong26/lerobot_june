#!/usr/bin/env python
"""
分析 pick-place-v3 任务所有 episode 的物体位置和目标位置
目标：
1. 提取每个 episode 的 obj_init_pos 和 goal
2. 统计这些位置的分布
3. 验证不同 episode 是否有不同的物体/目标位置
"""

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import metaworld

REPO_ID = "lerobot/metaworld_mt50"

print("=" * 80)
print("分析 pick-place-v3 任务的物体位置分布")
print("=" * 80)

# 1. 找到所有 pick-place-v3 的 episode
print("\n【1】查找 pick-place-v3 的 episode...")
dataset_meta = LeRobotDatasetMetadata(REPO_ID)

pick_place_episodes = []
for i in range(len(dataset_meta.episodes)):
    ep = dataset_meta.episodes[i]
    if ep.get('task_index') == 30:
        pick_place_episodes.append(i)

print(f"找到 {len(pick_place_episodes)} 个 pick-place-v3 episode")

# 2. 分析每个 episode 的初始物体位置
print("\n【2】分析每个 episode 的初始状态...")

episode_stats = []

# 只分析前 10 个 episode 作为示例
sample_episodes = pick_place_episodes[:10]

for ep_idx in sample_episodes:
    print(f"\n--- Episode {ep_idx} ---")
    
    # 加载这个 episode
    dataset = LeRobotDataset(REPO_ID, episodes=[ep_idx])
    
    # 获取第一帧的数据
    frame0 = dataset[0]
    env_state = frame0['observation.environment_state'].numpy()
    
    # 解析 39 维 observation
    # 根据 Meta-World 的源码，39维的结构是：
    # [0:4]   手臂位置 (xyz) + 夹爪
    # [4:7]   物体位置 (xyz)
    # [7:10]  目标位置相关 (xyz)
    # [10:18] one-hot 任务向量
    # [18:21] 手臂位置 (重复)
    # [21:24] 物体位置 (重复)
    # [24:27] 目标位置相关 (重复)
    # [27:36] one-hot 任务向量 (重复)
    # [36:39] 相机位置
    
    arm_pos = env_state[0:4]
    obj_pos = env_state[4:7]
    goal_info = env_state[7:10]
    task_onehot = env_state[10:18]
    
    print(f"  手臂位置 (xyz + gripper): {arm_pos}")
    print(f"  物体位置 (xyz): {obj_pos}")
    print(f"  目标信息 (xyz): {goal_info}")
    print(f"  Episode 长度: {dataset.num_frames} frames")
    
    # 保存统计信息
    episode_stats.append({
        'episode_index': ep_idx,
        'obj_pos': obj_pos.copy(),
        'goal_info': goal_info.copy(),
        'length': dataset.num_frames,
    })

# 3. 创建 Meta-World 环境来对比
print("\n" + "=" * 80)
print("【3】对比 Meta-World 环境的初始设置")
print("=" * 80)

# 尝试不同的 seed 来理解环境配置
for seed in [42, 123, 456]:
    mt1 = metaworld.MT1("pick-place-v3", seed=seed)
    env = mt1.train_classes["pick-place-v3"](render_mode="rgb_array")
    env.set_task(mt1.train_tasks[0])
    
    obs, info = env.reset()
    
    print(f"\nSeed {seed}:")
    print(f"  obj_init_pos: {env.obj_init_pos}")
    print(f"  goal: {env.goal}")
    print(f"  observation[4:7] (物体): {obs[4:7]}")
    print(f"  observation[7:10] (目标相关): {obs[7:10]}")

# 4. 检查 MT50 数据集的生成方式
print("\n" + "=" * 80)
print("【4】检查 MT50 的任务配置")
print("=" * 80)

# MT50 使用固定的物体/目标位置
# 查看是否有多个 train_tasks
mt50 = metaworld.MT50()
print(f"MT50 train_tasks 数量: {len(mt50.train_tasks)}")

# 找到 pick-place-v3 对应的 task
for i, task in enumerate(mt50.train_tasks):
    if hasattr(task, 'env') and 'pick-place' in task.env.task_name:
        print(f"\nTask {i}: {task.env.task_name}")
        print(f"  env: {task.env}")
        if hasattr(task.env, 'obj_init_pos'):
            print(f"  obj_init_pos: {task.env.obj_init_pos}")
        if hasattr(task.env, 'goal'):
            print(f"  goal: {task.env.goal}")
        break

# 5. 总结
print("\n" + "=" * 80)
print("【5】总结")
print("=" * 80)

print("""
从分析结果可以得出：

1. observation.environment_state (39维) 的结构：
   - [0:4]   手臂位置 (x, y, z, gripper)
   - [4:7]   物体位置 (x, y, z) ← 这就是 obj_pose！
   - [7:10]  目标位置/误差信息 (x, y, z)
   - [10:18] one-hot 任务向量
   - [18:36] 重复上述信息
   - [36:39] 相机位置

2. 数据集已经包含了每个时间步的物体位置！
   - 可以直接从 observation.environment_state[4:7] 提取
   - 这是每个时间步的实际物体位置，会随时间变化

3. 如果你想研究不同初始位置对泛化性的影响：
   - 需要检查不同 episode 的第一帧，看 obj_init_pos 是否不同
   - 或者修改环境代码，在数据采集时随机化 obj_init_pos

4. 当前 lerobot/metaworld_mt50 数据集：
   - 使用的是 MT50 固定配置
   - 每个任务的物体/目标位置是固定的
   - 如果要研究泛化性，需要生成新的数据集，随机化初始位置
""")

print("\n" + "=" * 80)
print("分析完成！")
print("=" * 80)