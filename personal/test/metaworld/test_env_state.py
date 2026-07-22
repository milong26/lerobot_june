#!/usr/bin/env python
"""
深入检查 observation.environment_state 是否包含 obj_pose 和 goal_pose
"""

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

REPO_ID = "lerobot/metaworld_mt50"

print("=" * 80)
print("检查 observation.environment_state 的内容")
print("=" * 80)

# 加载 pick-place-v3 对应的 episode
# 根据之前的输出，task_id=30 是 pick-place-v3
# 我们需要找到 task_id=30 的 episode

dataset_meta = LeRobotDatasetMetadata(REPO_ID)

# 找到 pick-place-v3 的 episode
print("\n查找 pick-place-v3 (task_id=30) 的 episode...")
pick_place_episodes = []
for i in range(len(dataset_meta.episodes)):
    ep = dataset_meta.episodes[i]
    # 检查 task_index（对应 task_id）
    if ep.get('task_index') == 30 or 'Pick and place a puck to a goal' in str(ep.get('tasks', [])):
        pick_place_episodes.append(i)

print(f"找到 {len(pick_place_episodes)} 个 pick-place-v3 的 episode")
print(f"Episode 索引: {pick_place_episodes[:10]}...")

# 加载第一个 pick-place-v3 的 episode
if pick_place_episodes:
    first_ep = pick_place_episodes[0]
    print(f"\n加载 episode {first_ep}...")
    dataset = LeRobotDataset(REPO_ID, episodes=[first_ep])
    
    # 查看第一条数据
    frame0 = dataset[0]
    
    print(f"\n【observation.environment_state】详细信息:")
    env_state = frame0['observation.environment_state']
    print(f"类型: {type(env_state)}")
    print(f"形状: {env_state.shape}")
    print(f"内容:\n{env_state}")
    
    # 对比 observation.state
    print(f"\n【observation.state】:")
    obs_state = frame0['observation.state']
    print(f"形状: {obs_state.shape}")
    print(f"内容: {obs_state}")
    
    # 查看 Meta-World 原始 observation 的结构
    print("\n" + "=" * 80)
    print("对比 Meta-World 原始 observation 结构")
    print("=" * 80)
    
    import metaworld
    
    mt1 = metaworld.MT1("pick-place-v3", seed=42)
    env = mt1.train_classes["pick-place-v3"](render_mode="rgb_array")
    env.set_task(mt1.train_tasks[0])
    
    raw_obs, info = env.reset()
    
    print(f"\n原始 observation 形状: {raw_obs.shape}")
    print(f"原始 observation 内容:\n{raw_obs}")
    
    # 分析 observation 的组成
    # 根据 Meta-World 文档，observation 通常包含：
    # - 手臂位置 (4维)
    # - 物体位置 (3维)
    # - 目标位置 (3维)
    # - 其他信息...
    
    print(f"\n尝试解析 39 维 observation:")
    print(f"  [0:4]   手臂位置 + 夹爪: {raw_obs[0:4]}")
    print(f"  [4:7]   可能是物体位置: {raw_obs[4:7]}")
    print(f"  [7:10]  可能是目标位置: {raw_obs[7:10]}")
    print(f"  [10:]   其他信息: {raw_obs[10:]}")
    
    # 检查环境的具体信息
    print(f"\n环境中的物体和目标位置:")
    print(f"  obj_init_pos: {env.obj_init_pos}")
    print(f"  goal: {env.goal}")
    print(f"  hand_init_pos: {env.hand_init_pos}")
    
    # 检查 environment_state 和原始 observation 的关系
    print(f"\n对比 environment_state 和 raw_obs:")
    if isinstance(env_state, torch.Tensor):
        env_state_np = env_state.numpy()
    else:
        env_state_np = env_state
    
    print(f"environment_state 形状: {env_state_np.shape}")
    print(f"raw_obs 形状: {raw_obs.shape}")
    
    if env_state_np.shape == raw_obs.shape:
        print("\n两者形状相同！检查是否相等:")
        diff = np.abs(env_state_np - raw_obs)
        print(f"最大差异: {diff.max()}")
        print(f"平均差异: {diff.mean()}")
        if diff.max() < 1e-5:
            print("✓ environment_state 就是原始的 39 维 observation！")
        else:
            print("✗ 两者不相等")
    else:
        print(f"\n形状不同，需要进一步分析...")
    
    # 查看多个 frame 的 environment_state 变化
    print(f"\n" + "=" * 80)
    print("查看多个 frame 的 environment_state 变化")
    print("=" * 80)
    
    for frame_idx in [0, 10, 20, 50]:
        if frame_idx < dataset.num_frames:
            frame = dataset[frame_idx]
            env_state = frame['observation.environment_state']
            if isinstance(env_state, torch.Tensor):
                env_state = env_state.numpy()
            
            print(f"\nFrame {frame_idx}:")
            print(f"  [0:4]   手臂: {env_state[0:4]}")
            print(f"  [4:7]   物体?: {env_state[4:7]}")
            print(f"  [7:10]  目标?: {env_state[7:10]}")

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)