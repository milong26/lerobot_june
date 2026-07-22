#!/usr/bin/env python
"""
测试脚本：检查 lerobot/metaworld_mt50 数据集中 pick-place-v3 任务的结构
目标：
1. 查看数据集的 features（有哪些字段）
2. 查看总共有多少 episode
3. 检查每个 episode 的 metadata（是否包含 obj_pose, goal_pose 等信息）
"""

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from pprint import pprint
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

# 数据集路径（根据你的实际情况修改）
REPO_ID = "lerobot/metaworld_mt50"

print("=" * 80)
print(f"检查数据集: {REPO_ID}")
print("=" * 80)

# 1. 先查看元数据（不需要下载实际数据）
print("\n【1】查看数据集元数据...")
meta = LeRobotDatasetMetadata(REPO_ID)

print(f"\n总 episode 数: {meta.total_episodes}")
print(f"总 frame 数: {meta.total_frames}")
print(f"FPS: {meta.fps}")
print(f"Robot 类型: {meta.robot_type}")

# 2. 查看 features（数据集有哪些字段）
print("\n【2】查看数据集 Features（字段结构）:")
pprint(meta.features)

# 3. 查看 tasks 信息
print("\n【3】查看 Tasks 信息:")
if meta.tasks is not None:
    print(f"Task 数量: {len(meta.tasks)}")
    print("Tasks 内容:")
    print(meta.tasks)
else:
    print("Tasks 为 None")

# 4. 查看 episodes metadata
print("\n【4】查看 Episodes Metadata:")
if meta.episodes is not None:
    print(f"Episodes 数量: {len(meta.episodes)}")
    print(f"\nEpisodes 的列名（字段）:")
    print(meta.episodes.column_names)
    
    # 查看前几个 episode 的 metadata
    print(f"\n前 5 个 episode 的 metadata:")
    for i in range(min(5, len(meta.episodes))):
        ep = meta.episodes[i]
        print(f"\nEpisode {i}:")
        for key, value in ep.items():
            print(f"  {key}: {value}")
else:
    print("Episodes 为 None")

# 5. 查看 stats（是否有 obj_pose, goal_pose 的统计信息）
print("\n【5】查看 Stats（字段统计信息）:")
if meta.stats is not None:
    print(f"Stats 的字段:")
    pprint(list(meta.stats.keys()))
    
    # 检查是否有 obj_pose 或 goal_pose 相关的统计
    obj_related = [k for k in meta.stats.keys() if 'obj' in k.lower() or 'goal' in k.lower() or 'pose' in k.lower()]
    if obj_related:
        print(f"\n找到可能与物体位置相关的字段:")
        for k in obj_related:
            print(f"  {k}: {meta.stats[k]}")
    else:
        print("\n未找到包含 'obj', 'goal', 'pose' 的字段")
else:
    print("Stats 为 None")

# 6. 加载实际数据集，查看具体数据
print("\n【6】加载实际数据集（查看第一条数据）...")
dataset = LeRobotDataset(REPO_ID, episodes=[0])

print(f"\n数据集 episode 数: {dataset.num_episodes}")
print(f"数据集 frame 数: {dataset.num_frames}")

# 查看第一条数据
print("\n第一条数据 (episode 0) 的 keys:")
sample = dataset[0]
pprint(list(sample.keys()))

# 查看 observation.state 的形状和内容
if "observation.state" in sample:
    print(f"\nobservation.state 形状: {sample['observation.state'].shape}")
    print(f"observation.state 内容:\n{sample['observation.state']}")

# 查看 action 的形状
if "action" in sample:
    print(f"\naction 形状: {sample['action'].shape}")

# 7. 检查 Meta-World 环境本身的结构
print("\n" + "=" * 80)
print("【7】检查 Meta-World pick-place-v3 环境本身的结构")
print("=" * 80)

import metaworld
import numpy as np

# 创建 pick-place-v3 环境
mt1 = metaworld.MT1("pick-place-v3", seed=42)
env = mt1.train_classes["pick-place-v3"](render_mode="rgb_array")
env.set_task(mt1.train_tasks[0])

print(f"\n环境创建成功！")
print(f"Train tasks 数量: {len(mt1.train_tasks)}")

# Reset 环境
obs, info = env.reset()

print(f"\nObservation 形状: {obs.shape}")
print(f"Observation 内容（前10个元素）: {obs[:10]}")

# 检查环境有哪些属性可以获取物体位置
print(f"\n环境对象的所有属性/方法:")
env_attrs = [attr for attr in dir(env) if not attr.startswith('_')]
print(env_attrs)

# 尝试获取常见的物体位置属性
print(f"\n尝试获取物体位置相关信息:")
possible_attrs = [
    'init_qpos', 'init_qvel', 'model', 'data',
    'site_pos', 'site_xpos', 'body_pos', 'geom_pos',
    'mocap_pos', 'part_pos',
]

for attr in possible_attrs:
    if hasattr(env, attr):
        val = getattr(env, attr)
        if hasattr(val, 'shape'):
            print(f"  {attr}: shape={val.shape}")
        elif hasattr(val, '__len__') and len(val) < 20:
            print(f"  {attr}: {val}")
        else:
            print(f"  {attr}: <存在>")

# 检查 model 的具体信息
if hasattr(env, 'model'):
    print(f"\nModel 相关信息:")
    model_attrs = ['body_names', 'geom_names', 'site_names', 'actuator_names']
    for attr in model_attrs:
        if hasattr(env.model, attr):
            val = getattr(env.model, attr)
            print(f"  {attr}: {val}")
    
    # 检查 body 位置
    if hasattr(env.model, 'body_pos'):
        print(f"\nbody_pos (所有物体的位置):")
        print(env.model.body_pos)
    
    if hasattr(env.model, 'site_pos'):
        print(f"\nsite_pos (所有site的位置):")
        print(env.model.site_pos)

# 检查 data 的信息
if hasattr(env, 'data'):
    print(f"\nData 相关信息:")
    if hasattr(env.data, 'xpos'):
        print(f"xpos (所有物体的世界坐标位置):")
        print(env.data.xpos)
    
    if hasattr(env.data, 'site_xpos'):
        print(f"\nsite_xpos (所有site的世界坐标位置):")
        print(env.data.site_xpos)

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)