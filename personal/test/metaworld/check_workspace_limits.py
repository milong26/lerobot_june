#!/usr/bin/env python
"""
检查 Meta-World pick-place-v3 的工作空间限制和目标位置范围
"""

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import metaworld

print("=" * 80)
print("检查 Meta-World pick-place-v3 的工作空间限制")
print("=" * 80)

# 创建环境
mt1 = metaworld.MT1("pick-place-v3", seed=42)
env = mt1.train_classes["pick-place-v3"](render_mode="rgb_array")
env.set_task(mt1.train_tasks[0])
env.reset()

# 检查环境的各种限制属性
print("\n【1】环境配置属性:")
attrs_to_check = [
    'goal_space', 'hand_high', 'hand_low', 'mocap_high', 'mocap_low',
    'maxPlacingDist', 'maxPushDist', 'objHeight', 'heightTarget',
    'obj_init_pos', 'goal', 'hand_init_pos',
]

for attr in attrs_to_check:
    if hasattr(env, attr):
        val = getattr(env, attr)
        print(f"  {attr}: {val}")

# 检查不同 seed 下的目标位置分布
print("\n【2】不同 seed 下的目标位置:")
goals = []
for seed in range(100):
    mt1 = metaworld.MT1("pick-place-v3", seed=seed)
    env = mt1.train_classes["pick-place-v3"](render_mode="rgb_array")
    env.set_task(mt1.train_tasks[0])
    env.reset()
    goals.append(env.goal.copy())
    env.close()

goals = np.array(goals)
print(f"  Goal X: min={goals[:, 0].min():.4f}, max={goals[:, 0].max():.4f}, mean={goals[:, 0].mean():.4f}")
print(f"  Goal Y: min={goals[:, 1].min():.4f}, max={goals[:, 1].max():.4f}, mean={goals[:, 1].mean():.4f}")
print(f"  Goal Z: min={goals[:, 2].min():.4f}, max={goals[:, 2].max():.4f}, mean={goals[:, 2].mean():.4f}")

# 检查物体位置分布
print("\n【3】不同 seed 下的物体初始位置:")
obj_positions = []
for seed in range(100):
    mt1 = metaworld.MT1("pick-place-v3", seed=seed)
    env = mt1.train_classes["pick-place-v3"](render_mode="rgb_array")
    env.set_task(mt1.train_tasks[0])
    env.reset()
    obj_positions.append(env.obj_init_pos.copy())
    env.close()

obj_positions = np.array(obj_positions)
print(f"  Obj X: min={obj_positions[:, 0].min():.4f}, max={obj_positions[:, 0].max():.4f}, mean={obj_positions[:, 0].mean():.4f}")
print(f"  Obj Y: min={obj_positions[:, 1].min():.4f}, max={obj_positions[:, 1].max():.4f}, mean={obj_positions[:, 1].mean():.4f}")
print(f"  Obj Z: min={obj_positions[:, 2].min():.4f}, max={obj_positions[:, 2].max():.4f}, mean={obj_positions[:, 2].mean():.4f}")

# 检查机械臂的工作空间
print("\n【4】机械臂工作空间限制:")
print(f"  hand_low: {env.hand_low}")
print(f"  hand_high: {env.hand_high}")
print(f"  mocap_low: {env.mocap_low}")
print(f"  mocap_high: {env.mocap_high}")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)
print(f"""
根据 100 个随机 seed 的统计：

1. 目标位置 (goal) 范围：
   X: [{goals[:, 0].min():.4f}, {goals[:, 0].max():.4f}]
   Y: [{goals[:, 1].min():.4f}, {goals[:, 1].max():.4f}]
   Z: [{goals[:, 2].min():.4f}, {goals[:, 2].max():.4f}]

2. 物体位置 (obj_init_pos) 范围：
   X: [{obj_positions[:, 0].min():.4f}, {obj_positions[:, 0].max():.4f}]
   Y: [{obj_positions[:, 1].min():.4f}, {obj_positions[:, 1].max():.4f}]
   Z: [{obj_positions[:, 2].min():.4f}, {obj_positions[:, 2].max():.4f}]

3. 机械臂工作空间：
   X: [{env.hand_low[0]:.4f}, {env.hand_high[0]:.4f}]
   Y: [{env.hand_low[1]:.4f}, {env.hand_high[1]:.4f}]
   Z: [{env.hand_low[2]:.4f}, {env.hand_high[2]:.4f}]

4. 建议的目标位置随机化范围：
   应该在手工作空间内，且与物体位置保持合理距离
""")