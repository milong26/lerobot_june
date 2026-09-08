#!/usr/bin/env python3
"""
检查MetaWorld原生环境的action space
"""
import sys
sys.path.insert(0, "/data/zhonglinye/jun/lerobot/src")

import metaworld
import numpy as np

task_name = "disassemble-v3"

mt1 = metaworld.MT1(task_name, seed=42)
env = mt1.train_classes[task_name](render_mode="rgb_array", camera_name="corner")
env.set_task(mt1.train_tasks[0])

print("="*60)
print("MetaWorld原生环境动作空间")
print("="*60)
print(f"  Action space: {env.action_space}")
print(f"  Action space low: {env.action_space.low}")
print(f"  Action space high: {env.action_space.high}")

# 运行几步看看专家策略输出的动作范围
policy_class_name = f"Sawyer{task_name.replace('-', ' ').title().replace(' ', '')}Policy"
try:
    import metaworld.policies as policies
    expert_policy = getattr(policies, policy_class_name)()
    
    print(f"\n专家策略: {policy_class_name}")
    
    obs, _ = env.reset()
    env._freeze_rand_vec = False
    
    actions = []
    for _ in range(100):
        action = expert_policy.get_action(obs)
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    
    actions = np.array(actions)
    print(f"\n专家策略输出动作范围 (100步):")
    print(f"  min: {actions.min(axis=0)}")
    print(f"  max: {actions.max(axis=0)}")
    print(f"  mean: {actions.mean(axis=0)}")
    print(f"  std: {actions.std(axis=0)}")
    
except AttributeError:
    print(f"\n找不到专家策略: {policy_class_name}")

env.close()