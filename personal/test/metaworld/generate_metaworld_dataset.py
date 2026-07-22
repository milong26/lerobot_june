#!/usr/bin/env python
"""
使用 Meta-World expert policy 生成更多 pick-place-v3 的示范数据
可以控制：
- 生成多少个 episode
- 使用哪些 seed（控制物体初始位置）
- 是否随机化目标位置
"""

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import time
import numpy as np
import torch
import metaworld
import metaworld.policies as policies
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ==================== 配置 ====================
TASK_NAME = "pick-place-v3"
NUM_EPISODES = 100  # 想生成多少个 episode
OUTPUT_DIR = "./outputs/metaworld_pick_place_expanded"
REPO_ID = "your-username/metaworld_pick_place_expanded"  # 如果要 push 到 HF Hub

# 物体位置随机化配置
OBJ_RANDOMIZATION = {
    "use_random": True,  # 是否随机化物体位置
    "seed_start": 0,     # 起始 seed
    "seed_end": NUM_EPISODES,  # 结束 seed
}

# 目标位置随机化配置（可选）
# 注意：原始数据集使用固定目标位置 [0.1, 0.8, 0.2]
# 如果想保持一致，设置 use_random=False
# 如果想研究目标位置对泛化性的影响，可以启用随机化
# 安全范围（来自 goal_space）：X:[-0.1, 0.1], Y:[0.8, 0.9], Z:[0.05, 0.3]
GOAL_RANDOMIZATION = {
    "use_random": False,  # 是否随机化目标位置
    "goal_range": {
        "x": (-0.1, 0.1),   # goal_space 的 X 范围
        "y": (0.8, 0.9),    # goal_space 的 Y 范围
        "z": (0.05, 0.3),   # goal_space 的 Z 范围
    }
}
# ==============================================

def create_metaworld_env(task_name, seed=None, custom_goal=None):
    """创建 Meta-World 环境，支持自定义物体位置和目标位置"""
    mt1 = metaworld.MT1(task_name, seed=seed)
    env = mt1.train_classes[task_name](render_mode="rgb_array", camera_name="corner2")
    task = mt1.train_tasks[0]
    env.set_task(task)
    
    # 调整相机位置
    env.model.cam_pos[2] = [0.75, 0.075, 0.7]
    
    # 可选：自定义目标位置
    if custom_goal is not None:
        env.goal = np.array(custom_goal)
    
    # 允许随机化
    env._freeze_rand_vec = False
    
    return env

def run_episode(env, expert_policy, max_steps=500):
    """运行一个完整的 episode，返回 (frames, actions, rewards, success)"""
    obs, info = env.reset()
    
    frames = []
    actions = []
    rewards = []
    successes = []
    
    for step in range(max_steps):
        # 使用 expert policy 生成动作
        action = expert_policy.get_action(obs)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 渲染图像
        image = env.render()
        if image is not None:
            # corner2 相机需要翻转
            image = np.flip(image, (0, 1))
        
        frames.append({
            "observation.image": image,
            "observation.state": obs[:4].copy(),  # 手臂位置 + 夹爪
            "observation.environment_state": obs.copy(),  # 完整 39 维
            "action": action.copy(),
            "next.reward": np.array([reward], dtype=np.float32),
            "next.success": np.array([info.get("success", 0)], dtype=bool),
        })
        
        actions.append(action)
        rewards.append(reward)
        successes.append(info.get("success", 0))
        
        if terminated or truncated:
            break
    
    return frames, successes

def main():
    print("=" * 80)
    print(f"生成 Meta-World {TASK_NAME} 数据集")
    print(f"目标 episode 数: {NUM_EPISODES}")
    print("=" * 80)
    
    # 创建 expert policy
    policy_class_name = f"Sawyer{TASK_NAME.replace('-', ' ').title().replace(' ', '')}Policy"
    policy_class = getattr(policies, policy_class_name)
    expert_policy = policy_class()
    
    print(f"\n使用 expert policy: {policy_class_name}")
    
    # 创建数据集
    features = {
        "observation.image": {"dtype": "image", "shape": (3, 480, 480), "names": ["channels", "height", "width"]},
        "observation.state": {"dtype": "float32", "shape": (4,)},
        "observation.environment_state": {"dtype": "float32", "shape": (39,), "names": ["keypoints"]},
        "action": {"dtype": "float32", "shape": (4,), "names": {"axes": ["x", "y", "z", "gripper"]}},
        "next.reward": {"dtype": "float32", "shape": (1,)},
        "next.success": {"dtype": "bool", "shape": (1,)},
    }
    
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=80,
        features=features,
        root=OUTPUT_DIR,
        robot_type="metaworld",
        use_videos=True,
    )
    
    print(f"\n数据集创建成功: {OUTPUT_DIR}")
    
    # 生成数据
    success_count = 0
    start_time = time.time()
    
    for ep_idx in range(NUM_EPISODES):
        ep_start_time = time.time()
        
        # 配置 seed
        seed = OBJ_RANDOMIZATION["seed_start"] + ep_idx
        
        # 可选：随机化目标位置
        custom_goal = None
        if GOAL_RANDOMIZATION["use_random"]:
            custom_goal = [
                np.random.uniform(*GOAL_RANDOMIZATION["goal_range"]["x"]),
                np.random.uniform(*GOAL_RANDOMIZATION["goal_range"]["y"]),
                np.random.uniform(*GOAL_RANDOMIZATION["goal_range"]["z"]),
            ]
        
        # 创建环境
        env = create_metaworld_env(TASK_NAME, seed=seed, custom_goal=custom_goal)
        
        # 运行 episode
        frames, successes = run_episode(env, expert_policy)
        
        # 检查是否成功
        episode_success = any(successes)
        if episode_success:
            success_count += 1
        
        # 添加帧到数据集
        for frame in frames:
            frame["task"] = "Pick and place a puck to a goal"
            dataset.add_frame(frame)
        
        # 保存 episode
        dataset.save_episode()
        
        # 打印进度
        elapsed = time.time() - ep_start_time
        print(f"Episode {ep_idx + 1}/{NUM_EPISODES} | "
              f"Frames: {len(frames):4d} | "
              f"Success: {'✓' if episode_success else '✗'} | "
              f"Obj pos: {env.obj_init_pos} | "
              f"Time: {elapsed:.2f}s")
        
        env.close()
    
    # 完成
    dataset.finalize()
    
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"数据集生成完成！")
    print(f"总 episode 数: {NUM_EPISODES}")
    print(f"成功 episode 数: {success_count} ({success_count/NUM_EPISODES*100:.1f}%)")
    print(f"总用时: {total_time:.1f}s")
    print(f"保存路径: {OUTPUT_DIR}")
    print("=" * 80)
    
    # 可选：push 到 HuggingFace Hub
    # dataset.push_to_hub(tags=["metaworld", "pick-place"])

if __name__ == "__main__":
    main()