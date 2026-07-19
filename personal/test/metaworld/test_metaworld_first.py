import os

# ===== 必须在任何 mujoco / gymnasium import 之前 =====
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import cv2
import imageio
import numpy as np

import gymnasium as gym
import metaworld

# -----------------------------
# 1. 创建 MetaWorld task
# -----------------------------
ml1 = metaworld.ML1("assembly-v3")  # 可以换成 dial-turn-v3 等

env = ml1.train_classes["assembly-v3"]()
task = ml1.train_tasks[0]

env.set_task(task)

# -----------------------------
# 2. reset
# -----------------------------
obs, info = env.reset()

frames = []

# -----------------------------
# 3. rollout
# -----------------------------
for step in range(200):

    # 随机 action（只是测试环境）
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    # MetaWorld RGB 渲染（关键）
    frame = env.render(mode="rgb_array")

    frames.append(frame)

    if terminated or truncated:
        break

env.close()

# -----------------------------
# 4. 保存视频
# -----------------------------
video_path = "metaworld_test.mp4"

imageio.mimsave(video_path, frames, fps=30)

print(f"Saved video to {video_path}")