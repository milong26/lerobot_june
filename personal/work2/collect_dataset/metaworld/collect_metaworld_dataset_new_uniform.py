#!/usr/bin/env python
"""
生成均匀分布在工作空间上的 episode 数据集。

策略：
1. 将 x-y 工作空间划分为网格（如 14x8 = 112 个格子）
2. 对每个格子，通过搜索 seed 找到物体初始位置落在该格子内的环境配置
3. 每个格子生成 1 个成功的 episode
4. 最终得到均匀分布的 100 个 episode

使用示例:
cd /data/zhonglinye/jun/lerobot/personal/work2/collect_dataset
    python collect_metaworld_dataset_new_uniform.py \
        --task pick-place-v3 \
        --num-episodes 100 \
        --output-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_corner3/ \
        --repo-id work2/pick_place_corner3 \
        --fps 80 \
        --image-size 480 \
        --max-steps 500
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import mujoco
import metaworld
import metaworld.policies as policies

os.environ["HF_LEROBOT_HOME"] = str(Path(__file__).parent / "outputs")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "src" / "lerobot" / "envs" / "metaworld_config.json"
with open(CONFIG_PATH) as f:
    METAWORLD_CONFIG = json.load(f)

TASK_DESCRIPTIONS = METAWORLD_CONFIG.get("TASK_DESCRIPTIONS", {})

ENV_STATE_DESCRIPTION = {
    "0:3": "末端执行器(手)位置 xyz",
    "3:4": "夹爪开合度(归一化)",
    "4:7": "物体1位置 xyz (= obj_pose)",
    "7:11": "物体1四元数朝向(4维)",
    "11:14": "物体2位置(单物体任务中恒为0)",
    "14:18": "物体2四元数(恒为0)",
    "18:36": "上一帧的[0:18]原样重复(frame-stack)",
    "36:39": "目标位置 xyz (= goal_pose)",
}

# 工作空间范围（来自 SawyerPickPlaceEnvV3）
OBJ_LOW = np.array([-0.1, 0.6, 0.02])
OBJ_HIGH = np.array([0.1, 0.7, 0.02])


def create_metaworld_env(task_name, seed=None, camera_name="corner3"):
    mt1 = metaworld.MT1(task_name, seed=seed)
    env = mt1.train_classes[task_name](render_mode="rgb_array", camera_name=camera_name)
    task = mt1.train_tasks[0]
    env.set_task(task)
    env._freeze_rand_vec = True
    return env


def sync_env_state(src_env, dst_env):
    dst_env.data.qpos[:] = src_env.data.qpos[:]
    dst_env.data.qvel[:] = src_env.data.qvel[:]
    dst_env.data.ctrl[:] = src_env.data.ctrl[:]
    mujoco.mj_forward(dst_env.model, dst_env.data)


def render_dual_camera(env_top, env_wrist, image_size=480):
    top_image = env_top.render()
    if top_image is not None:
        top_image = np.flip(top_image, (0, 1))
        top_image = resize_image(top_image, image_size)

    wrist_image = env_wrist.render()
    if wrist_image is not None:
        wrist_image = resize_image(wrist_image, image_size)

    return top_image, wrist_image


def resize_image(image, target_size):
    from PIL import Image
    img = Image.fromarray(image)
    img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img)


def get_obj_pose_from_env(env):
    return env.obj_init_pos.copy()


def get_goal_pose_from_env(env):
    return env.goal.copy()


def run_episode(env_top, env_wrist, expert_policy, task_name, max_steps=500, image_size=480, extra_frames_after_success=10):
    obs, info = env_top.reset()
    env_wrist.reset()
    sync_env_state(env_top, env_wrist)

    obj_pose = get_obj_pose_from_env(env_top)
    goal_pose = get_goal_pose_from_env(env_top)

    frames = []
    success_flags = []
    success_detected = False
    frames_after_success = 0

    for step in range(max_steps):
        action = expert_policy.get_action(obs)
        obs, reward, terminated, truncated, info = env_top.step(action)
        sync_env_state(env_top, env_wrist)

        top_image, wrist_image = render_dual_camera(env_top, env_wrist, image_size)

        if top_image is None or wrist_image is None:
            print(f"  Warning: Failed to render at step {step}, skipping frame.")
            continue

        task_description = TASK_DESCRIPTIONS.get(task_name, task_name)
        frame = {
            "observation.images.top": top_image,
            "observation.images.wrist": wrist_image,
            "observation.state": obs[:4].copy().astype(np.float32),
            "observation.environment_state": obs.copy().astype(np.float32),
            "action": action.copy().astype(np.float32),
            "next.reward": np.array([reward], dtype=np.float32),
            "next.success": np.array([info.get("success", 0)], dtype=bool),
            "task": task_description,
        }
        frames.append(frame)
        success_flags.append(info.get("success", 0))

        current_success = info.get("success", 0)
        if current_success and not success_detected:
            success_detected = True
            frames_after_success = 0
            print(f"  >>> Success detected at step {step}, collecting {extra_frames_after_success} more frames...")

        if success_detected:
            frames_after_success += 1
            if frames_after_success >= extra_frames_after_success:
                print(f"  >>> Collected {extra_frames_after_success} extra frames, ending episode.")
                break

        if terminated or truncated:
            break

    episode_info = {
        "obj_init_pos": obj_pose,
        "goal_pose": goal_pose,
        "success": any(success_flags),
        "num_frames": len(frames),
    }

    return frames, episode_info


def load_existing_dataset(repo_id, output_dir):
    """加载已有数据集以 resume。"""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在: {output_dir}")

    metadata_file = output_dir / "episode_initial_states.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"元数据文件不存在: {metadata_file}")

    print(f"加载已有数据集: {output_dir}")
    dataset = LeRobotDataset.resume(repo_id=repo_id, root=output_dir)
    print(f"已加载 {dataset.num_episodes} 个 episode")
    return dataset


def set_obj_position_directly(env, target_pos):
    """直接设置物体初始位置，goal_pose 随机生成。"""
    env._freeze_rand_vec = True
    
    # 从 goal_space 随机采样 goal_pose
    # SawyerPickPlaceEnvV3 的 goal 范围: (-0.1, 0.8, 0.05) 到 (0.1, 0.9, 0.3)
    goal_low = np.array([-0.1, 0.8, 0.05])
    goal_high = np.array([0.1, 0.9, 0.3])
    
    # 随机采样 goal，确保与 target 距离 >= 0.15
    max_attempts = 100
    for _ in range(max_attempts):
        goal_pos = np.random.uniform(goal_low, goal_high)
        if np.linalg.norm(goal_pos[:2] - target_pos[:2]) >= 0.15:
            break
    else:
        # 如果 100 次都没找到，使用默认值并强制调整
        print(f"  WARNING: Could not find suitable goal after {max_attempts} attempts, using adjusted default")
        goal_pos = np.array([0.1, 0.8, 0.2])
        while np.linalg.norm(goal_pos[:2] - target_pos[:2]) < 0.15:
            goal_pos[0] += 0.15

    rand_vec = np.hstack([target_pos.copy(), goal_pos])
    env._get_state_rand_vec = lambda: rand_vec.copy()


def compute_grid_centers(num_episodes):
    """计算网格中心点坐标，确保均匀覆盖工作空间。"""
    range_x = OBJ_HIGH[0] - OBJ_LOW[0]  # 0.2
    range_y = OBJ_HIGH[1] - OBJ_LOW[1]  # 0.1
    aspect_ratio = range_x / range_y  # 2.0

    n_x = int(round(np.sqrt(num_episodes * aspect_ratio)))
    n_y = int(round(num_episodes / n_x))

    while n_x * n_y < num_episodes:
        n_y += 1

    print(f"Grid size: {n_x} x {n_y} = {n_x * n_y} cells (target: {num_episodes})")

    x_centers = np.linspace(OBJ_LOW[0], OBJ_HIGH[0], n_x + 1)
    x_centers = (x_centers[:-1] + x_centers[1:]) / 2

    y_centers = np.linspace(OBJ_LOW[1], OBJ_HIGH[1], n_y + 1)
    y_centers = (y_centers[:-1] + y_centers[1:]) / 2

    centers = []
    for i in range(n_x):
        for j in range(n_y):
            if len(centers) >= num_episodes:
                break
            centers.append(np.array([x_centers[i], y_centers[j], OBJ_LOW[2]]))

    return centers


def find_seed_for_target_position(task_name, target_pos, seed_start=10000, max_search=50000, tol=0.015):
    """搜索能使物体初始位置接近目标位置的 seed。"""
    for seed in range(seed_start, seed_start + max_search):
        env = create_metaworld_env(task_name, seed=seed)
        env.reset()
        obj_pos = env.obj_init_pos.copy()
        env.close()

        dist = np.sqrt((obj_pos[0] - target_pos[0]) ** 2 + (obj_pos[1] - target_pos[1]) ** 2)
        if dist < tol:
            return seed, obj_pos

    return None, None


def save_episode_metadata(output_dir, episode_infos, task_name):
    metadata_file = Path(output_dir) / "episode_initial_states.json"

    metadata = {
        "task": task_name,
        "num_episodes": len(episode_infos),
        "env_state_structure": ENV_STATE_DESCRIPTION,
        "episodes": [],
    }

    for i, info in enumerate(episode_infos):
        ep_data = {
            "episode_index": i,
            "obj_init_pos": info["obj_init_pos"].tolist(),
            "goal_pose": info["goal_pose"].tolist(),
            "success": bool(info["success"]),
            "num_frames": info["num_frames"],
            "seed_used": info.get("seed_used", None),
        }
        metadata["episodes"].append(ep_data)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nEpisode初始环境信息已保存到: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description="生成均匀分布在工作空间上的 episode 数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--task", type=str, default="pick-place-v3")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pickplacev3")
    parser.add_argument("--repo-id", type=str, default="work2/metaworld_pick_place")
    parser.add_argument("--fps", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--max-steps", type=int, default=500)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("Meta-World 均匀分布数据采集")
    print("=" * 80)
    print(f"任务: {args.task}")
    print(f"目标Episode数量: {args.num_episodes}")
    print(f"输出目录: {args.output_dir}")
    print(f"Repo ID: {args.repo_id}")
    print(f"FPS: {args.fps}")
    print(f"图像分辨率: {args.image_size}x{args.image_size}")
    print(f"最大步数/episode: {args.max_steps}")
    print("=" * 80)

    policy_class_name = f"Sawyer{args.task.replace('-', ' ').title().replace(' ', '')}Policy"
    try:
        policy_class = getattr(policies, policy_class_name)
        expert_policy = policy_class()
        print(f"\n使用专家策略: {policy_class_name}")
    except AttributeError:
        print(f"\n错误: 找不到任务 {args.task} 的专家策略 {policy_class_name}")
        sys.exit(1)

    # 计算网格中心点
    print("\n计算网格中心点...")
    grid_centers = compute_grid_centers(args.num_episodes)
    print(f"共 {len(grid_centers)} 个目标位置")

    # 加载已有数据集
    print("\n加载已有LeRobot数据集...")
    dataset = load_existing_dataset(args.repo_id, args.output_dir)
    existing_episodes = dataset.num_episodes
    print(f"已有 {existing_episodes} 个 episode")

    # 加载已有 episode 的 obj_pos，用于跳过已完成的格子
    existing_positions = set()
    metadata_file = Path(args.output_dir) / "episode_initial_states.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            old_metadata = json.load(f)
        for ep in old_metadata.get("episodes", []):
            pos = tuple(round(v, 3) for v in ep["obj_init_pos"])
            existing_positions.add(pos)
        print(f"已加载 {len(existing_positions)} 个已有位置记录")

    episode_infos = []
    success_count = 0
    start_time = time.time()

    print(f"\n开始采集均匀分布的episode...")
    print("-" * 80)

    for cell_idx, target_pos in enumerate(grid_centers):
        # 跳过已存在的 episode
        target_key = tuple(round(v, 3) for v in target_pos)
        if target_key in existing_positions:
            print(f"\nCell {cell_idx + 1}/{len(grid_centers)} | Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] - SKIPPED (already exists)")
            continue

        print(f"\nCell {cell_idx + 1}/{len(grid_centers)} | Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")

        # 直接设置物体位置，不搜索 seed
        env_top = create_metaworld_env(args.task, seed=42, camera_name="corner3")
        env_wrist = create_metaworld_env(args.task, seed=42, camera_name="gripperPOV")

        set_obj_position_directly(env_top, target_pos)
        set_obj_position_directly(env_wrist, target_pos)

        env_top.reset()
        env_wrist.reset()
        sync_env_state(env_top, env_wrist)

        actual_pos = env_top.obj_init_pos.copy()
        print(f"  Set obj_pos: [{actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f}]")

        frames, ep_info = run_episode(env_top, env_wrist, expert_policy, args.task, args.max_steps, args.image_size)
        ep_info["seed_used"] = "direct_set"

        env_top.close()
        env_wrist.close()

        if ep_info["success"]:
            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            episode_infos.append(ep_info)
            success_count += 1
            print(f"  SUCCESS | Frames: {ep_info['num_frames']}")
        else:
            print(f"  FAILED | Frames: {ep_info['num_frames']}")

    print("\n" + "-" * 80)
    print("正在保存数据集...")
    dataset.finalize()

    save_episode_metadata(args.output_dir, episode_infos, args.task)

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("采集完成！")
    print("=" * 80)
    print(f"目标Episode数: {len(grid_centers)}")
    print(f"本次成功Episode: {success_count}")
    print(f"数据集总Episode: {dataset.num_episodes}")
    print(f"总用时: {total_time:.1f}s")
    print(f"数据集路径: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()