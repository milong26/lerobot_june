#!/usr/bin/env python
"""
生成均匀分布在工作空间上的 episode 数据集（task-aware，支持任意 MetaWorld task）。

策略：
1. 根据 --task 创建 MetaWorld 环境，自动解析该任务的 reset randomization space
2. 对所有可采样变量（obj、goal 等）进行均匀覆盖采样
3. 通过 monkeypatch _get_state_rand_vec 注入完整初始状态
4. 使用 expert policy rollout 并保存 LeRobot episode

使用示例:
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


class RandVecExhaustedError(RuntimeError):
    pass


def create_metaworld_env(task_name, seed=None, camera_name="corner3"):
    mt1 = metaworld.MT1(task_name, seed=seed)
    env = mt1.train_classes[task_name](render_mode="rgb_array", camera_name=camera_name)
    task = mt1.train_tasks[0]
    env.set_task(task)
    env._freeze_rand_vec = True
    return env


def introspect_task_space(task_name, seed=42):
    """解析当前 MetaWorld task 的可采样空间。

    从 env._random_reset_space 读取 low/high/shape，
    并通过一次默认 reset 获取 obj_init_pos、goal 等属性的存在性和维度。

    返回 dict:
        space: gym.spaces.Box 对象 (_random_reset_space)
        low: np.ndarray
        high: np.ndarray
        dim: int
        obj_pos_dim: int (obj_init_pos 的维度，通常 3)
        goal_pos_dim: int (goal 的维度，通常 3，可能不存在)
        has_goal: bool
        rejection_constraint: str or None (描述拒绝采样条件)
    """
    env = create_metaworld_env(task_name, seed=seed)
    env._freeze_rand_vec = False

    space = env._random_reset_space
    low = space.low.copy()
    high = space.high.copy()
    dim = space.shape[0]

    has_goal = hasattr(env, "goal") and env.goal is not None
    goal_pos_dim = len(env.goal) if has_goal else 0

    env.reset()
    obj_pos_dim = len(env.obj_init_pos) if env.obj_init_pos is not None else 0

    rejection_constraint = None
    if task_name == "pick-place-v3":
        rejection_constraint = "planar_dist(obj_xy, goal_xy) >= 0.15"

    env.close()

    return {
        "space": space,
        "low": low,
        "high": high,
        "dim": dim,
        "obj_pos_dim": obj_pos_dim,
        "goal_pos_dim": goal_pos_dim,
        "has_goal": has_goal,
        "rejection_constraint": rejection_constraint,
    }


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


def inject_rand_vec(env, rand_vec, max_reject_attempts=200):
    """通过 monkeypatch _get_state_rand_vec 注入完整初始状态。

    rand_vec 必须与 env._random_reset_space 的维度一致。
    任务内部的 reset_model() 可能有拒绝采样循环（如 pick-place 要求 obj/goal 平面距离 >= 0.15），
    我们每次都返回相同的 rand_vec，如果拒绝采样循环超过 max_reject_attempts 次，
    抛出 RandVecExhaustedError 表示该状态对该任务不合法。
    """
    env._freeze_rand_vec = False
    vec = np.asarray(rand_vec, dtype=np.float64).copy()
    state = {"calls": 0}

    def _patched():
        state["calls"] += 1
        if state["calls"] > max_reject_attempts:
            raise RandVecExhaustedError(
                f"rand_vec 被任务内部拒绝采样循环拒绝超过 {max_reject_attempts} 次，"
                "该初始状态对当前 task 不合法。"
            )
        return vec.copy()

    env._get_state_rand_vec = _patched


def set_task_state_directly(env, rand_vec):
    """根据当前 task 的环境定义，注入完整的初始状态 rand_vec。

    rand_vec 应该覆盖该任务 _random_reset_space 中的所有可移动变量
    （例如 obj position、goal position 等）。

    调用此函数后，需要再调用 env.reset() 来应用注入的状态。
    """
    inject_rand_vec(env, rand_vec)


def compute_uniform_samples(task_space_info, num_samples, existing_states=None):
    """根据 task 的可采样空间生成均匀采样点。

    对于 pick-place-v3 等有 obj + goal 的任务，在 obj x-y 平面生成网格，
    对 goal 在其合法空间内均匀随机采样，并应用拒绝采样约束。

    对于没有 goal 或其他简单任务，直接在 _random_reset_space 内均匀随机采样。

    返回 list of np.ndarray，每个元素是一个完整的 rand_vec。
    """
    low = task_space_info["low"]
    high = task_space_info["high"]
    dim = task_space_info["dim"]
    obj_dim = task_space_info["obj_pos_dim"]
    has_goal = task_space_info["has_goal"]
    goal_dim = task_space_info["goal_pos_dim"]
    constraint = task_space_info["rejection_constraint"]

    existing_set = set()
    if existing_states is not None:
        for s in existing_states:
            existing_set.add(tuple(np.round(s, 4)))

    samples = []
    max_attempts_per_sample = 50

    if has_goal and obj_dim >= 3 and goal_dim >= 3:
        obj_low = low[:obj_dim]
        obj_high = high[:obj_dim]
        goal_low = low[obj_dim:obj_dim + goal_dim]
        goal_high = high[obj_dim:obj_dim + goal_dim]

        obj_xy_low = obj_low[:2]
        obj_xy_high = obj_high[:2]

        aspect_ratio = (obj_xy_high[0] - obj_xy_low[0]) / max(obj_xy_high[1] - obj_xy_low[1], 1e-6)
        n_x = int(round(np.sqrt(num_samples * aspect_ratio)))
        n_y = int(round(num_samples / n_x))
        while n_x * n_y < num_samples:
            n_y += 1

        x_edges = np.linspace(obj_xy_low[0], obj_xy_high[0], n_x + 1)
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_edges = np.linspace(obj_xy_low[1], obj_xy_high[1], n_y + 1)
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2

        obj_xy_grid = []
        for xi in x_centers:
            for yi in y_centers:
                obj_xy_grid.append(np.array([xi, yi]))

        rng = np.random.RandomState(42)
        for obj_xy in obj_xy_grid:
            if len(samples) >= num_samples:
                break

            for _ in range(max_attempts_per_sample):
                obj_z = rng.uniform(obj_low[2], obj_high[2]) if obj_dim > 2 else 0.0
                obj_pos = np.array([obj_xy[0], obj_xy[1], obj_z])

                goal_pos = rng.uniform(goal_low, goal_high)

                if constraint == "planar_dist(obj_xy, goal_xy) >= 0.15":
                    if np.linalg.norm(goal_pos[:2] - obj_xy) < 0.15:
                        continue

                rand_vec = np.concatenate([obj_pos, goal_pos])
                if dim > obj_dim + goal_dim:
                    extra = rng.uniform(low[obj_dim + goal_dim:], high[obj_dim + goal_dim:])
                    rand_vec = np.concatenate([rand_vec, extra])

                key = tuple(np.round(rand_vec, 4))
                if key not in existing_set:
                    samples.append(rand_vec)
                    existing_set.add(key)
                    break

    else:
        rng = np.random.RandomState(42)
        for _ in range(num_samples * max_attempts_per_sample):
            if len(samples) >= num_samples:
                break
            rand_vec = rng.uniform(low, high)
            key = tuple(np.round(rand_vec, 4))
            if key not in existing_set:
                samples.append(rand_vec)
                existing_set.add(key)

    return samples


def run_episode(env_top, env_wrist, expert_policy, task_name, max_steps=500, image_size=480, extra_frames_after_success=10):
    obs, info = env_top.reset()
    env_wrist.reset()
    sync_env_state(env_top, env_wrist)

    obj_pose = get_obj_pose_from_env(env_top)
    goal_pose = get_goal_pose_from_env(env_top) if hasattr(env_top, "goal") and env_top.goal is not None else None

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


def load_existing_states(metadata_file):
    if not Path(metadata_file).exists():
        return [], []

    with open(metadata_file, "r") as f:
        old_metadata = json.load(f)

    existing_positions = set()
    existing_rand_vecs = []
    for ep in old_metadata.get("episodes", []):
        pos = tuple(round(v, 3) for v in ep["obj_init_pos"])
        existing_positions.add(pos)

        if "rand_vec" in ep:
            existing_rand_vecs.append(np.array(ep["rand_vec"]))

    return existing_positions, existing_rand_vecs


def save_episode_metadata(output_dir, episode_infos, task_name, task_space_info=None):
    metadata_file = Path(output_dir) / "episode_initial_states.json"

    metadata = {
        "task": task_name,
        "num_episodes": len(episode_infos),
        "env_state_structure": {
            "reset_space_dim": task_space_info["dim"] if task_space_info else None,
            "obj_pos_dim": task_space_info["obj_pos_dim"] if task_space_info else None,
            "has_goal": task_space_info["has_goal"] if task_space_info else None,
            "goal_pos_dim": task_space_info["goal_pos_dim"] if task_space_info else None,
            "rejection_constraint": task_space_info["rejection_constraint"] if task_space_info else None,
        },
        "episodes": [],
    }

    for i, info in enumerate(episode_infos):
        ep_data = {
            "episode_index": i,
            "obj_init_pos": info["obj_init_pos"].tolist(),
            "goal_pose": info["goal_pose"].tolist() if info["goal_pose"] is not None else None,
            "success": bool(info["success"]),
            "num_frames": info["num_frames"],
            "seed_used": info.get("seed_used", None),
        }
        if "rand_vec" in info:
            ep_data["rand_vec"] = info["rand_vec"].tolist()
        metadata["episodes"].append(ep_data)

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nEpisode初始环境信息已保存到: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description="生成均匀分布在工作空间上的 episode 数据集（task-aware）",
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
    print("Meta-World 均匀分布数据采集 (Task-Aware)")
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

    print("\n解析任务的可采样空间...")
    task_space_info = introspect_task_space(args.task)
    print(f"  Reset space dim: {task_space_info['dim']}")
    print(f"  Obj pos dim: {task_space_info['obj_pos_dim']}")
    print(f"  Has goal: {task_space_info['has_goal']} (dim={task_space_info['goal_pos_dim']})")
    print(f"  Rejection constraint: {task_space_info['rejection_constraint']}")
    print(f"  Space low: {task_space_info['low'].tolist()}")
    print(f"  Space high: {task_space_info['high'].tolist()}")

    metadata_file = output_dir / "episode_initial_states.json"
    existing_positions, existing_rand_vecs = load_existing_states(metadata_file)
    print(f"已加载 {len(existing_positions)} 个已有位置记录")

    print(f"\n生成 {args.num_episodes} 个均匀采样点...")
    grid_centers = compute_uniform_samples(task_space_info, args.num_episodes, existing_rand_vecs if existing_rand_vecs else None)
    print(f"共 {len(grid_centers)} 个目标初始状态")

    print("\n加载已有LeRobot数据集...")
    dataset = load_existing_dataset(args.repo_id, args.output_dir)
    existing_episodes = dataset.num_episodes
    print(f"已有 {existing_episodes} 个 episode")

    episode_infos = []
    success_count = 0
    start_time = time.time()

    print(f"\n开始采集均匀分布的episode...")
    print("-" * 80)

    for cell_idx, rand_vec in enumerate(grid_centers):
        obj_pos = rand_vec[:task_space_info["obj_pos_dim"]]
        target_key = tuple(round(v, 3) for v in obj_pos)
        if target_key in existing_positions:
            print(f"\nCell {cell_idx + 1}/{len(grid_centers)} | Target obj: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}] - SKIPPED (already exists)")
            continue

        print(f"\nCell {cell_idx + 1}/{len(grid_centers)} | Target obj: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")

        try:
            env_top = create_metaworld_env(args.task, seed=42, camera_name="corner3")
            env_wrist = create_metaworld_env(args.task, seed=42, camera_name="gripperPOV")

            set_task_state_directly(env_top, rand_vec)
            set_task_state_directly(env_wrist, rand_vec)

            env_top.reset()
            env_wrist.reset()
            sync_env_state(env_top, env_wrist)

            actual_pos = env_top.obj_init_pos.copy()
            print(f"  Set obj_pos: [{actual_pos[0]:.3f}, {actual_pos[1]:.3f}, {actual_pos[2]:.3f}]")

            frames, ep_info = run_episode(env_top, env_wrist, expert_policy, args.task, args.max_steps, args.image_size)
            ep_info["seed_used"] = "direct_set"
            ep_info["rand_vec"] = rand_vec

            env_top.close()
            env_wrist.close()

        except RandVecExhaustedError as e:
            print(f"  SKIPPED: {e}")
            continue

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

    save_episode_metadata(args.output_dir, episode_infos, args.task, task_space_info)

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