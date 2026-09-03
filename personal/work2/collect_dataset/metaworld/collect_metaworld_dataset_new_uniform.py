#!/usr/bin/env python
"""
生成均匀分布在 reset random vector 空间上的 episode 数据集（task-aware，支持任意 MetaWorld task）。

策略：
1. 根据 --task 创建 MetaWorld 环境，自动解析该任务的 _random_reset_space
2. 对完整 random reset vector 空间进行均匀覆盖采样（Latin Hypercube Sampling）
3. 通过 monkeypatch _get_state_rand_vec 注入完整初始状态
4. 使用 expert policy rollout，仅保存成功的 demonstration
5. 失败自动重新采样，支持在已有 LeRobot dataset 基础上 resume 新增 episode

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

MAX_REJECT_ATTEMPTS = 200
MAX_ATTEMPTS_PER_EPISODE = 50
EXTRA_FRAMES_AFTER_SUCCESS = 10


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
    """解析当前 MetaWorld task 的完整 reset randomization space。

    核心对象是 env._random_reset_space（gym.spaces.Box），
    不同 task 的 random vector 含义不同（如 pick-place-v3 是 obj_xyz+goal_xyz，
    reach-v3 可能只有 goal，其他 task 可能包含 task-specific 变量）。

    返回 dict:
        space: gym.spaces.Box 对象 (_random_reset_space)
        low: np.ndarray
        high: np.ndarray
        dim: int
        has_obj: bool (是否存在 obj_init_pos 属性)
        obj_pos_dim: int (obj_init_pos 的维度，不存在则为 0)
        has_goal: bool (是否存在 goal 属性)
        goal_pos_dim: int (goal 的维度，不存在则为 0)
        rejection_constraint: str or None
    """
    env = create_metaworld_env(task_name, seed=seed)
    env._freeze_rand_vec = False

    space = env._random_reset_space
    assert space is not None, f"{task_name} 的 _random_reset_space 为 None"
    low = space.low.copy()
    high = space.high.copy()
    dim = space.shape[0]

    has_obj = hasattr(env, "obj_init_pos") and env.obj_init_pos is not None
    obj_pos_dim = len(env.obj_init_pos) if has_obj else 0

    has_goal = hasattr(env, "goal") and env.goal is not None
    goal_pos_dim = len(env.goal) if has_goal else 0

    env.reset()

    rejection_constraint = None
    if task_name == "pick-place-v3":
        rejection_constraint = "planar_dist(obj_xy, goal_xy) >= 0.15"

    env.close()

    return {
        "space": space,
        "low": low,
        "high": high,
        "dim": dim,
        "has_obj": has_obj,
        "obj_pos_dim": obj_pos_dim,
        "has_goal": has_goal,
        "goal_pos_dim": goal_pos_dim,
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
    if hasattr(env, "obj_init_pos") and env.obj_init_pos is not None:
        return env.obj_init_pos.copy()
    return None


def get_goal_pose_from_env(env):
    if hasattr(env, "goal") and env.goal is not None:
        return env.goal.copy()
    return None


def inject_rand_vec(env, rand_vec, max_reject_attempts=MAX_REJECT_ATTEMPTS):
    """通过 monkeypatch _get_state_rand_vec 注入完整初始状态。

    rand_vec 必须与 env._random_reset_space 的维度一致。
    任务内部的 reset_model() 可能有拒绝采样循环（如 pick-place 要求 obj/goal 平面距离 >= 0.15），
    我们每次都返回相同的 rand_vec，如果拒绝采样循环超过 max_reject_attempts 次，
    抛出 RandVecExhaustedError 表示该状态对该任务不合法。
    """
    env._freeze_rand_vec = False
    env.seeded_rand_vec = False
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

    rand_vec 应该覆盖该任务 _random_reset_space 中的所有可移动变量。
    调用此函数后，需要再调用 env.reset() 来应用注入的状态。
    """
    inject_rand_vec(env, rand_vec)


def latin_hypercube_sample(dim, n_samples, low, high, rng):
    """Latin Hypercube Sampling for uniform space coverage.

    对每个维度划分 n_samples 个等宽区间，每个区间内随机采样一个点，
    然后对各维度的采样值进行随机排列组合，保证高维空间覆盖均匀。
    """
    samples = np.zeros((n_samples, dim))
    for d in range(dim):
        edges = np.linspace(low[d], high[d], n_samples + 1)
        points = rng.uniform(edges[:-1], edges[1:])
        rng.shuffle(points)
        samples[:, d] = points
    return samples


def compute_uniform_samples(task_space_info, num_samples, existing_rand_vecs=None):
    """根据 task 的完整 _random_reset_space 生成均匀采样点。

    使用 Latin Hypercube Sampling 对完整 random vector 空间进行覆盖采样，
    保证每个维度均匀分布，同时避免与已有采样点重复。

    返回 list of np.ndarray，每个元素是一个完整的 rand_vec，
    维度与 env._random_reset_space.shape 一致。
    """
    low = task_space_info["low"]
    high = task_space_info["high"]
    dim = task_space_info["dim"]

    existing_set = set()
    if existing_rand_vecs is not None:
        for s in existing_rand_vecs:
            existing_set.add(tuple(np.round(s, 4)))

    rng = np.random.RandomState(42)
    samples = []
    batch_size = max(num_samples * 3, 500)

    for _ in range(20):
        if len(samples) >= num_samples:
            break

        lhc_samples = latin_hypercube_sample(dim, batch_size, low, high, rng)

        for vec in lhc_samples:
            if len(samples) >= num_samples:
                break
            key = tuple(np.round(vec, 4))
            if key not in existing_set:
                samples.append(vec.copy())
                existing_set.add(key)

    return samples


def run_episode(env_top, env_wrist, expert_policy, task_name, max_steps=500, image_size=480, extra_frames_after_success=EXTRA_FRAMES_AFTER_SUCCESS):
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
        return []

    with open(metadata_file, "r") as f:
        old_metadata = json.load(f)

    existing_rand_vecs = []
    for ep in old_metadata.get("episodes", []):
        if "rand_vec" in ep:
            existing_rand_vecs.append(np.array(ep["rand_vec"]))

    return existing_rand_vecs


def save_episode_metadata(output_dir, episode_infos, task_name, task_space_info=None, append=False):
    metadata_file = Path(output_dir) / "episode_initial_states.json"

    if append and metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "task": task_name,
            "num_episodes": 0,
            "env_state_structure": {
                "reset_space_dim": task_space_info["dim"] if task_space_info else None,
                "has_obj": task_space_info["has_obj"] if task_space_info else None,
                "obj_pos_dim": task_space_info["obj_pos_dim"] if task_space_info else None,
                "has_goal": task_space_info["has_goal"] if task_space_info else None,
                "goal_pos_dim": task_space_info["goal_pos_dim"] if task_space_info else None,
                "rejection_constraint": task_space_info["rejection_constraint"] if task_space_info else None,
            },
            "episodes": [],
        }

    base_index = len(metadata["episodes"])
    for i, info in enumerate(episode_infos):
        ep_data = {
            "episode_index": base_index + i,
            "success": bool(info["success"]),
            "num_frames": info["num_frames"],
        }
        if info.get("obj_init_pos") is not None:
            ep_data["obj_init_pos"] = info["obj_init_pos"].tolist()
        if info.get("goal_pose") is not None:
            ep_data["goal_pose"] = info["goal_pose"].tolist()
        if "rand_vec" in info:
            ep_data["rand_vec"] = info["rand_vec"].tolist()
        metadata["episodes"].append(ep_data)

    metadata["num_episodes"] = len(metadata["episodes"])

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nEpisode初始环境信息已保存到: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(
        description="生成均匀分布在 reset random vector 空间上的 episode 数据集（task-aware）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--task", type=str, default="pick-place-v3")
    parser.add_argument("--num-episodes", type=int, default=100,
                        help="本次希望新增采集的 episode 数量（仅统计成功的 episode）")
    parser.add_argument("--output-dir", type=str, default="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pickplacev3")
    parser.add_argument("--repo-id", type=str, default="work2/metaworld_pick_place")
    parser.add_argument("--fps", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-attempts-per-episode", type=int, default=MAX_ATTEMPTS_PER_EPISODE,
                        help=f"每个 episode 的最大重试次数（默认: {MAX_ATTEMPTS_PER_EPISODE}）")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    is_resume = output_dir.exists() and (output_dir / "episode_initial_states.json").exists()

    print("=" * 80)
    print("Meta-World 均匀分布数据采集 (Task-Aware, Resume-Supported)")
    print("=" * 80)
    print(f"任务: {args.task}")
    print(f"目标新增Episode数量: {args.num_episodes}")
    print(f"输出目录: {args.output_dir}")
    print(f"Repo ID: {args.repo_id}")
    print(f"FPS: {args.fps}")
    print(f"图像分辨率: {args.image_size}x{args.image_size}")
    print(f"最大步数/episode: {args.max_steps}")
    print(f"模式: {'Resume (继续已有数据集)' if is_resume else '从头开始'}")
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
    print(f"  Has obj: {task_space_info['has_obj']} (dim={task_space_info['obj_pos_dim']})")
    print(f"  Has goal: {task_space_info['has_goal']} (dim={task_space_info['goal_pos_dim']})")
    print(f"  Rejection constraint: {task_space_info['rejection_constraint']}")
    print(f"  Space low: {task_space_info['low'].tolist()}")
    print(f"  Space high: {task_space_info['high'].tolist()}")

    existing_rand_vecs = []
    if is_resume:
        metadata_file = output_dir / "episode_initial_states.json"
        existing_rand_vecs = load_existing_states(metadata_file)
        print(f"已加载 {len(existing_rand_vecs)} 个已有 rand_vec 记录")

    print(f"\n生成 {args.num_episodes} 个均匀采样点 (Latin Hypercube Sampling)...")
    grid_centers = compute_uniform_samples(task_space_info, args.num_episodes, existing_rand_vecs if existing_rand_vecs else None)
    print(f"共 {len(grid_centers)} 个目标初始状态")

    if is_resume:
        print("\n加载已有LeRobot数据集...")
        dataset = load_existing_dataset(args.repo_id, args.output_dir)
        existing_episodes = dataset.num_episodes
        print(f"已有 {existing_episodes} 个 episode")
    else:
        features = {
            "observation.images.top": {
                "dtype": "image",
                "shape": (3, args.image_size, args.image_size),
                "names": ["channels", "height", "width"],
            },
            "observation.images.wrist": {
                "dtype": "image",
                "shape": (3, args.image_size, args.image_size),
                "names": ["channels", "height", "width"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (4,),
            },
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (39,),
                "names": ["keypoints"],
            },
            "action": {
                "dtype": "float32",
                "shape": (4,),
                "names": {"axes": ["x", "y", "z", "gripper"]},
            },
            "next.reward": {
                "dtype": "float32",
                "shape": (1,),
            },
            "next.success": {
                "dtype": "bool",
                "shape": (1,),
            },
        }
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            features=features,
            root=output_dir,
            robot_type="metaworld",
            use_videos=True,
        )
        existing_episodes = 0
        print(f"\n创建新LeRobot数据集: {output_dir}")

    episode_infos = []
    success_count = 0
    total_attempts = 0
    start_time = time.time()

    print(f"\n开始采集 {args.num_episodes} 个成功的 episode...")
    print("-" * 80)

    sample_idx = 0
    while success_count < args.num_episodes:
        if sample_idx >= len(grid_centers):
            print(f"\n预生成的采样点已用完，重新生成一批新的采样点...")
            new_samples = compute_uniform_samples(task_space_info, args.num_episodes - success_count, existing_rand_vecs)
            if not new_samples:
                print("错误: 无法生成新的不重复采样点，终止采集。")
                break
            grid_centers.extend(new_samples)
            print(f"新增 {len(new_samples)} 个采样点")

        rand_vec = grid_centers[sample_idx]
        sample_idx += 1

        attempt = 0
        episode_saved = False

        while attempt < args.max_attempts_per_episode:
            attempt += 1
            total_attempts += 1

            try:
                env_top = create_metaworld_env(args.task, seed=42, camera_name="corner3")
                env_wrist = create_metaworld_env(args.task, seed=42, camera_name="gripperPOV")

                set_task_state_directly(env_top, rand_vec)
                set_task_state_directly(env_wrist, rand_vec)

                env_top.reset()
                env_wrist.reset()
                sync_env_state(env_top, env_wrist)

                frames, ep_info = run_episode(
                    env_top, env_wrist, expert_policy, args.task,
                    args.max_steps, args.image_size, EXTRA_FRAMES_AFTER_SUCCESS
                )
                ep_info["rand_vec"] = rand_vec

                env_top.close()
                env_wrist.close()

            except RandVecExhaustedError as e:
                print(f"  Attempt {attempt}: SKIPPED (rand_vec rejected by task): {e}")
                rand_vec = None
                break

            if ep_info["success"]:
                for frame in frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                episode_infos.append(ep_info)
                success_count += 1
                existing_rand_vecs.append(rand_vec)
                episode_saved = True

                obj_str = ""
                if ep_info.get("obj_init_pos") is not None:
                    obj_str = f" | obj: [{ep_info['obj_init_pos'][0]:.3f}, {ep_info['obj_init_pos'][1]:.3f}, {ep_info['obj_init_pos'][2]:.3f}]"
                print(
                    f"Episode {success_count}/{args.num_episodes} | "
                    f"Attempt {attempt} | "
                    f"Frames: {ep_info['num_frames']:4d} | "
                    f"Success{obj_str}"
                )
                break
            else:
                print(f"  Attempt {attempt}: FAILED, retrying with new rand_vec...")
                rand_vec = None
                break

        if not episode_saved:
            new_vecs = compute_uniform_samples(task_space_info, 1, existing_rand_vecs)
            if new_vecs:
                grid_centers.append(new_vecs[0])
            else:
                print(f"  Warning: 无法生成新的不重复采样点，跳过该 episode。")

    print("\n" + "-" * 80)
    print("正在保存数据集...")
    dataset.finalize()

    save_episode_metadata(args.output_dir, episode_infos, args.task, task_space_info, append=is_resume)

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("采集完成！")
    print("=" * 80)
    print(f"本次新增成功Episode: {success_count}")
    print(f"总尝试次数: {total_attempts}")
    print(f"成功率: {success_count / max(total_attempts, 1) * 100:.1f}%")
    print(f"数据集总Episode: {dataset.num_episodes}")
    print(f"总用时: {total_time:.1f}s")
    print(f"数据集路径: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()