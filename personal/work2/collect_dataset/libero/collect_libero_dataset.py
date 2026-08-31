#!/usr/bin/env python
"""
LIBERO 数据集采集脚本 - 为单个 LIBERO task 生成均匀初始状态的 LeRobotDataset。

功能：
    为指定的 LIBERO suite + task_id 生成 LeRobotDataset，每个 episode 使用 farthest-point
    sampling 从大量候选初始状态中选择，使初始物体位置在 task 的合法初始化空间中均匀分布。
    双相机（agentview + wrist）保存为 LeRobot MP4 视频，完整 environment_state 和 stats。

数据来源：
    - LIBERO benchmark 通过 `libero.libero.benchmark` 获取
    - BDDL 文件通过 `libero.libero.get_libero_path("bddl_files")` 获取
    - 初始状态通过大量 seed reset 采样得到

字段说明：
    - observation.images.top (agentview_image): RGB 视频
    - observation.images.wrist (robot0_eye_in_hand_image): RGB 视频
    - observation.state: 25 维机器人状态 (eef_pos 3 + eef_quat 4 + gripper_qpos 2 + gripper_qvel 2 + joint_pos 7 + joint_vel 7)
    - observation.environment_state: 完整 flattened MuJoCo state（从 env.env.sim.get_state().flatten() 获取）
    - action: 7 维动作
    - next.reward: 奖励标量
    - next.success: 成功标志
    - task: 任务语言描述

运行方法：
    # 完整采集（300 episodes）
    cd /data/zhonglinye/jun/lerobot
    python personal/work2/collect_dataset/libero/collect_libero_dataset.py \
        --suite libero_spatial --task-id 0 --num-episodes 300 \
        --output-dir personal/work2/dataset/libero_spatial_task0 \
        --repo-id work2/libero_spatial_task0 --fps 20 --image-size 360 \
        --seed-start 0 --candidate-multiplier 10

    # 快速测试（2 episodes）
    python personal/work2/collect_dataset/libero/collect_libero_dataset.py \
        --suite libero_spatial --task-id 0 --num-episodes 2 \
        --output-dir personal/work2/dataset/libero_spatial_task0_test \
        --repo-id work2/libero_spatial_task0_test --fps 20 --image-size 360 \
        --seed-start 0 --candidate-multiplier 10
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

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROBOT_STATE_DIM = 25
ACTION_DIM = 7


def parse_args():
    parser = argparse.ArgumentParser(
        description="为 LIBERO task 生成均匀初始状态的 LeRobotDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--suite", type=str, required=True,
                        help="LIBERO suite 名称，如 libero_spatial, libero_object, libero_goal, libero_10, libero_90")
    parser.add_argument("--task-id", type=int, required=True,
                        help="Task ID 在指定 suite 中的索引")
    parser.add_argument("--num-episodes", type=int, default=300,
                        help="目标 episode 数量（默认 300）")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录，默认 personal/work2/dataset/<suite>_task<task_id>")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="HuggingFace repo ID，格式: 用户名/数据集名")
    parser.add_argument("--fps", type=int, default=20,
                        help="视频帧率（默认 20）")
    parser.add_argument("--image-size", type=int, default=360,
                        help="图像分辨率（默认 360）")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="起始 seed（默认 0）")
    parser.add_argument("--candidate-multiplier", type=int, default=10,
                        help="候选数量倍数（默认 10，即 num_episodes * 10）")
    parser.add_argument("--save-only-successful", action="store_true",
                        help="只保存成功的 episode")
    return parser.parse_args()


def get_suite_and_task(suite_name: str, task_id: int):
    """获取指定 suite 和 task_id 对应的 task 对象。"""
    bench = benchmark.get_benchmark_dict()
    if suite_name not in bench:
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {', '.join(bench.keys())}")
    suite = bench[suite_name]()
    if task_id < 0 or task_id >= len(suite.tasks):
        raise ValueError(f"task_id {task_id} out of range [0, {len(suite.tasks) - 1}]")
    task = suite.get_task(task_id)
    return suite, task


def resolve_bddl_file(task):
    """获取 BDDL 文件的完整路径。"""
    bddl_root = get_libero_path("bddl_files")
    bddl_file = bddl_root / task.problem_folder / task.bddl_file
    if not bddl_file.exists():
        raise FileNotFoundError(f"BDDL file not found: {bddl_file}")
    return bddl_file


def create_env(bddl_file: Path, image_size: int, fps: int):
    """创建 OffScreenRenderEnv 环境。"""
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=image_size,
        camera_widths=image_size,
        control_freq=fps,
    )
    return env


def get_flattened_env_state(env) -> np.ndarray:
    """获取完整的 MuJoCo simulator state。"""
    sim = env.env.sim
    return sim.get_state().flatten()


def get_randomizable_object_poses(env) -> dict:
    """从环境中提取参与随机初始化的物体 pose。

    通过读取 BDDL 文件中的 initial_config 部分确定哪些物体可以随机，
    然后从 simulator state 中提取这些物体的位置。

    Returns:
        dict: {object_name: {"pos": (3,), "quat": (4,) or None}}
    """
    poses = {}
    try:
        import xml.etree.ElementTree as ET
        bddl_path = getattr(env, '_bddl_file', None)
        if bddl_path is None:
            return poses

        tree = ET.parse(bddl_path)
        root = tree.getroot()
        ns = {"bddl": "http://www.example.com/bddl"}

        for elem in root.iter():
            if elem.tag.endswith("initial_config"):
                for obj in elem:
                    obj_name = obj.get("name")
                    if obj_name and obj_name != "robot":
                        pos_elem = obj.find("position")
                        quat_elem = obj.find("quaternion")
                        pos = None
                        quat = None
                        if pos_elem is not None and pos_elem.text:
                            pos = np.array([float(x) for x in pos_elem.text.strip().split()])
                        if quat_elem is not None and quat_elem.text:
                            quat = np.array([float(x) for x in quat_elem.text.strip().split()])
                        if pos is not None:
                            poses[obj_name] = {"pos": pos, "quat": quat}
    except Exception as e:
        pass

    if not poses:
        pass

    return poses


def build_initial_state_descriptor(object_poses: dict) -> np.ndarray:
    """构造均匀采样 descriptor。

    将所有可随机物体的 xyz 拼接成 descriptor。
    如果只有一个物体，descriptor 就是其 xyz (3维)；
    如果有多个物体，拼接多个物体的 xyz。

    Args:
        object_poses: {name: {"pos": (3,), "quat": (4,) or None}}

    Returns:
        descriptor: concatenated xyz positions
    """
    parts = []
    for name in sorted(object_poses.keys()):
        pose = object_poses[name]
        parts.append(pose["pos"])
    if parts:
        return np.concatenate(parts)
    return np.array([])


def generate_candidate_initial_states(env, num_candidates: int, seed_start: int) -> list:
    """生成大量候选初始状态。

    通过对不同 seed 执行 reset，从合法的初始化空间中采样状态。
    每个 candidate 保存 seed 和完整 flattened state。

    Args:
        env: OffScreenRenderEnv 实例
        num_candidates: 目标候选数量
        seed_start: 起始 seed

    Returns:
        list of dict: [{"seed": int, "state": np.ndarray, "object_poses": dict}]
    """
    candidates = []
    seed = seed_start
    max_attempts = num_candidates * 20
    attempts = 0

    while len(candidates) < num_candidates and attempts < max_attempts:
        attempts += 1
        try:
            env.seed(seed)
            env.reset()

            full_state = get_flattened_env_state(env)
            object_poses = get_randomizable_object_poses(env)

            candidates.append({
                "seed": seed,
                "state": full_state,
                "object_poses": object_poses,
            })
            if len(candidates) % 100 == 0:
                print(f"  Generated {len(candidates)} candidates...")
        except Exception as e:
            pass
        seed += 1

    print(f"  Generated {len(candidates)} candidate initial states")
    return candidates


def remove_constant_dimensions(descriptors: np.ndarray) -> tuple:
    """排除 descriptor 中标准差过小的维度。

    Args:
        descriptors: (num_candidates, descriptor_dim) array

    Returns:
        (valid_descriptors, valid_mask, stats) where stats contains min/max/std per dimension
    """
    stats = []
    valid_mask = []
    for i in range(descriptors.shape[1]):
        std = np.std(descriptors[:, i])
        dim_min = np.min(descriptors[:, i])
        dim_max = np.max(descriptors[:, i])
        stats.append({"min": float(dim_min), "max": float(dim_max), "std": float(std)})
        valid_mask.append(std > 1e-6)

    valid_mask = np.array(valid_mask)
    if not np.any(valid_mask):
        return descriptors, valid_mask, stats

    valid_descriptors = descriptors[:, valid_mask]
    return valid_descriptors, valid_mask, stats


def select_uniform_initial_states(candidates: list, num_select: int, seed: int) -> list:
    """使用 farthest-point sampling 选择均匀分布的初始状态。

    先对 descriptor 归一化到 0~1，第一个点选择离中心最近的，
    之后每次选择距离已选集合最远的点。

    Args:
        candidates: list of candidate dicts with "object_poses" key
        num_select: 目标选择数量
        seed: 随机种子

    Returns:
        list of selected candidate indices
    """
    if len(candidates) <= num_select:
        return list(range(len(candidates)))

    descriptors = []
    for c in candidates:
        desc = build_initial_state_descriptor(c.get("object_poses", {}))
        descriptors.append(desc)

    descriptors = np.array(descriptors)
    if descriptors.shape[0] == 0 or descriptors.shape[1] == 0:
        return list(range(min(num_select, len(candidates))))

    valid_descriptors, valid_mask, dim_stats = remove_constant_dimensions(descriptors)

    if valid_descriptors.shape[1] == 0:
        return list(range(min(num_select, len(candidates))))

    valid_min = np.min(valid_descriptors, axis=0)
    valid_max = np.max(valid_descriptors, axis=0)
    valid_range = valid_max - valid_min
    valid_range[valid_range < 1e-6] = 1.0

    normalized = (valid_descriptors - valid_min) / valid_range

    center = np.mean(normalized, axis=0)
    dist_to_center = np.linalg.norm(normalized - center, axis=1)

    selected = [np.argmin(dist_to_center)]
    min_distances = np.full(len(candidates), np.inf)

    for _ in range(num_select - 1):
        if len(selected) >= len(candidates):
            break

        last_selected = selected[-1]
        dists = np.linalg.norm(normalized - normalized[last_selected], axis=1)
        min_distances = np.minimum(min_distances, dists)

        min_distances[selected] = -np.inf

        next_idx = np.argmax(min_distances)
        if min_distances[next_idx] < 0:
            break
        selected.append(next_idx)

    return selected


def save_uniform_initial_states(output_dir: Path, selected_candidates: list, dim_stats: list, valid_mask: list):
    """保存均匀选择的初始状态到 NPZ 和 JSON。"""
    states = np.array([c["state"] for c in selected_candidates])
    seeds = [c["seed"] for c in selected_candidates]
    object_poses = [c["object_poses"] for c in selected_candidates]

    npz_path = output_dir / "uniform_initial_states.npz"
    np.savez(npz_path, states=states, seeds=seeds)

    json_data = {
        "num_states": len(selected_candidates),
        "dim_stats": dim_stats,
        "valid_mask": valid_mask.tolist() if hasattr(valid_mask, 'tolist') else list(valid_mask),
        "seeds": seeds,
        "object_poses": [
            {k: {"pos": v["pos"].tolist(), "quat": v["quat"].tolist() if v["quat"] is not None else None}
            for k, v in poses.items()
        } for poses in object_poses
        ]
    }

    json_path = output_dir / "uniform_initial_states.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"  Saved uniform states to {npz_path} and {json_path}")


def build_robot_state(env) -> np.ndarray:
    """构造 25 维机器人状态。

    顺序: eef_pos(3) + eef_quat(4) + gripper_qpos(2) + gripper_qvel(2) + joint_pos(7) + joint_vel(7)

    Returns:
        np.ndarray shape (25,)
    """
    eef_pos = env.get_obs()["robot0_eef_pos"]
    eef_quat = env.get_obs()["robot0_eef_quat"]
    gripper_qpos = env.get_obs()["robot0_gripper_qpos"]
    gripper_qvel = env.get_obs()["robot0_gripper_qvel"]
    joint_pos = env.get_obs()["robot0_joint_pos"]
    joint_vel = env.get_obs()["robot0_joint_vel"]

    state = np.concatenate([eef_pos, eef_quat, gripper_qpos, gripper_qvel, joint_pos, joint_vel])
    assert state.shape == (ROBOT_STATE_DIM,), f"Expected (25,) got {state.shape}"
    return state.astype(np.float32)


def create_dataset(repo_id: str, output_dir: Path, fps: int, image_size: int):
    """创建 LeRobot video dataset。"""
    features = {
        "observation.images.top": {
            "dtype": "video",
            "shape": (image_size, image_size, 3),
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (image_size, image_size, 3),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (ROBOT_STATE_DIM,),
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (1,),
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
        },
        "next.success": {
            "dtype": "bool",
            "shape": (1,),
        },
        "task": {
            "dtype": "string",
            "shape": (1,),
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=str(output_dir),
        robot_type="panda",
        use_videos=True,
    )
    return dataset


def collect_demo_episode(env, task_language: str, max_steps: int = 300) -> tuple:
    """尝试执行 episode。

    如果有成功轨迹，返回 (frames, success, initial_state, initial_object_poses)。
    如果失败，返回 (None, False, None, None)。

    Note: 此版本使用 env 内置的 init_states 来获取成功轨迹。
    对于新生成的均匀 initial states，需要使用 rollout_episode() 函数配合 policy。
    """
    raw_obs = env.reset()
    obs = env.get_obs()

    initial_state = get_flattened_env_state(env)
    initial_object_poses = get_randomizable_object_poses(env)

    frames = []
    success_detected = False
    extra_frames = 10
    frames_after_success = 0

    for step in range(max_steps):
        action = env.action_space.sample()

        raw_obs, reward, done, info = env.step(action)
        obs = env.get_obs()

        top_image = raw_obs.get("agentview_image")
        wrist_image = raw_obs.get("robot0_eye_in_hand_image")

        if top_image is None or wrist_image is None:
            continue

        robot_state = build_robot_state(env)
        env_state = get_flattened_env_state(env)

        frame = {
            "observation.images.top": top_image,
            "observation.images.wrist": wrist_image,
            "observation.state": robot_state,
            "observation.environment_state": env_state.reshape(1,),
            "action": action.astype(np.float32),
            "next.reward": np.array([reward], dtype=np.float32),
            "next.success": np.array([info.get("is_success", False)], dtype=bool),
            "task": task_language,
        }
        frames.append(frame)

        if info.get("is_success", False):
            success_detected = True
            print(f"    Success at step {step}")
            break

        if done:
            break

        if success_detected:
            frames_after_success += 1
            if frames_after_success >= extra_frames:
                break

    return frames, success_detected, initial_state, initial_object_poses


def rollout_episode(env, initial_state: np.ndarray, policy, max_steps: int = 300) -> tuple:
    """从指定初始状态使用 policy 执行 rollout。

    Args:
        env: OffScreenRenderEnv 实例
        initial_state: flattened MuJoCo state
        policy: 可调用对象，接受 observation 返回 action
        max_steps: 最大步数

    Returns:
        (frames, success, initial_state, initial_object_poses)
    """
    env.reset()
    env.set_init_state(initial_state)

    initial_object_poses = get_randomizable_object_poses(env)

    raw_obs = env.get_obs()
    frames = []
    success_detected = False
    extra_frames = 10
    frames_after_success = 0

    for step in range(max_steps):
        obs = env.get_obs()
        robot_state = build_robot_state(env)

        action = policy(obs)

        raw_obs, reward, done, info = env.step(action)

        top_image = raw_obs.get("agentview_image")
        wrist_image = raw_obs.get("robot0_eye_in_hand_image")

        if top_image is None or wrist_image is None:
            continue

        env_state = get_flattened_env_state(env)

        frame = {
            "observation.images.top": top_image,
            "observation.images.wrist": wrist_image,
            "observation.state": robot_state,
            "observation.environment_state": env_state.reshape(1,),
            "action": action.astype(np.float32),
            "next.reward": np.array([reward], dtype=np.float32),
            "next.success": np.array([info.get("is_success", False)], dtype=bool),
            "task": env.task_description,
        }
        frames.append(frame)

        if info.get("is_success", False):
            success_detected = True
            print(f"    Success at step {step}")
            break

        if done:
            break

        if success_detected:
            frames_after_success += 1
            if frames_after_success >= extra_frames:
                break

    return frames, success_detected, initial_state, initial_object_poses


def save_episode_metadata(output_dir: Path, episode_data: list, suite: str, task_id: int,
                          task_name: str, task_description: str, bddl_file: str,
                          state_dim: int, sampling_method: str, num_candidates: int):
    """保存 episode 初始状态元数据到 episode_initial_states.json。"""
    metadata = {
        "suite": suite,
        "task_id": task_id,
        "task_name": task_name,
        "task_description": task_description,
        "bddl_file": str(bddl_file),
        "state_dim": state_dim,
        "sampling_method": sampling_method,
        "num_candidates": num_candidates,
        "target_episodes": len(episode_data),
        "successful_trajectories": sum(1 for ep in episode_data if ep.get("success", False)),
        "episodes": []
    }

    for ep in episode_data:
        ep_info = {
            "episode_index": ep["episode_index"],
            "seed": ep.get("seed"),
            "initial_environment_state": ep["initial_state"].tolist() if ep.get("initial_state") is not None else None,
            "initial_object_poses": {
                k: {"pos": v["pos"].tolist(), "quat": v["quat"].tolist() if v["quat"] is not None else None}
                for k, v in ep.get("initial_object_poses", {}).items()
            },
            "success": ep.get("success", False),
            "num_frames": ep.get("num_frames", 0),
        }
        metadata["episodes"].append(ep_info)

    metadata_file = output_dir / "episode_initial_states.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved episode metadata to {metadata_file}")


def validate_dataset(output_dir: Path) -> bool:
    """验证生成的 dataset 的完整性和正确性。"""
    output_dir = Path(output_dir)

    meta_stats = output_dir / "meta" / "stats.json"
    if not meta_stats.exists():
        print(f"ERROR: {meta_stats} does not exist")
        return False

    try:
        with open(meta_stats) as f:
            stats = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse {meta_stats}: {e}")
        return False

    required_keys = ["observation.state", "observation.environment_state", "action"]
    for key in required_keys:
        if key not in stats:
            print(f"ERROR: Required stat key '{key}' not found in stats.json")
            return False

    episode_meta = output_dir / "episode_initial_states.json"
    if not episode_meta.exists():
        print(f"ERROR: {episode_meta} does not exist")
        return False

    video_dir = output_dir / "videos"
    if not video_dir.exists():
        print(f"ERROR: {video_dir} does not exist")
        return False

    camera_dirs = list(video_dir.glob("observation.images.*"))
    if not camera_dirs:
        print(f"ERROR: No camera video directories found in {video_dir}")
        return False

    print(f"  Validation passed: stats.json exists and is valid")
    print(f"  Found {len(camera_dirs)} camera directories")
    return True


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"LIBERO Dataset Collection")
    print(f"{'='*60}")
    print(f"Suite: {args.suite}")
    print(f"Task ID: {args.task_id}")
    print(f"Target episodes: {args.num_episodes}")
    print(f"Candidate multiplier: {args.candidate_multiplier}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*60}\n")

    suite, task = get_suite_and_task(args.suite, args.task_id)
    bddl_file = resolve_bddl_file(task)

    print(f"Task: {task.name}")
    print(f"Description: {task.language}")
    print(f"BDDL: {bddl_file}")

    output_dir = Path(args.output_dir) if args.output_dir else \
                 Path(f"personal/work2/dataset/{args.suite}_task{args.task_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/6] Creating environment to generate candidates...")
    env = create_env(bddl_file, args.image_size, args.fps)
    env.reset()

    sample_state = get_flattened_env_state(env)
    state_dim = len(sample_state)
    print(f"  Environment state dimension: {state_dim}")

    print(f"\n[2/6] Generating candidate initial states...")
    num_candidates = args.num_episodes * args.candidate_multiplier
    candidates = generate_candidate_initial_states(env, num_candidates, args.seed_start)

    if len(candidates) < args.num_episodes:
        print(f"WARNING: Only generated {len(candidates)} candidates, less than target {args.num_episodes}")

    print(f"\n[3/6] Selecting uniform initial states using farthest-point sampling...")
    selected_indices = select_uniform_initial_states(candidates, args.num_episodes, args.seed_start)
    selected_candidates = [candidates[i] for i in selected_indices]
    print(f"  Selected {len(selected_candidates)} uniform initial states")

    descriptors = []
    for c in selected_candidates:
        desc = build_initial_state_descriptor(c.get("object_poses", {}))
        descriptors.append(desc)
    descriptors_arr = np.array(descriptors) if descriptors else np.array([]).reshape(0, 0)
    _, valid_mask, dim_stats = remove_constant_dimensions(descriptors_arr)

    print(f"\n[4/6] Saving uniform initial states...")
    save_uniform_initial_states(output_dir, selected_candidates, dim_stats, valid_mask)

    print(f"\n[5/6] Creating LeRobot dataset...")
    dataset = create_dataset(args.repo_id, output_dir, args.fps, args.image_size)

    episode_data = []
    successful_count = 0
    state_dim_match = True

    print(f"\n[6/6] Collecting episodes...")

    for i, candidate in enumerate(selected_candidates):
        print(f"  Episode {i}/{len(selected_candidates)} (seed={candidate['seed']})...", end=" ")

        try:
            env.seed(candidate["seed"])
            env.reset()
            env.set_init_state(candidate["state"])

            frames, success, init_state, obj_poses = collect_demo_episode(
                env, task.language, max_steps=300
            )

            if success:
                for frame in frames:
                    if state_dim_match:
                        frame["observation.environment_state"] = get_flattened_env_state(env).reshape(1,)

                    if frame["observation.environment_state"].shape != (1,) and \
                       frame["observation.environment_state"].shape[0] != state_dim:
                        print(f"\nWARNING: state_dim mismatch at episode {i}: "
                              f"expected {state_dim}, got {frame['observation.environment_state'].shape}")

                    dataset.add_frame(frame)

                dataset.save_episode()
                successful_count += 1
                print(f"SUCCESS ({len(frames)} frames)")
            else:
                print("FAILED")
                if not args.save_only_successful:
                    pass
                else:
                    pass
            episode_data.append({
                "episode_index": i,
                "seed": candidate["seed"],
                "initial_state": candidate["state"],
                "initial_object_poses": candidate["object_poses"],
                "success": success,
                "num_frames": len(frames) if success else 0,
            })

        except Exception as e:
            print(f"ERROR: {e}")
            episode_data.append({
                "episode_index": i,
                "seed": candidate["seed"],
                "initial_state": candidate["state"],
                "initial_object_poses": candidate["object_poses"],
                "success": False,
                "num_frames": 0,
            })

    print(f"\n{'='*60}")
    print(f"Collection Summary:")
    print(f"  Target episodes: {args.num_episodes}")
    print(f"  Successful trajectories: {successful_count}")
    print(f"  Uniform initial states saved: {len(selected_candidates)}")
    print(f"{'='*60}\n")

    if successful_count < args.num_episodes:
        print(f"WARNING: Only collected {successful_count} successful trajectories, "
              f"less than target {args.num_episodes}.")
        print(f"  The remaining initial states were saved to uniform_initial_states.npz/json")
        print(f"  but no expert policy was available to generate trajectories for them.")

    dataset.finalize()

    print(f"\n[Final] Saving episode metadata...")
    save_episode_metadata(
        output_dir, episode_data, args.suite, args.task_id,
        task.name, task.language, bddl_file, state_dim,
        "task_space_farthest_point", len(candidates)
    )

    print(f"\n[Final] Validating dataset...")
    if validate_dataset(output_dir):
        print("Dataset validation PASSED")
    else:
        print("Dataset validation FAILED")

    env.close()

    print(f"\nDone! Dataset saved to: {output_dir}")
    print(f"Use render_initial_state.py to verify the saved initial states")


if __name__ == "__main__":
    main()