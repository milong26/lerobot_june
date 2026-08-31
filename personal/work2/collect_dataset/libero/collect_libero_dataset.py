#!/usr/bin/env python
"""
LIBERO 数据集生成代码 - 为单个 LIBERO task 生成约 300 个 episode 的 LeRobotDataset。

功能说明：
    本脚本为指定的 LIBERO task 生成均匀初始状态采样的 LeRobotDataset。
    支持双相机视频保存、完整 environment_state、episode 初始状态和正确 stats。

数据来源：
    - LIBERO 官方 demonstrations (HDF5 格式) 作为成功轨迹来源
    - 通过大量 seed reset 生成均匀初始状态候选

数据字段：
    - observation.images.top: agentview 相机视频 (agentview_image)
    - observation.images.wrist: wrist 相机视频 (robot0_eye_in_hand_image)
    - observation.state: 25维机械臂状态 (eef_pos(3)+eef_quat(4)+gripper_qpos(2)+gripper_qvel(2)+joint_pos(7)+joint_vel(7))
    - observation.environment_state: 完整 flattened MuJoCo state
    - action: 7维动作
    - next.reward, next.success: 奖励和成功标志
    - task: 任务描述

运行方法：
    完整采集 300 episode:
    cd /data/zhonglinye/jun/lerobot
    python personal/work2/collect_dataset/libero/collect_libero_dataset.py \
        --suite libero_spatial --task-id 0 --num-episodes 300 \
        --output-dir personal/work2/dataset/libero_spatial_task0 \
        --repo-id work2/libero_spatial_task0 --fps 20 --image-size 360 \
        --seed-start 0 --candidate-multiplier 10

    快速测试 2 episode:
    cd /data/zhonglinye/jun/lerobot
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

import h5py
import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from lerobot.datasets.lerobot_dataset import LeRobotDataset

LIBERO_ACTION_DIM = 7
LIBERO_DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]

CAMERA_NAME_MAPPING = {
    "agentview_image": "image",
    "robot0_eye_in_hand_image": "image2",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="为 LIBERO task 生成均匀初始状态采样的 LeRobotDataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--suite", type=str, required=True,
                        help="LIBERO suite 名称，如 libero_spatial, libero_object, libero_goal, libero_10, libero_90")
    parser.add_argument("--task-id", type=int, required=True,
                        help="Task ID (0-indexed)")
    parser.add_argument("--num-episodes", type=int, default=300,
                        help="目标 episode 数量 (默认 300)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录路径 (默认 personal/work2/dataset/<suite>_task<task_id>)")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="HuggingFace repo ID，格式: 用户名/数据集名")
    parser.add_argument("--fps", type=int, default=20,
                        help="视频帧率 (默认 20)")
    parser.add_argument("--image-size", type=int, default=360,
                        help="图像分辨率 (默认 360)")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="随机种子起始值 (默认 0)")
    parser.add_argument("--candidate-multiplier", type=int, default=10,
                        help="candidate 数量 = num_episodes * candidate_multiplier (默认 10)")
    parser.add_argument("--save-only-successful", action="store_true",
                        help="只保存成功的 episode")
    return parser.parse_args()


def get_suite_and_task(suite_name: str, task_id: int):
    """
    获取 LIBERO suite 和 task 对象。

    Args:
        suite_name: suite 名称
        task_id: task ID

    Returns:
        suite: LIBERO Benchmark 对象
        task: LIBERO Task 对象
    """
    bench_dict = benchmark.get_benchmark_dict()
    if suite_name not in bench_dict:
        raise ValueError(f"Unknown suite '{suite_name}'. Available: {', '.join(sorted(bench_dict.keys()))}")
    suite = bench_dict[suite_name]()
    if task_id < 0 or task_id >= len(suite.tasks):
        raise ValueError(f"task_id {task_id} out of range [0, {len(suite.tasks) - 1}]")
    task = suite.get_task(task_id)
    return suite, task


def resolve_bddl_file(task):
    """
    获取 BDDL 文件路径。

    Args:
        task: LIBERO Task 对象

    Returns:
        Path: BDDL 文件的完整路径
    """
    bddl_path = get_libero_path("bddl_files") / task.problem_folder / task.bddl_file
    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL file not found: {bddl_path}")
    return bddl_path


def create_env(bddl_file_path: str, image_size: int = 360, control_freq: int = 20):
    """
    创建 OffScreenRenderEnv 环境。

    Args:
        bddl_file_path: BDDL 文件路径
        image_size: 图像分辨率
        control_freq: 控制频率

    Returns:
        env: OffScreenRenderEnv 实例
    """
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file_path),
        camera_heights=image_size,
        camera_widths=image_size,
        control_freq=control_freq,
    )
    return env


def get_flattened_env_state(env) -> np.ndarray:
    """
    获取完整的 flattened MuJoCo simulator state。

    Args:
        env: OffScreenRenderEnv 实例

    Returns:
        flattened state as numpy array
    """
    sim = env.env.sim
    qpos = sim.data.qpos.copy()
    qvel = sim.data.qvel.copy()
    act = sim.data.act.copy() if sim.data.act is not None else np.array([])
    return np.concatenate([qpos, qvel, act])


def get_randomizable_object_poses(env, bddl_file_path: str) -> dict:
    """
    从当前环境和 BDDL 文件中提取参与随机初始化的物体 pose。

    Args:
        env: OffScreenRenderEnv 实例
        bddl_file_path: BDDL 文件路径

    Returns:
        dict: 物体名称到 pose 的映射
    """
    poses = {}

    bddl_content = Path(bddl_file_path).read_text()

    import re
    init_state_pattern = re.compile(r'init\s*\((.*?)\)', re.DOTALL)
    matches = init_state_pattern.findall(bddl_content)

    if not matches:
        return poses

    geom_ids = []
    for match in matches:
        parts = match.strip().split(',')
        if len(parts) >= 4:
            try:
                geom_name = parts[0].strip()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                if geom_name not in poses:
                    poses[geom_name] = []
                poses[geom_name].append(np.array([x, y, z]))
            except (ValueError, IndexError):
                continue

    final_poses = {}
    for name, pos_list in poses.items():
        if len(pos_list) > 0:
            final_poses[name] = np.mean(pos_list, axis=0)

    return final_poses


def build_initial_state_descriptor(object_poses: dict, env_state: np.ndarray) -> np.ndarray:
    """
    构建用于均匀采样的 descriptor。

    将所有可随机物体的 xyz 拼接成 descriptor。

    Args:
        object_poses: 物体名称到 pose 的映射
        env_state: 完整的 environment state (未使用，保留接口兼容性)

    Returns:
        descriptor array
    """
    if not object_poses:
        return np.array([])

    parts = []
    for name in sorted(object_poses.keys()):
        pose = object_poses[name]
        if len(pose) >= 3:
            parts.append(pose[:3])
        elif len(pose) == 1:
            parts.append(pose)

    if not parts:
        return np.array([])

    return np.concatenate(parts)


def generate_candidate_initial_states(
    env, bddl_file_path: str, num_candidates: int, seed_start: int = 0
) -> list:
    """
    通过大量不同 seed reset 生成合法的 initial states。

    Args:
        env: OffScreenRenderEnv 实例
        bddl_file_path: BDDL 文件路径
        num_candidates: 需要生成的 candidate 数量
        seed_start: 起始 seed

    Returns:
        list of dict, 每个 dict 包含:
            - seed: 随机种子
            - state: 完整的 flattened MuJoCo state
            - object_poses: 物体初始 pose
            - descriptor: 用于均匀采样的 descriptor
    """
    candidates = []
    env.reset()

    for i in range(num_candidates):
        seed = seed_start + i
        env.seed(seed)
        env.reset()

        for _ in range(10):
            env.step(LIBERO_DUMMY_ACTION)

        full_state = get_flattened_env_state(env)
        object_poses = get_randomizable_object_poses(env, bddl_file_path)
        descriptor = build_initial_state_descriptor(object_poses, full_state)

        candidates.append({
            "seed": seed,
            "state": full_state,
            "object_poses": object_poses,
            "descriptor": descriptor,
        })

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{num_candidates} candidates")

    return candidates


def remove_constant_dimensions(descriptors: np.ndarray) -> tuple:
    """
    移除 descriptor 中标准差过小的维度。

    Args:
        descriptors: shape (num_candidates, descriptor_dim)

    Returns:
        (valid_descriptors, valid_dims_mask)
        - valid_descriptors: 移除常数列后的 descriptors
        - valid_dims_mask: 布尔数组，标记哪些维度是有效的
    """
    std_vals = np.std(descriptors, axis=0)
    valid_dims = std_vals > 1e-6
    valid_descriptors = descriptors[:, valid_dims] if valid_descriptors := descriptors[:, valid_dims] else descriptors
    return valid_descriptors, valid_dims


def select_uniform_initial_states(candidates: list, target_num: int) -> list:
    """
    使用 farthest-point sampling 选择均匀分布的 initial states。

    Args:
        candidates: candidate 列表
        target_num: 目标选择数量

    Returns:
        选中的 candidate 索引列表
    """
    if len(candidates) <= target_num:
        return list(range(len(candidates)))

    descriptors = np.array([c["descriptor"] for c in candidates])

    if descriptors.shape[1] == 0:
        return list(range(target_num))

    valid_descriptors, valid_dims_mask = remove_constant_dimensions(descriptors)

    if valid_descriptors.shape[1] == 0:
        return list(range(target_num))

    min_vals = np.min(valid_descriptors, axis=0)
    max_vals = np.max(valid_descriptors, axis=0)
    range_vals = max_vals - min_vals
    range_vals = np.where(range_vals < 1e-10, 1.0, range_vals)

    normalized = (valid_descriptors - min_vals) / range_vals

    center = np.mean(normalized, axis=0)
    dists_to_center = np.linalg.norm(normalized - center, axis=1)

    selected_indices = [int(np.argmin(dists_to_center))]

    for _ in range(target_num - 1):
        if len(selected_indices) >= len(candidates):
            break

        selected_descriptors = normalized[selected_indices]
        dists = np.linalg.norm(normalized - selected_descriptors[:, np.newaxis], axis=2)
        min_dists_to_selected = np.min(dists, axis=0)

        for idx in selected_indices:
            min_dists_to_selected[idx] = -np.inf

        next_idx = int(np.argmax(min_dists_to_selected))
        selected_indices.append(next_idx)

    return selected_indices


def save_uniform_initial_states(output_dir: Path, selected_candidates: list, all_candidates: list):
    """
    保存均匀初始状态到 NPZ 和 JSON 文件。

    Args:
        output_dir: 输出目录
        selected_candidates: 选中的 candidate 列表
        all_candidates: 所有 candidate 列表 (用于索引)
    """
    states = np.array([c["state"] for c in selected_candidates])
    npz_path = output_dir / "uniform_initial_states.npz"
    np.savez(npz_path, states=states)
    print(f"Saved uniform initial states to {npz_path}")

    json_data = {
        "num_selected": len(selected_candidates),
        "num_total_candidates": len(all_candidates),
        "episodes": []
    }

    for ep_idx, cand in enumerate(selected_candidates):
        ep_data = {
            "episode_index": ep_idx,
            "seed": int(cand["seed"]),
            "object_poses": {k: v.tolist() if isinstance(v, np.ndarray) else v
                           for k, v in cand["object_poses"].items()},
            "descriptor": cand["descriptor"].tolist() if isinstance(cand["descriptor"], np.ndarray) else [],
        }
        json_data["episodes"].append(ep_data)

    json_path = output_dir / "uniform_initial_states.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved uniform initial states metadata to {json_path}")


def build_robot_state(env) -> np.ndarray:
    """
    构建 25 维机械臂状态。

    顺序: eef_pos(3) + eef_quat(4) + gripper_qpos(2) + gripper_qvel(2) + joint_pos(7) + joint_vel(7)

    Args:
        env: OffScreenRenderEnv 实例

    Returns:
        25 维 numpy array
    """
    raw_obs = env.env._get_observations()

    eef_pos = raw_obs.get("robot0_eef_pos")
    eef_quat = raw_obs.get("robot0_eef_quat")
    gripper_qpos = raw_obs.get("robot0_gripper_qpos")
    gripper_qvel = raw_obs.get("robot0_gripper_qvel")
    joint_pos = raw_obs.get("robot0_joint_pos")
    joint_vel = raw_obs.get("robot0_joint_vel")

    if eef_pos is None or eef_quat is None or gripper_qpos is None or joint_pos is None:
        raise ValueError(f"Missing robot state fields. "
                        f"eef_pos={eef_pos is not None}, eef_quat={eef_quat is not None}, "
                        f"gripper_qpos={gripper_qpos is not None}, joint_pos={joint_pos is not None}")

    state = np.concatenate([
        eef_pos,
        eef_quat,
        gripper_qpos,
        gripper_qvel if gripper_qvel is not None else np.zeros(2),
        joint_pos,
        joint_vel if joint_vel is not None else np.zeros(7),
    ])

    assert state.shape == (25,), f"Expected state shape (25,), got {state.shape}"
    return state.astype(np.float32)


def create_dataset(
    repo_id: str,
    output_dir: Path,
    fps: int,
    image_size: int,
    state_dim: int,
):
    """
    创建 LeRobotDataset。

    Args:
        repo_id: HuggingFace repo ID
        output_dir: 本地保存路径
        fps: 帧率
        image_size: 图像分辨率
        state_dim: environment_state 维度

    Returns:
        LeRobotDataset 实例
    """
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
            "shape": (25,),
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (state_dim,),
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
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
        use_videos=True,
    )

    return dataset


def collect_demo_episode(
    env,
    demo_states: np.ndarray,
    demo_actions: np.ndarray,
    task_language: str,
    image_size: int,
    state_dim: int,
):
    """
    通过 state replay 收集一个 episode。

    使用官方 demonstration 的 states 和 actions，通过 state replay 生成轨迹。

    Args:
        env: OffScreenRenderEnv 实例
        demo_states: demonstration states (T, state_dim)
        demo_actions: demonstration actions (T, 7)
        task_language: 任务描述
        image_size: 图像分辨率
        state_dim: environment_state 维度

    Returns:
        (frames, success) - 帧列表和是否成功
    """
    frames = []
    success = False

    for t in range(len(demo_states)):
        state = demo_states[t]
        env.env.sim.set_state_from_flatten(state)

        for _ in range(3):
            env.step(LIBERO_DUMMY_ACTION)

        raw_obs = env.env._get_observations()

        top_image = raw_obs.get("agentview_image")
        wrist_image = raw_obs.get("robot0_eye_in_hand_image")

        if top_image is not None:
            top_image = top_image[::-1, ::-1]
        if wrist_image is not None:
            wrist_image = wrist_image[::-1, ::-1]

        if top_image is None or wrist_image is None:
            continue

        robot_state = build_robot_state(env)
        full_env_state = get_flattened_env_state(env)

        if full_env_state.shape[0] != state_dim:
            continue

        action = demo_actions[t] if t < len(demo_actions) else np.zeros(7)

        frame = {
            "observation.images.top": top_image,
            "observation.images.wrist": wrist_image,
            "observation.state": robot_state,
            "observation.environment_state": full_env_state.astype(np.float32),
            "action": action.astype(np.float32),
            "next.reward": np.array([0.0], dtype=np.float32),
            "next.success": np.array([False], dtype=bool),
            "task": task_language,
        }
        frames.append(frame)

    if frames:
        last_frame = frames[-1]
        env.env.sim.set_state_from_flatten(demo_states[-1])
        for _ in range(10):
            env.step(LIBERO_DUMMY_ACTION)
        success = env.check_success()

    return frames, success


def get_demo_data(task):
    """
    获取 LIBERO 官方 demonstration 数据。

    Args:
        task: LIBERO Task 对象

    Returns:
        (demo_states, demo_actions) - states 和 actions
    """
    demos_path = get_libero_path("demos")
    demo_dir = demos_path / task.problem_folder

    if not demo_dir.exists():
        return None, None

    hdf5_files = list(demo_dir.glob("*.h5"))
    if not hdf5_files:
        return None, None

    demo_file = hdf5_files[0]
    try:
        with h5py.File(demo_file, "r") as f:
            if "data/demo_0/states" in f:
                demo_states = f["data/demo_0/states"][:]
                demo_actions = f["data/demo_0/actions"][:]
                return demo_states, demo_actions
    except Exception as e:
        print(f"Warning: Could not read demo file {demo_file}: {e}")

    return None, None


def save_episode_metadata(
    output_dir: Path,
    suite_name: str,
    task_id: int,
    task_name: str,
    task_language: str,
    bddl_file: str,
    state_dim: int,
    episode_initial_states: list,
    num_successful: int,
    descriptor_info: dict,
):
    """
    保存 episode 初始状态元数据。

    Args:
        output_dir: 输出目录
        suite_name: suite 名称
        task_id: task ID
        task_name: task 名称
        task_language: 任务描述
        bddl_file: BDDL 文件路径
        state_dim: environment_state 维度
        episode_initial_states: 每个 episode 的初始状态信息
        num_successful: 成功 episode 数量
        descriptor_info: descriptor 维度信息
    """
    metadata = {
        "suite": suite_name,
        "task_id": task_id,
        "task_name": task_name,
        "task_description": task_language,
        "bddl_file": str(bddl_file),
        "state_dim": state_dim,
        "sampling_method": "task_space_farthest_point",
        "num_candidates_generated": descriptor_info.get("num_candidates", 0),
        "num_episodes_target": len(episode_initial_states),
        "num_successful_trajectories": num_successful,
        "descriptor_dim": descriptor_info.get("dim", 0),
        "descriptor_object_names": descriptor_info.get("object_names", []),
        "descriptor_min": descriptor_info.get("min", []),
        "descriptor_max": descriptor_info.get("max", []),
        "descriptor_std": descriptor_info.get("std", []),
        "episodes": [],
    }

    for ep_data in episode_initial_states:
        ep_meta = {
            "episode_index": ep_data["episode_index"],
            "seed": int(ep_data["seed"]),
            "initial_environment_state": ep_data["state"].tolist() if isinstance(ep_data["state"], np.ndarray) else ep_data["state"],
            "initial_object_poses": {k: v.tolist() if isinstance(v, np.ndarray) else v
                                    for k, v in ep_data.get("object_poses", {}).items()},
            "success": ep_data.get("success", False),
            "num_frames": ep_data.get("num_frames", 0),
        }
        metadata["episodes"].append(ep_meta)

    json_path = output_dir / "episode_initial_states.json"
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved episode metadata to {json_path}")


def validate_dataset(output_dir: Path):
    """
    验证生成的数据集。

    Args:
        output_dir: 数据集目录
    """
    print("\n=== Validating dataset ===")

    stats_path = output_dir / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        print(f"stats.json exists and is valid JSON")
        required_keys = ["observation.state", "observation.environment_state", "action"]
        for key in required_keys:
            if key in stats:
                print(f"  {key}: present in stats")
            else:
                print(f"  WARNING: {key} missing from stats")
    else:
        print(f"WARNING: stats.json not found at {stats_path}")

    episode_meta_path = output_dir / "episode_initial_states.json"
    if episode_meta_path.exists():
        print(f"episode_initial_states.json exists")
    else:
        print(f"WARNING: episode_initial_states.json not found")

    video_dir = output_dir / "videos"
    if video_dir.exists():
        video_count = len(list(video_dir.rglob("*.mp4")))
        print(f"Found {video_count} video files")
    else:
        print(f"WARNING: videos directory not found")


def main():
    args = parse_args()

    print(f"=== LIBERO Dataset Collection ===")
    print(f"Suite: {args.suite}")
    print(f"Task ID: {args.task_id}")
    print(f"Target episodes: {args.num_episodes}")
    print(f"Candidate multiplier: {args.candidate_multiplier}")
    print(f"Output: {args.output_dir}")
    print()

    suite, task = get_suite_and_task(args.suite, args.task_id)
    task_language = task.language
    task_name = task.name
    print(f"Task: {task_name}")
    print(f"Description: {task_language}")

    bddl_file = resolve_bddl_file(task)
    print(f"BDDL: {bddl_file}")

    if args.output_dir is None:
        output_dir = Path("personal/work2/dataset") / f"{args.suite}_task{args.task_id}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Step 1: Creating environment to infer state_dim ===")
    env = create_env(bddl_file, args.image_size, args.fps)
    env.reset()
    for _ in range(10):
        env.step(LIBERO_DUMMY_ACTION)

    state_dim = get_flattened_env_state(env).shape[0]
    print(f"Inferred state_dim: {state_dim}")

    print("\n=== Step 2: Generating candidate initial states ===")
    num_candidates = args.num_episodes * args.candidate_multiplier
    print(f"Generating {num_candidates} candidates...")

    candidates = generate_candidate_initial_states(
        env, bddl_file, num_candidates, args.seed_start
    )
    print(f"Generated {len(candidates)} candidates")

    if len(candidates) == 0:
        print("ERROR: No valid candidates generated!")
        env.close()
        return

    print("\n=== Step 3: Selecting uniform initial states ===")
    selected_indices = select_uniform_initial_states(candidates, args.num_episodes)
    selected_candidates = [candidates[i] for i in selected_indices]
    print(f"Selected {len(selected_candidates)} uniform initial states")

    print("\n=== Step 4: Saving uniform initial states ===")
    save_uniform_initial_states(output_dir, selected_candidates, candidates)

    print("\n=== Step 5: Getting demo data ===")
    demo_states, demo_actions = get_demo_data(task)

    if demo_states is None:
        print("WARNING: No demo data found for this task. Cannot collect trajectories.")
        print("Generating uniform initial states only...")

        descriptor_info = {
            "num_candidates": len(candidates),
            "dim": selected_candidates[0]["descriptor"].shape[0] if selected_candidates else 0,
            "object_names": list(selected_candidates[0]["object_poses"].keys()) if selected_candidates else [],
            "min": [],
            "max": [],
            "std": [],
        }

        episode_initial_states = []
        for i, cand in enumerate(selected_candidates):
            episode_initial_states.append({
                "episode_index": i,
                "seed": cand["seed"],
                "state": cand["state"],
                "object_poses": cand["object_poses"],
                "success": False,
                "num_frames": 0,
            })

        save_episode_metadata(
            output_dir, args.suite, args.task_id, task_name, task_language,
            bddl_file, state_dim, episode_initial_states, 0, descriptor_info
        )

        print("\n" + "="*60)
        print("WARNING: No expert policy available for this task.")
        print(f"Generated {len(selected_candidates)} uniform initial states,")
        print("but 0 successful trajectories (no demo data or policy).")
        print("Dataset contains only initial states, no trajectories.")
        print("="*60 + "\n")

        validate_dataset(output_dir)
        env.close()
        return

    print(f"Demo data found: {demo_states.shape[0]} frames")

    print("\n=== Step 6: Creating LeRobot dataset ===")
    dataset = create_dataset(
        repo_id=args.repo_id,
        output_dir=output_dir,
        fps=args.fps,
        image_size=args.image_size,
        state_dim=state_dim,
    )

    print("\n=== Step 7: Collecting trajectories ===")
    episode_initial_states = []
    successful_episodes = 0
    max_episodes = min(args.num_episodes, len(selected_candidates))

    for ep_idx in range(max_episodes):
        cand = selected_candidates[ep_idx]

        env.seed(cand["seed"])
        env.reset()
        for _ in range(10):
            env.step(LIBERO_DUMMY_ACTION)
        env.env.sim.set_state_from_flatten(cand["state"])
        for _ in range(5):
            env.step(LIBERO_DUMMY_ACTION)

        frames, success = collect_demo_episode(
            env, demo_states, demo_actions, task_language, args.image_size, state_dim
        )

        if success and len(frames) > 0:
            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            successful_episodes += 1
            print(f"Episode {ep_idx}: SUCCESS ({len(frames)} frames)")
        else:
            print(f"Episode {ep_idx}: FAILED (demo replay did not succeed)")

        episode_initial_states.append({
            "episode_index": ep_idx,
            "seed": cand["seed"],
            "state": cand["state"],
            "object_poses": cand["object_poses"],
            "success": success,
            "num_frames": len(frames),
        })

        if (ep_idx + 1) % 50 == 0:
            print(f"Progress: {ep_idx + 1}/{max_episodes} episodes processed")

    print("\n=== Step 8: Finalizing dataset ===")
    dataset.finalize()

    print("\n=== Step 9: Saving episode metadata ===")
    descriptors = np.array([c["descriptor"] for c in selected_candidates])
    desc_min = np.min(descriptors, axis=0).tolist() if len(descriptors) > 0 else []
    desc_max = np.max(descriptors, axis=0).tolist() if len(descriptors) > 0 else []
    desc_std = np.std(descriptors, axis=0).tolist() if len(descriptors) > 0 else []

    descriptor_info = {
        "num_candidates": len(candidates),
        "dim": descriptors.shape[1] if len(descriptors) > 0 else 0,
        "object_names": list(selected_candidates[0]["object_poses"].keys()) if selected_candidates else [],
        "min": desc_min,
        "max": desc_max,
        "std": desc_std,
    }

    save_episode_metadata(
        output_dir, args.suite, args.task_id, task_name, task_language,
        bddl_file, state_dim, episode_initial_states, successful_episodes, descriptor_info
    )

    print("\n" + "="*60)
    print(f"Collection complete!")
    print(f"  Suite: {args.suite}")
    print(f"  Task ID: {args.task_id}")
    print(f"  Task: {task_name}")
    print(f"  Uniform initial states: {len(selected_candidates)}")
    print(f"  Successful trajectories: {successful_episodes}")
    print(f"  Environment state_dim: {state_dim}")
    print(f"  Output: {output_dir}")
    print("="*60 + "\n")

    validate_dataset(output_dir)
    env.close()


if __name__ == "__main__":
    main()