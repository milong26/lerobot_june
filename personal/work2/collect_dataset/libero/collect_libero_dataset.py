#!/usr/bin/env python
"""
LIBERO 数据集生成工具：为单个 LIBERO task 生成约 300 个 episode 的 LeRobot video dataset。

功能说明：
    1. 通过 LIBERO 官方 benchmark API 获取指定 suite + task_id 的 task 信息。
    2. 解析 BDDL 文件中的 (:objects ...) 声明，获取 task 相关物体列表。
    3. 创建 OffScreenRenderEnv，使用不同 seed 多次 reset，记录每个物体的真实 pose 变化。
    4. 计算每个物体在不同 seed 之间的位置标准差，过滤掉 std < threshold 的固定物体
       （如 wooden_cabinet、flat_stove、plate 等 fixture）。
    5. 使用真正随机化的物体的 xyz 构建 descriptor，对大量 candidate 做归一化后，
       使用 farthest-point sampling 选出约 300 个均匀覆盖该 task 实际合法初始化空间的 candidate。
    6. 使用 LIBERO 官方 demonstration 中的 state replay 生成真实专家轨迹：
       读取 HDF5 中的 demo_x/states 和 actions，逐帧执行 env.set_init_state(states[t])
       得到对应的 observation，确保 observation 与 action 严格对应。
       observation.environment_state = states[t]（完整 MuJoCo flattened state）。
    7. 将轨迹写入 LeRobotDataset（use_videos=True，视频由 LeRobot 自己编码）。
    8. 保存 episode_initial_states.json 和 uniform_initial_states.npz。
    9. 验证生成的 dataset：episode 数量、video 文件、stats.json、metadata。

数据来源：
    - LIBERO 官方 benchmark：libero.libero.benchmark
    - LIBERO 官方 demonstration HDF5：通过 get_libero_path("datasets") 获取
    - BDDL 文件：通过 get_libero_path("bddl_files") 获取
    - 初始状态文件：通过 get_libero_path("init_states") 获取

重要说明：
    - 本工具使用 LIBERO task-specific 的初始化空间采样，不是 MetaWorld 那种固定 x-y 工作空间。
    - 随机物体通过 BDDL object declaration + 多次 reset 检测真实变化 + std 过滤确定。
    - observation.environment_state 保存完整 MuJoCo flattened state（不是 25 维 robot state）。
    - observation.state 为 8 维（与 LeRobot 文档一致）。
    - LeRobot 自动生成 video 和 stats，不需要手工处理。

LeRobot frame 字段：
    - observation.images.top: agentview_image (RGB, 3 x image_size x image_size)
    - observation.images.wrist: robot0_eye_in_hand_image (RGB, 3 x image_size x image_size)
    - observation.state: 8 维机器人状态（与 LeRobot 文档一致）
        robot0_eef_pos(3) + robot0_eef_axis_angle(3) + robot0_gripper_qpos(2)
    - observation.environment_state: 完整 MuJoCo flattened state（维度由第一个 state 推断）
    - action: 7 维 LIBERO action
    - next.reward: float
    - next.success: bool
    - task: task.language 字符串

运行方法：
    cd /data/zhonglinye/jun/lerobot
    python personal/work2/collect_dataset/libero/collect_libero_dataset.py \
        --suite libero_spatial \
        --task-id 0 \
        --num-episodes 300 \
        --output-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset/libero_spatial_task0 \
        --repo-id work2/libero_spatial_task0 \
        --fps 20 \
        --image-size 360 \
        --seed-start 0 \
        --candidate-multiplier 10

    快速测试（2 episode）：
    python personal/work2/collect_dataset/libero/collect_libero_dataset.py \
        --suite libero_spatial --task-id 0 --num-episodes 2 \
        --output-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset/libero_spatial_task0_test \
        --repo-id work2/libero_spatial_task0_test --candidate-multiplier 5
"""

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
from tqdm import tqdm

os.environ["HF_LEROBOT_HOME"] = str(Path(__file__).parent.parent.parent / "dataset" / "hf_cache")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from lerobot.datasets.lerobot_dataset import LeRobotDataset


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="为单个 LIBERO task 生成约 N 个 episode 的 LeRobot video dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-episodes", type=int, default=300)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=360)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--candidate-multiplier", type=int, default=10)
    parser.add_argument("--save-only-successful", action="store_true", default=True)
    parser.add_argument("--gpu-id", type=int, default=None, help="GPU ID for EGL rendering (e.g. 0)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Task / BDDL helpers
# ---------------------------------------------------------------------------

def get_suite_and_task(suite_name, task_id):
    """获取 LIBERO suite 和指定 task。"""
    bench = benchmark.get_benchmark_dict()
    if suite_name not in bench:
        raise ValueError(f"Unknown LIBERO suite '{suite_name}'. Available: {', '.join(sorted(bench.keys()))}")
    suite = bench[suite_name]()
    if task_id < 0 or task_id >= len(suite.tasks):
        raise ValueError(f"task_id {task_id} out of range [0, {len(suite.tasks) - 1}] for suite '{suite_name}'")
    task = suite.get_task(task_id)
    return suite, task


def resolve_bddl_file(task):
    """构造 task 的 BDDL 文件路径。"""
    bddl_root = Path(get_libero_path("bddl_files"))
    bddl_path = bddl_root / task.problem_folder / task.bddl_file
    if not bddl_path.exists():
        raise FileNotFoundError(f"BDDL file not found: {bddl_path}")
    return bddl_path


def parse_bddl_objects(bddl_path):
    """
    解析 BDDL 文件中的 (:objects ...) 声明，返回 task 声明的物体名称列表。
    排除 fixture（在 (:fixtures ...) 中声明的物体）。
    """
    with open(bddl_path) as f:
        content = f.read()

    # 解析 fixtures（固定物体，不参与随机初始化）
    fixtures = set()
    fixture_match = re.search(r"\(:fixtures\s+(.*?)\)", content, re.DOTALL)
    if fixture_match:
        fixture_block = fixture_match.group(1)
        # 提取物体名称（- 前面的部分，去掉 _数字 后缀）
        for line in fixture_block.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # e.g. "main_table - table"
            parts = line.split("-")
            if len(parts) >= 2:
                name = parts[0].strip().split("_")[0]  # e.g. main_table
                # 也保存完整名称
                full_name = parts[0].strip()
                fixtures.add(full_name)

    # 解析 objects
    objects = []
    obj_match = re.search(r"\(:objects\s+(.*?)\)", content, re.DOTALL)
    if obj_match:
        obj_block = obj_match.group(1)
        for line in obj_block.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # e.g. "akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl"
            parts = line.split("-")
            if len(parts) >= 2:
                names = parts[0].strip().split()
                objects.extend(names)

    return objects, fixtures


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def create_env(bddl_path, image_size=360, control_freq=20, gpu_id=None):
    """创建 LIBERO OffScreenRenderEnv 环境。"""
    if gpu_id is not None:
        os.environ["EGL_DEVICE_ID"] = str(gpu_id)
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_path),
        camera_heights=image_size,
        camera_widths=image_size,
        control_freq=control_freq,
    )
    env.reset()
    return env


def get_flattened_env_state(env):
    """获取环境的完整 flattened MuJoCo simulator state。"""
    if hasattr(env, "env") and hasattr(env.env, "sim"):
        state = env.env.sim.get_state()
        return state.flatten().copy()
    if hasattr(env, "sim"):
        state = env.sim.get_state()
        return state.flatten().copy()
    raise RuntimeError("Cannot access MuJoCo simulator state from env")


def _get_mujoco_model(env):
    """获取 MuJoCo model 对象。"""
    if hasattr(env, "env") and hasattr(env.env, "sim"):
        return env.env.sim.model
    if hasattr(env, "sim"):
        return env.sim.model
    raise RuntimeError("Cannot access MuJoCo model from env")


def _get_mujoco_data(env):
    """获取 MuJoCo data 对象。"""
    if hasattr(env, "env") and hasattr(env.env, "sim"):
        return env.env.sim.data
    if hasattr(env, "sim"):
        return env.sim.data
    raise RuntimeError("Cannot access MuJoCo data from env")


# ---------------------------------------------------------------------------
# Object detection via BDDL + variance analysis
# ---------------------------------------------------------------------------

def is_candidate_movable_object(object_name, fixtures, bddl_objects):
    """
    判断 object 是否是真正可能随机化的物体。
    排除 fixture 和不在 BDDL objects 声明中的物体。
    """
    # 检查是否在 BDDL objects 列表中
    if object_name not in bddl_objects:
        return False
    # 检查是否是 fixture
    if object_name in fixtures:
        return False
    return True


def get_object_pose_from_sim(env, object_name):
    """
    从 MuJoCo simulator 中获取指定物体的 position。
    通过 body name 匹配，返回 xpos (3,)。
    """
    model = _get_mujoco_model(env)
    data = _get_mujoco_data(env)

    num_bodies = model.nbody
    for i in range(num_bodies):
        name = model.body(i).name
        if name == object_name:
            return data.xpos[i].copy()
    return None


def collect_object_pose_variance(env, object_names, num_samples=50, seed_start=0):
    """
    使用多个 seed reset 环境，记录每个 object 的位置变化。
    返回 dict: {object_name: [pose_array, ...]}
    """
    pose_records = {name: [] for name in object_names}

    for i in range(num_samples):
        seed = seed_start + i
        try:
            env.seed(seed)
            env.reset()
            for name in object_names:
                pose = get_object_pose_from_sim(env, name)
                if pose is not None:
                    pose_records[name].append(pose)
        except Exception as e:
            warnings.warn(f"Failed seed={seed}: {e}")

    return pose_records


def filter_varying_objects(pose_records, threshold=1e-6):
    """
    根据位置标准差过滤固定物体。
    返回 (varying_objects, object_stds)：
      - varying_objects: std > threshold 的物体名称列表
      - object_stds: {object_name: std_array}
    """
    varying_objects = []
    object_stds = {}

    for name, poses in pose_records.items():
        if len(poses) < 2:
            object_stds[name] = np.array([0.0, 0.0, 0.0])
            continue
        pose_array = np.array(poses)  # (N, 3)
        stds = np.std(pose_array, axis=0)  # (3,)
        object_stds[name] = stds
        # 如果任一维度 std > threshold，认为该物体可随机化
        if np.any(stds > threshold):
            varying_objects.append(name)

    return varying_objects, object_stds


def get_randomizable_object_poses(env, varying_objects):
    """
    提取当前 task 中真正参与随机初始化的物体的 pose。
    只返回经过 variance 分析确认可移动的物体。
    """
    result = {}
    for name in varying_objects:
        pose = get_object_pose_from_sim(env, name)
        if pose is not None:
            result[name] = {"position": pose}
    return result


def build_initial_state_descriptor(object_poses):
    """
    构造初始状态 descriptor，用于均匀采样。
    把所有可移动物体的 xyz position 拼接成一个向量，不包括机器人 joint state。
    """
    parts = []
    names = []
    for name, pose in sorted(object_poses.items()):
        parts.append(pose["position"])
        names.append(name)
    if not parts:
        return np.array([]), []
    descriptor = np.concatenate([p.flatten() for p in parts])
    return descriptor, names


# ---------------------------------------------------------------------------
# Candidate generation with progress bar
# ---------------------------------------------------------------------------

def generate_candidate_initial_states(env, num_candidates, seed_start=0):
    """
    用大量不同 seed reset 环境，生成合法 candidate initial states。
    使用 tqdm 显示进度条。
    """
    candidates = []
    failed_count = 0
    max_attempts = num_candidates * 5

    pbar = tqdm(
        range(seed_start, seed_start + max_attempts),
        desc="Generating initial states",
        total=max_attempts,
    )

    for seed in pbar:
        if len(candidates) >= num_candidates:
            break
        try:
            env.seed(seed)
            obs = env.reset()
            flattened_state = get_flattened_env_state(env)
            candidates.append({
                "seed": seed,
                "flattened_state": flattened_state,
            })
        except Exception as e:
            failed_count += 1
            warnings.warn(f"Failed seed={seed}: {e}")

    pbar.close()

    print(f"  Candidate 生成完成: {len(candidates)} 个成功, {failed_count} 个失败")
    return candidates


# ---------------------------------------------------------------------------
# Dimension filtering & uniform selection
# ---------------------------------------------------------------------------

def remove_constant_dimensions(descriptors, threshold=1e-6):
    """排除 descriptor 中固定不变的维度（标准差 < threshold）。"""
    if descriptors.size == 0:
        return np.array([], dtype=bool)
    stds = np.std(descriptors, axis=0)
    return stds > threshold


def select_uniform_initial_states(candidates, num_target, varying_mask=None):
    """
    使用 farthest-point sampling 从 candidate 中选择均匀覆盖初始化空间的子集。
    1. 对 descriptor 每一维按 min/max 归一化到 [0, 1]。
    2. 第一个点选择离归一化空间中心 (0.5, 0.5, ...) 最近的 candidate。
    3. 之后每次选择距离已选集合最近距离最大的 candidate（max-min diversity）。
    使用 tqdm 显示进度条。
    """
    if not candidates:
        return []

    n = len(candidates)
    descriptors = np.array([c["descriptor"] for c in candidates])
    if descriptors.ndim == 1:
        descriptors = descriptors.reshape(-1, 1)

    if varying_mask is not None and varying_mask.size > 0:
        descriptors = descriptors[:, varying_mask]

    desc_min = descriptors.min(axis=0)
    desc_max = descriptors.max(axis=0)
    desc_range = desc_max - desc_min
    desc_range[desc_range < 1e-12] = 1.0
    descriptors_norm = (descriptors - desc_min) / desc_range

    num_select = min(num_target, n)
    selected_indices = []

    center = np.full(descriptors_norm.shape[1], 0.5)
    distances_to_center = np.linalg.norm(descriptors_norm - center, axis=1)
    first_idx = int(np.argmin(distances_to_center))
    selected_indices.append(first_idx)

    min_distances = np.full(n, np.inf)
    for i in range(n):
        min_distances[i] = np.linalg.norm(descriptors_norm[i] - descriptors_norm[first_idx])

    pbar = tqdm(range(1, num_select), desc="Selecting uniform states")
    for _ in pbar:
        next_idx = int(np.argmax(min_distances))
        selected_indices.append(next_idx)
        new_distances = np.linalg.norm(descriptors_norm - descriptors_norm[next_idx], axis=1)
        min_distances = np.minimum(min_distances, new_distances)
        pbar.set_postfix({"selected": len(selected_indices), "max_dist": f"{min_distances.max():.4f}"})

    pbar.close()
    return selected_indices


# ---------------------------------------------------------------------------
# Save initial states
# ---------------------------------------------------------------------------

def save_uniform_initial_states(output_dir, selected_candidates, all_candidates,
                                 suite_name, task_id, task, mujoco_state_dim,
                                 descriptor_dim, varying_objects, object_stds,
                                 varying_mask, descriptor_stats):
    """保存均匀初始状态到 NPZ 和 JSON 文件。"""
    output_dir = Path(output_dir)

    npz_file = output_dir / "uniform_initial_states.npz"
    states = np.array([c["flattened_state"] for c in selected_candidates])
    seeds = np.array([c["seed"] for c in selected_candidates])
    all_states = np.array([c["flattened_state"] for c in all_candidates])
    all_seeds = np.array([c["seed"] for c in all_candidates])
    np.savez(str(npz_file), states=states, seeds=seeds, all_states=all_states, all_seeds=all_seeds)
    print(f"  Uniform initial states saved to {npz_file}")

    json_file = output_dir / "episode_initial_states.json"
    metadata = {
        "suite": suite_name,
        "task_id": task_id,
        "task_name": task.name,
        "task_description": task.language,
        "bddl_file": str(resolve_bddl_file(task)),
        "mujoco_state_dim": mujoco_state_dim,
        "sampling_method": "task_space_farthest_point",
        "num_candidates": len(all_candidates),
        "num_target_episodes": len(selected_candidates),
        "descriptor_dim": descriptor_dim,
        "varying_objects": varying_objects,
        "object_stds": {k: v.tolist() for k, v in object_stds.items()},
        "varying_mask": varying_mask.tolist() if varying_mask is not None else None,
        "descriptor_stats": descriptor_stats,
        "episodes": [],
    }

    for i, c in enumerate(selected_candidates):
        ep_data = {
            "episode_index": i,
            "seed": int(c["seed"]),
            "initial_environment_state": c["flattened_state"].tolist(),
            "success": None,
            "num_frames": None,
        }
        metadata["episodes"].append(ep_data)

    with open(json_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Episode initial states metadata saved to {json_file}")


# ---------------------------------------------------------------------------
# Robot state builders
# ---------------------------------------------------------------------------

def build_robot_state(obs):
    """
    从 LIBERO observation 构造 8 维机器人状态向量（与 LeRobot 文档一致）。
    固定顺序：robot0_eef_pos(3) + robot0_eef_axis_angle(3) + robot0_gripper_qpos(2)
    """
    eef_pos = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
    eef_quat = np.asarray(obs.get("robot0_eef_quat", np.zeros(4)), dtype=np.float32)
    gripper_qpos = np.asarray(obs.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float32)

    from scipy.spatial.transform import Rotation as R
    rot = R.from_quat([eef_quat[1], eef_quat[2], eef_quat[3], eef_quat[0]])
    axis_angle = rot.as_rotvec().astype(np.float32)

    state = np.concatenate([eef_pos, axis_angle, gripper_qpos])
    assert state.shape == (8,), f"Expected state shape (8,), got {state.shape}"
    return state


def build_full_robot_state(obs):
    """
    从 LIBERO observation 构造 25 维完整机器人状态向量（仅用于 debug，不写入 dataset）。
    """
    eef_pos = np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
    eef_quat = np.asarray(obs.get("robot0_eef_quat", np.zeros(4)), dtype=np.float32)
    gripper_qpos = np.asarray(obs.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float32)
    gripper_qvel = np.asarray(obs.get("robot0_gripper_qvel", np.zeros(2)), dtype=np.float32)
    joint_pos = np.asarray(obs.get("robot0_joint_pos", np.zeros(7)), dtype=np.float32)
    joint_vel = np.asarray(obs.get("robot0_joint_vel", np.zeros(7)), dtype=np.float32)

    state = np.concatenate([eef_pos, eef_quat, gripper_qpos, gripper_qvel, joint_pos, joint_vel])
    assert state.shape == (25,), f"Expected state shape (25,), got {state.shape}"
    return state


def _resize_image(image, target_size):
    """使用 PIL 调整图像大小。"""
    from PIL import Image
    img = Image.fromarray(image)
    img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img)


# ---------------------------------------------------------------------------
# LeRobotDataset creation
# ---------------------------------------------------------------------------

def create_dataset(repo_id, output_dir, fps=20, image_size=360, mujoco_state_dim=None):
    """创建空的 LeRobotDataset，启用视频编码。"""
    if mujoco_state_dim is None:
        raise ValueError("mujoco_state_dim must be provided")

    features = {
        "observation.images.top": {
            "dtype": "video",
            "shape": (3, image_size, image_size),
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist": {
            "dtype": "video",
            "shape": (3, image_size, image_size),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (mujoco_state_dim,),
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
    }

    output_path = Path(output_dir)
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type="libero",
        use_videos=True,
    )
    return dataset


# ---------------------------------------------------------------------------
# Episode collection (state replay)
# ---------------------------------------------------------------------------

def collect_demo_episode(env, demo_data, image_size=360, task_description="", extra_frames_after_success=10):
    """
    使用 state replay 从 LIBERO 官方 demonstration 生成一个 episode。
    读取 HDF5 中的 data/demo_x/states 和 actions，逐帧执行 env.set_init_state(states[t])
    得到与该 state 对应的 observation。
    observation.environment_state = states[t]（完整 MuJoCo flattened state）。
    任务完成后额外等待 extra_frames_after_success 帧，让模型看到任务完成后的稳定状态。
    """
    states = demo_data["states"]
    actions = demo_data["actions"]

    frames = []
    success = False

    for t in range(len(states)):
        env.set_init_state(states[t])
        obs = env.env._get_observations()

        agentview_img = obs.get("agentview_image")
        wrist_img = obs.get("robot0_eye_in_hand_image")

        if agentview_img is None or wrist_img is None:
            continue

        agentview_img = np.flip(agentview_img, (0, 1))
        wrist_img = np.flip(wrist_img, (0, 1))
        agentview_img = _resize_image(agentview_img, image_size)
        wrist_img = _resize_image(wrist_img, image_size)

        robot_state = build_robot_state(obs)

        frame = {
            "observation.images.top": agentview_img,
            "observation.images.wrist": wrist_img,
            "observation.state": robot_state,
            "observation.environment_state": states[t].copy().astype(np.float32),
            "action": actions[t].copy().astype(np.float32),
            "next.reward": np.array([1.0 if t == len(states) - 1 else 0.0], dtype=np.float32),
            "next.success": np.array([t == len(states) - 1], dtype=bool),
            "task": task_description,
        }
        frames.append(frame)

        if t == len(states) - 1:
            success = True

    # 任务完成后额外等待帧：保持最后一帧的状态，继续采集 observation
    if success and extra_frames_after_success > 0:
        env.set_init_state(states[-1])
        for _ in range(extra_frames_after_success):
            obs = env.env._get_observations()

            agentview_img = obs.get("agentview_image")
            wrist_img = obs.get("robot0_eye_in_hand_image")

            if agentview_img is None or wrist_img is None:
                continue

            agentview_img = np.flip(agentview_img, (0, 1))
            wrist_img = np.flip(wrist_img, (0, 1))
            agentview_img = _resize_image(agentview_img, image_size)
            wrist_img = _resize_image(wrist_img, image_size)

            robot_state = build_robot_state(obs)

            frame = {
                "observation.images.top": agentview_img,
                "observation.images.wrist": wrist_img,
                "observation.state": robot_state,
                "observation.environment_state": states[-1].copy().astype(np.float32),
                "action": np.zeros(7, dtype=np.float32),
                "next.reward": np.array([1.0], dtype=np.float32),
                "next.success": np.array([True], dtype=bool),
                "task": task_description,
            }
            frames.append(frame)

    return frames, success, len(frames)


# ---------------------------------------------------------------------------
# Load demonstrations
# ---------------------------------------------------------------------------

def load_libero_demonstrations(suite_name, task_id):
    """加载 LIBERO 官方 demonstration HDF5 文件。"""
    import h5py

    datasets_dir = Path(get_libero_path("datasets"))
    demo_files = list(datasets_dir.glob(f"**/{suite_name}/**demo*.hdf5"))
    if not demo_files:
        demo_files = list(datasets_dir.glob(f"**/*{suite_name}*demo*.hdf5"))
    if not demo_files:
        demo_files = list(datasets_dir.glob(f"**/*demo*.hdf5"))

    if not demo_files:
        print(f"WARNING: No demonstration HDF5 files found in {datasets_dir}")
        return []

    all_demos = []
    for demo_file in demo_files:
        try:
            with h5py.File(demo_file, "r") as f:
                demo_keys = [k for k in f["data"].keys() if k.startswith("demo_")]
                for dk in demo_keys:
                    demo_grp = f["data"][dk]
                    if "states" in demo_grp and "actions" in demo_grp:
                        demo_data = {
                            "states": demo_grp["states"][:],
                            "actions": demo_grp["actions"][:],
                        }
                        if "dones" in demo_grp:
                            demo_data["dones"] = demo_grp["dones"][:]
                        all_demos.append(demo_data)
        except Exception as e:
            warnings.warn(f"Failed to read demo file {demo_file}: {e}")

    print(f"  Loaded {len(all_demos)} demonstrations from {len(demo_files)} files")
    return all_demos


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dataset(output_dir, expected_episodes, mujoco_state_dim):
    """验证生成的 LeRobotDataset 是否完整正确。"""
    output_path = Path(output_dir)
    results = {"valid": True, "issues": []}

    stats_file = output_path / "meta" / "stats.json"
    if not stats_file.exists():
        results["valid"] = False
        results["issues"].append("stats.json not found")
    else:
        with open(stats_file) as f:
            stats = json.load(f)
        required_keys = ["observation.state", "observation.environment_state", "action"]
        for key in required_keys:
            if key not in stats:
                results["valid"] = False
                results["issues"].append(f"stats.json missing key: {key}")

        env_state_stats = stats.get("observation.environment_state", {})
        if "mean" in env_state_stats:
            mean_arr = np.array(env_state_stats["mean"])
            if mean_arr.shape[0] != mujoco_state_dim:
                results["valid"] = False
                results["issues"].append(
                    f"environment_state stat dim {mean_arr.shape[0]} != expected {mujoco_state_dim}"
                )

        obs_state_stats = stats.get("observation.state", {})
        if "mean" in obs_state_stats:
            mean_arr = np.array(obs_state_stats["mean"])
            if mean_arr.shape[0] != 8:
                results["valid"] = False
                results["issues"].append(
                    f"observation.state stat dim {mean_arr.shape[0]} != expected 8"
                )

    ep_json = output_path / "episode_initial_states.json"
    if not ep_json.exists():
        results["valid"] = False
        results["issues"].append("episode_initial_states.json not found")

    videos_dir = output_path / "videos"
    if videos_dir.exists():
        video_files = list(videos_dir.glob("**/*.mp4"))
        print(f"  Found {len(video_files)} video files")
    else:
        results["issues"].append("videos directory not found")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.gpu_id is not None:
        os.environ["EGL_DEVICE_ID"] = str(args.gpu_id)
        print(f"Using GPU ID: {args.gpu_id}")

    if args.output_dir is None:
        args.output_dir = str(
            Path(__file__).parent.parent.parent / "dataset" / f"{args.suite}_task{args.task_id}"
        )
    if args.repo_id is None:
        args.repo_id = f"work2/{args.suite}_task{args.task_id}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LIBERO 数据采集")
    print("=" * 80)
    print(f"Suite: {args.suite}")
    print(f"Task ID: {args.task_id}")
    print(f"目标 Episode 数量: {args.num_episodes}")
    print(f"输出目录: {args.output_dir}")
    print(f"Repo ID: {args.repo_id}")
    print(f"FPS: {args.fps}")
    print(f"图像分辨率: {args.image_size}x{args.image_size}")
    print(f"Seed 起始: {args.seed_start}")
    print(f"Candidate Multiplier: {args.candidate_multiplier}")
    if args.gpu_id is not None:
        print(f"GPU ID: {args.gpu_id}")
    print("=" * 80)

    # 1. 获取 suite 和 task
    print("\n[1/9] 获取 LIBERO suite 和 task...")
    suite, task = get_suite_and_task(args.suite, args.task_id)
    print(f"  Task name: {task.name}")
    print(f"  Task description: {task.language}")
    print(f"  Problem folder: {task.problem_folder}")
    print(f"  BDDL file: {task.bddl_file}")

    # 2. 解析 BDDL
    print("\n[2/9] 解析 BDDL 文件...")
    bddl_path = resolve_bddl_file(task)
    print(f"  BDDL path: {bddl_path}")

    bddl_objects, fixtures = parse_bddl_objects(bddl_path)
    print(f"  BDDL objects: {bddl_objects}")
    print(f"  Fixtures: {fixtures}")
    print(f"  BDDL object 数量: {len(bddl_objects)}")

    # 3. 创建环境，识别真正随机化的物体
    print("\n[3/9] 创建环境，通过 variance 分析识别随机初始化物体...")
    env = create_env(bddl_path, image_size=args.image_size, control_freq=args.fps, gpu_id=args.gpu_id)

    # 3a. 收集物体位置方差
    num_variance_samples = 50
    print(f"  使用 {num_variance_samples} 个 seed 收集物体位置方差...")
    pose_records = collect_object_pose_variance(env, bddl_objects, num_variance_samples, args.seed_start)

    # 3b. 过滤固定物体
    varying_objects, object_stds = filter_varying_objects(pose_records)

    print(f"  BDDL objects 总数: {len(bddl_objects)}")
    print(f"  每个物体位置 std:")
    for name, stds in object_stds.items():
        is_varying = name in varying_objects
        marker = "✓" if is_varying else "✗"
        print(f"    {marker} {name}: std=[{stds[0]:.6f}, {stds[1]:.6f}, {stds[2]:.6f}]")

    print(f"  过滤后随机化物体 ({len(varying_objects)} 个): {varying_objects}")

    # 3c. 获取当前环境的随机化物体 pose
    object_poses = get_randomizable_object_poses(env, varying_objects)
    descriptor, _ = build_initial_state_descriptor(object_poses)
    descriptor_dim = len(descriptor)
    print(f"  Descriptor 维度: {descriptor_dim}")

    # 4. 生成 candidate initial states
    num_candidates = args.num_episodes * args.candidate_multiplier
    print(f"\n[4/9] 生成 {num_candidates} 个 candidate initial states...")
    candidates = generate_candidate_initial_states(env, num_candidates, seed_start=args.seed_start)

    if len(candidates) == 0:
        print("ERROR: No candidates generated. Check environment setup.")
        env.close()
        sys.exit(1)

    # 为每个 candidate 附加 descriptor
    print("  为 candidate 附加 descriptor...")
    for c in tqdm(candidates, desc="Building descriptors"):
        # 需要重新 set_init_state 来获取正确的 object poses
        env.set_init_state(c["flattened_state"])
        obj_poses = get_randomizable_object_poses(env, varying_objects)
        desc, _ = build_initial_state_descriptor(obj_poses)
        c["descriptor"] = desc
        c["object_poses"] = obj_poses

    # 5. 排除固定维度 + 均匀采样
    print(f"\n[5/9] 排除固定维度 + farthest-point sampling...")
    descriptors = np.array([c["descriptor"] for c in candidates])
    if descriptors.ndim == 1:
        descriptors = descriptors.reshape(-1, 1)

    varying_mask = remove_constant_dimensions(descriptors)
    n_varying = int(varying_mask.sum()) if varying_mask.size > 0 else 0
    n_total = descriptors.shape[1]
    n_removed = n_total - n_varying
    print(f"  Before filtering: {n_total}")
    print(f"  Removed fixed dimensions: {n_removed}")
    print(f"  After filtering: {n_varying}")

    descriptor_stats = {
        "min": descriptors.min(axis=0).tolist(),
        "max": descriptors.max(axis=0).tolist(),
        "mean": descriptors.mean(axis=0).tolist(),
        "std": descriptors.std(axis=0).tolist(),
        "varying_dimensions": n_varying,
    }

    print(f"\n  使用 farthest-point sampling 选择 {args.num_episodes} 个均匀初始状态...")
    selected_indices = select_uniform_initial_states(candidates, args.num_episodes, varying_mask)
    selected_candidates = [candidates[i] for i in selected_indices]
    print(f"  选中 {len(selected_candidates)} 个 uniform initial states")

    # 6. 保存 uniform initial states
    print("\n[6/9] 保存 uniform initial states...")
    mujoco_state_dim = len(selected_candidates[0]["flattened_state"])
    save_uniform_initial_states(
        output_dir, selected_candidates, candidates,
        args.suite, args.task_id, task, mujoco_state_dim,
        descriptor_dim, varying_objects, object_stds, varying_mask, descriptor_stats,
    )

    # 7. 生成 trajectory
    print("\n[7/9] 生成 expert trajectory...")
    print("  加载 LIBERO 官方 demonstration...")
    all_demos = load_libero_demonstrations(args.suite, args.task_id)

    if not all_demos:
        print("  WARNING: 没有找到官方 demonstration 文件。")
        print("  无法为新的随机初始状态生成专家 action。")
        print("  只能保存 uniform initial states，无法生成 trajectory dataset。")
        env.close()
        print("\n" + "=" * 80)
        print("完成！")
        print("=" * 80)
        print(f"Uniform initial states 已保存到: {output_dir}")
        print("WARNING: 由于缺少 expert policy，未生成 LeRobot trajectory dataset。")
        return

    print("  创建 LeRobotDataset...")
    dataset = create_dataset(args.repo_id, args.output_dir, args.fps, args.image_size, mujoco_state_dim)

    success_count = 0
    total_success_frames = 0
    max_demos = len(all_demos)

    demo_idx = 0
    ep_idx = 0

    pbar = tqdm(range(args.num_episodes), desc="Collecting episodes")
    while ep_idx < args.num_episodes and demo_idx < max_demos * 3:
        current_demo_idx = demo_idx % max_demos
        demo_data = all_demos[current_demo_idx]

        frames, success, num_frames = collect_demo_episode(
            env, demo_data, image_size=args.image_size,
            task_description=task.language,
        )

        if num_frames > 0:
            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()

            if ep_idx < len(selected_candidates):
                selected_candidates[ep_idx]["success"] = success
                selected_candidates[ep_idx]["num_frames"] = num_frames

            success_count += 1
            total_success_frames += num_frames
            pbar.set_postfix({"frames": num_frames, "success": "Y" if success else "N"})
            pbar.update(1)
            ep_idx += 1

        demo_idx += 1

    pbar.close()
    env.close()

    if success_count < args.num_episodes:
        print(f"\n  WARNING: 只生成了 {success_count}/{args.num_episodes} 个 episode。")
        print(f"  原因：官方 demonstration 数量有限，无法为新的随机初始状态生成专家 action。")
        print(f"  实际成功 trajectory 数量: {success_count}")

    # 更新 JSON metadata
    json_file = output_dir / "episode_initial_states.json"
    if json_file.exists():
        with open(json_file, "r") as f:
            metadata = json.load(f)
        metadata["num_successful_trajectories"] = success_count
        metadata["total_frames"] = total_success_frames
        for i, c in enumerate(selected_candidates):
            if i < len(metadata["episodes"]):
                metadata["episodes"][i]["success"] = c.get("success")
                metadata["episodes"][i]["num_frames"] = c.get("num_frames")
        with open(json_file, "w") as f:
            json.dump(metadata, f, indent=2)

    # 8. 完成数据集
    print("\n[8/9] 完成数据集...")
    dataset.finalize()

    # 9. 验证
    print("\n[9/9] 验证数据集...")
    validation = validate_dataset(args.output_dir, success_count, mujoco_state_dim)
    if validation["valid"]:
        print("  验证通过！")
    else:
        print(f"  验证发现问题: {validation['issues']}")

    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"Task: {task.name}")
    print(f"Task description: {task.language}")
    print(f"BDDL objects: {bddl_objects}")
    print(f"Fixtures: {fixtures}")
    print(f"Varying objects (after filtering): {varying_objects}")
    print(f"Candidate 数量: {len(candidates)}")
    print(f"Descriptor 维度 (before filtering): {descriptor_dim}")
    print(f"Descriptor 维度 (after filtering): {n_varying}")
    print(f"Uniform initial states 数量: {len(selected_candidates)}")
    print(f"真实成功 trajectory 数量: {success_count}")
    print(f"observation.state dim: 8 (eef_pos + axis_angle + gripper_qpos)")
    print(f"observation.environment_state dim: {mujoco_state_dim} (完整 MuJoCo flattened state)")
    print(f"数据集路径: {args.output_dir}")

    if success_count < args.num_episodes:
        print(f"\nWARNING: 只生成了 {success_count}/{args.num_episodes} 个 episode。")
        print("  由于缺少能为新随机初始状态产生专家 action 的 policy，")
        print("  无法生成完整的 expert trajectory。")
        print("  uniform_initial_states.json 中已保存均匀初始状态，")
        print(f"  但 LeRobot dataset 中只有 {success_count} 个真实 trajectory。")

    print("=" * 80)


if __name__ == "__main__":
    main()