#!/usr/bin/env python
"""
两阶段采集 RoboMME 数据集：
  阶段1 — 随机采样：通过 episode_idx (seed) 递增随机采集 N 个成功 episode
  阶段2 — 均匀采样：对可动物体空间进行均匀网格化采样 M 个 episode

两阶段共享同一个 LeRobot 数据集（resume 模式追加），最终生成统一的 episode_initial_states.json。

使用示例:
    # 阶段1 采集300个随机episode + 阶段2 采集100个uniform episode
    python collect_robomme_bytask.py \
        --task PickXtimes \
        --num-random-episodes 300 \
        --num-uniform-episodes 100 \
        --output-dir personal/work2/dataset_view_robomme/PickXtimes/ \
        --repo-id work2/robomme_PickXtimes \
        --seed-start 0

    # 只采集随机阶段
    python collect_robomme_bytask.py \
        --task BinFill \
        --num-random-episodes 400 \
        --num-uniform-episodes 0 \
        --output-dir ./outputs/robomme_binfill \
        --repo-id work2/robomme_binfill
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Headless rendering setup - MUST be set before importing robomme/maniskill
# RoboMME uses Vulkan rendering via SAPIEN. Configure Vulkan for headless rendering.
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# Point Vulkan to NVIDIA ICD explicitly
nvidia_icd = "/usr/share/vulkan/icd.d/nvidia_icd.json"
if os.path.exists(nvidia_icd):
    os.environ["VK_ICD_FILENAMES"] = nvidia_icd

# Force SAPIEN to use the first Vulkan device (NVIDIA GPU)
os.environ["SAPIEN_VULKAN_DEVICE_INDEX"] = "0"

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ─── Task-specific configuration ────────────────────────────────────────

TASK_DESCRIPTIONS = {
    "BinFill": "Fill the bin with objects",
    "PickXtimes": "Pick up the object X times",
    "SwingXtimes": "Swing the object X times",
    "StopCube": "Stop the cube at the target",
    "VideoUnmask": "Unmask the object from video observation",
    "VideoUnmaskSwap": "Unmask and swap objects from video observation",
    "ButtonUnmask": "Press the unmasked button",
    "ButtonUnmaskSwap": "Press the swapped unmasked button",
    "PickHighlight": "Pick the highlighted object",
    "VideoRepick": "Re-pick the object from video observation",
    "VideoPlaceButton": "Place object and press button from video",
    "VideoPlaceOrder": "Place objects in order from video observation",
    "MoveCube": "Move the cube to the target",
    "InsertPeg": "Insert the peg into the hole",
    "PatternLock": "Unlock the pattern lock",
    "RouteStick": "Route the stick through the path",
}

# Movable object space bounds per task (for uniform grid sampling)
# Format: {task_name: {"bounds": [(x_min, x_max), (y_min, y_max), (z_min, z_max)], "num_objects": N}}
TASK_OBJECT_BOUNDS = {
    "BinFill": {"bounds": [(-0.3, 0.3), (0.4, 0.8), (0.02, 0.02)], "num_objects": 1},
    "PickXtimes": {"bounds": [(-0.3, 0.3), (0.4, 0.8), (0.02, 0.02)], "num_objects": 1},
    "SwingXtimes": {"bounds": [(-0.3, 0.3), (0.4, 0.8), (0.02, 0.02)], "num_objects": 1},
    "StopCube": {"bounds": [(-0.3, 0.3), (0.4, 0.8), (0.02, 0.02)], "num_objects": 1},
    "VideoUnmask": {"bounds": [(-0.3, 0.3), (0.4, 0.8), (0.02, 0.02)], "num_objects": 1},
    "VideoUnmaskSwap": {"bounds": [(-0.3, 0.3), (0.4, 0.8), (0.02, 0.02)], "num_objects": 2},
    "ButtonUnmask": {"bounds": [(-0.3, 0.3), (0.4, 0.8), (0.02, 0.02)], "num_objects": 1},
    "ButtonUnmaskSwap": {"bounds": [(-0.3, 0.3), (0.4, 0.8), (0.02, 0.02)], "num_objects": 2},
    "PickHighlight": {"bounds": [(-0.3, 0.3), (0.4, 0.8), (0.02, 0.02)], "num_objects": 1},
    "VideoRepick": {"bounds": [(-0.3, 0.3), (0.4, 0.8), (0.02, 0.02)], "num_objects": 1},
    "VideoPlaceButton": {"bounds": [(-0.3, 0.3), (0.4, 0.8), (0.02, 0.02)], "num_objects": 1},
    "VideoPlaceOrder": {"bounds": [(-0.3, 0.3), (0.4, 0.8), (0.02, 0.02)], "num_objects": 2},
    "MoveCube": {"bounds": [(-0.3, 0.3), (0.4, 0.8), (0.02, 0.02)], "num_objects": 1},
    "InsertPeg": {"bounds": [(-0.3, 0.3), (0.4, 0.8), (0.02, 0.02)], "num_objects": 1},
    "PatternLock": {"bounds": [(-0.3, 0.3), (0.4, 0.8), (0.02, 0.02)], "num_objects": 1},
    "RouteStick": {"bounds": [(-0.3, 0.3), (0.4, 0.8), (0.02, 0.02)], "num_objects": 1},
}

# Grid resolution for uniform sampling (per axis)
GRID_RESOLUTION = {"x": 5, "y": 5, "z": 1}

# Max perturbation retries for uniform sampling
MAX_PERTURB_RETRIES = 4
# Extra frames to collect after success
EXTRA_FRAMES_AFTER_SUCCESS = 10
# Max consecutive failures before skipping
MAX_CONSECUTIVE_FAILURES = 50


# ─── Environment creation ────────────────────────────────────────────────

def create_robomme_env(task, episode_idx, dataset="test", action_space="joint_angle",
                       max_steps=300):
    """Create a single RoboMME environment for a given episode index."""
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    builder = BenchmarkEnvBuilder(
        env_id=task,
        dataset=dataset,
        action_space=action_space,
        gui_render=False,
        max_steps=max_steps,
    )
    env = builder.make_env_for_episode(
        episode_idx=episode_idx,
        max_steps=max_steps,
    )
    return env, builder


def get_episode_metadata_safe(builder, task, episode_idx):
    """Safely get episode metadata (seed, difficulty)."""
    from robomme.env_record_wrapper import get_episode_metadata
    try:
        meta = get_episode_metadata(builder.metadata_index, task, episode_idx)
        return meta or {}
    except Exception:
        return {}


def get_object_positions_from_env(env):
    """Extract object positions from the environment after reset.
    
    This attempts to get actor positions from the underlying ManiSkill env.
    Returns a dict of {actor_name: position_xyz}.
    """
    positions = {}
    try:
        # Navigate to the innermost ManiSkill env
        inner = env
        while hasattr(inner, "env") and not hasattr(inner, "scene"):
            inner = inner.env
        
        if hasattr(inner, "scene") and inner.scene is not None:
            scene = inner.scene
            # Try to get actor positions from the scene
            if hasattr(scene, "actors"):
                actors = scene.actors
                if actors is not None:
                    for i, actor in enumerate(actors):
                        if hasattr(actor, "get_pose"):
                            pose = actor.get_pose()
                            if hasattr(pose, "p"):
                                positions[f"actor_{i}"] = pose.p.tolist()
                        elif hasattr(actor, "pose"):
                            pose = actor.pose
                            if hasattr(pose, "p"):
                                positions[f"actor_{i}"] = pose.p.tolist()
    except Exception as e:
        print(f"  Warning: Could not extract object positions: {e}")
    
    return positions


def get_initial_state_info(env, builder, task, episode_idx):
    """Collect all available initial state information for an episode."""
    info = {
        "episode_idx": episode_idx,
        "task": task,
    }
    
    # Get metadata (seed, difficulty)
    meta = get_episode_metadata_safe(builder, task, episode_idx)
    if meta:
        info["seed"] = meta.get("seed")
        info["difficulty"] = meta.get("difficulty")
    
    # Try to get object positions
    obj_positions = get_object_positions_from_env(env)
    if obj_positions:
        info["object_positions"] = obj_positions
    
    return info


# ─── Episode execution ───────────────────────────────────────────────────

def run_episode(env, task, max_steps, image_size, extra_frames_after_success):
    """Run a single episode and collect frames.
    
    Returns (frames, episode_info) where frames is a list of dicts
    and episode_info contains metadata about the episode.
    """
    obs, info = env.reset()
    
    frames = []
    success_flags = []
    success_detected = False
    frames_after_success = 0
    
    # Get initial state info
    initial_obj_positions = get_object_positions_from_env(env)
    
    for step in range(max_steps):
        # For data collection, we use zero actions (or could use a policy)
        # Since RoboMME is for evaluation, we collect demonstration data
        # by using the environment's built-in demonstration if available
        action_dim = env.action_space.shape[0]
        action = np.zeros(action_dim, dtype=np.float32)
        
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print(f"  Step {step} error: {e}")
            break
        
        # Get images from observation
        pixels = obs.get("pixels", obs)
        front_rgb = None
        wrist_rgb = None
        
        if "image" in pixels:
            front_rgb = np.asarray(pixels["image"], dtype=np.uint8)
        elif "front_rgb_list" in obs:
            front_list = obs["front_rgb_list"]
            front_rgb = np.asarray(front_list[-1] if isinstance(front_list, list) else front_list, dtype=np.uint8)
        
        if "wrist_image" in pixels:
            wrist_rgb = np.asarray(pixels["wrist_image"], dtype=np.uint8)
        elif "wrist_rgb_list" in obs:
            wrist_list = obs["wrist_rgb_list"]
            wrist_rgb = np.asarray(wrist_list[-1] if isinstance(wrist_list, list) else wrist_list, dtype=np.uint8)
        
        if front_rgb is None or wrist_rgb is None:
            continue
        
        # Resize if needed
        if front_rgb.shape[:2] != (image_size, image_size):
            from PIL import Image
            front_rgb = np.array(Image.fromarray(front_rgb).resize((image_size, image_size), Image.BILINEAR))
            wrist_rgb = np.array(Image.fromarray(wrist_rgb).resize((image_size, image_size), Image.BILINEAR))
        
        # Get state
        agent_pos = obs.get("agent_pos", obs.get("joint_state_list", []))
        if isinstance(agent_pos, list):
            agent_pos = np.asarray(agent_pos[-1] if isinstance(agent_pos, list) and len(agent_pos) > 0 else agent_pos, dtype=np.float32)
        state = np.asarray(agent_pos, dtype=np.float32).flatten()[:8]
        if len(state) < 8:
            state = np.pad(state, (0, 8 - len(state)), mode="constant")
        
        # Get success status
        status = info.get("status", "ongoing")
        is_success = status == "success"
        
        frame = {
            "observation.images.image": front_rgb,
            "observation.images.wrist_image": wrist_rgb,
            "observation.state": state,
            "action": np.asarray(action, dtype=np.float32),
            "next.reward": np.array([float(reward)], dtype=np.float32),
            "next.success": np.array([is_success], dtype=bool),
            "task": TASK_DESCRIPTIONS.get(task, task),
        }
        frames.append(frame)
        success_flags.append(is_success)
        
        # Check for success
        if is_success and not success_detected:
            success_detected = True
            frames_after_success = 0
            print(f"  >>> Success at step {step}, collecting {extra_frames_after_success} more frames...")
        
        if success_detected:
            frames_after_success += 1
            if frames_after_success >= extra_frames_after_success:
                break
        
        terminated_bool = bool(terminated.item()) if hasattr(terminated, "item") else bool(terminated)
        truncated_bool = bool(truncated.item()) if hasattr(truncated, "item") else bool(truncated)
        
        if terminated_bool or truncated_bool:
            break
    
    episode_info = {
        "success": any(success_flags),
        "num_frames": len(frames),
        "episode_idx": -1,  # Will be set by caller
        "seed": None,
        "difficulty": None,
        "object_positions": initial_obj_positions,
    }
    
    return frames, episode_info


# ─── Dataset I/O ─────────────────────────────────────────────────────────

def create_dataset(repo_id, output_dir, fps=10, image_size=256,
                   streaming_encoding=False, encoder_threads=None,
                   image_writer_processes=0, image_writer_threads=8,
                   batch_encoding_size=1, vcodec=None):
    """Create a new LeRobot dataset for RoboMME data."""
    from lerobot.configs.video import rgb_encoder_defaults, RGBEncoderConfig
    
    features = {
        "observation.images.image": {"dtype": "video", "shape": (3, image_size, image_size), "names": ["channels", "height", "width"]},
        "observation.images.wrist_image": {"dtype": "video", "shape": (3, image_size, image_size), "names": ["channels", "height", "width"]},
        "observation.state": {"dtype": "float32", "shape": (8,)},
        "action": {"dtype": "float32", "shape": (8,)},
        "next.reward": {"dtype": "float32", "shape": (1,)},
        "next.success": {"dtype": "bool", "shape": (1,)},
    }
    
    rgb_enc = RGBEncoderConfig(vcodec=vcodec) if vcodec else rgb_encoder_defaults()
    total_writer_threads = image_writer_threads * 2 if image_writer_threads > 0 else 0
    
    return LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=features, root=output_dir,
        robot_type="robomme", use_videos=True,
        image_writer_processes=image_writer_processes, image_writer_threads=total_writer_threads,
        batch_encoding_size=batch_encoding_size, rgb_encoder=rgb_enc,
        encoder_threads=encoder_threads, streaming_encoding=streaming_encoding,
    )


def save_episode_metadata(output_dir, all_episode_infos, task_name):
    """Save episode initial states to JSON file."""
    metadata_file = Path(output_dir) / "episode_initial_states.json"
    
    metadata = {
        "task": task_name,
        "num_episodes": len(all_episode_infos),
        "episodes": [],
    }
    
    for i, info in enumerate(all_episode_infos):
        ep = {
            "episode_index": i,
            "success": bool(info.get("success", False)),
            "num_frames": info.get("num_frames", 0),
        }
        if info.get("episode_idx") is not None:
            ep["episode_idx"] = info["episode_idx"]
        if info.get("seed") is not None:
            ep["seed"] = info["seed"]
        if info.get("difficulty") is not None:
            ep["difficulty"] = info["difficulty"]
        if info.get("object_positions"):
            ep["object_positions"] = info["object_positions"]
        if info.get("_sampling_method"):
            ep["sampling_method"] = info["_sampling_method"]
        
        metadata["episodes"].append(ep)
    
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nEpisode初始环境信息已保存到: {metadata_file}")


# ─── Uniform grid sampling ───────────────────────────────────────────────

def generate_grid_points(bounds, resolution, num_objects):
    """Generate grid points for uniform sampling of object positions.
    
    Args:
        bounds: List of (min, max) tuples for each axis
        resolution: Dict with x, y, z grid resolutions
        num_objects: Number of movable objects
    
    Returns:
        List of configurations, where each config is a list of object positions
    """
    x_bounds, y_bounds, z_bounds = bounds
    
    x_points = np.linspace(x_bounds[0], x_bounds[1], resolution["x"])
    y_points = np.linspace(y_bounds[0], y_bounds[1], resolution["y"])
    z_points = np.linspace(z_bounds[0], z_bounds[1], resolution["z"])
    
    # Generate all combinations for single object
    grid_points = []
    for x in x_points:
        for y in y_points:
            for z in z_points:
                grid_points.append([[x, y, z]])
    
    # For multiple objects, generate combinations
    if num_objects > 1:
        multi_obj_configs = []
        for config in grid_points:
            # For simplicity, we'll use the same grid for all objects
            # but ensure they don't overlap
            for config2 in grid_points:
                if num_objects == 2:
                    multi_obj_configs.append([config[0], config2[0]])
                # Could extend for more objects
        grid_points = multi_obj_configs
    
    return grid_points


def find_episode_for_grid_point(task, target_positions, builder, max_attempts=20):
    """Try to find an episode whose object positions match the target grid point.
    
    Since RoboMME uses seeds to determine positions, we search through episodes
    to find ones with positions close to our target.
    
    Returns (episode_idx, actual_positions) or (None, None) if not found.
    """
    # For now, we'll use random episode indices and hope the seed produces
    # positions close to our target. In practice, you might want to:
    # 1. Pre-compute a mapping from episode_idx to object positions
    # 2. Select episodes closest to target positions
    
    # Simple approach: try random episode indices
    rng = np.random.RandomState(42)
    num_episodes = builder.get_episode_num()
    
    best_idx = None
    best_dist = float("inf")
    best_positions = None
    
    for _ in range(max_attempts):
        idx = rng.randint(0, num_episodes)
        try:
            env, _ = create_robomme_env(task, idx, max_steps=10)
            env.reset()
            positions = get_object_positions_from_env(env)
            env.close()
            
            if positions:
                # Calculate distance to target
                pos_list = list(positions.values())
                if len(pos_list) == len(target_positions):
                    dist = 0
                    for p1, p2 in zip(pos_list, target_positions):
                        dist += np.linalg.norm(np.array(p1) - np.array(p2))
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
                        best_positions = positions
        except Exception:
            continue
    
    if best_idx is not None:
        return best_idx, best_positions
    
    return None, None


# ─── Phase 1: Random collection ──────────────────────────────────────────

def phase_random(args, dataset, task, start_ep_idx=0):
    """Phase 1: Random collection by iterating through episode indices."""
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    
    print(f"\n{'='*60}")
    print(f"阶段1: 随机采集 ({args.num_random_episodes} 个 episode)")
    print(f"{'='*60}")
    
    builder = BenchmarkEnvBuilder(
        env_id=task, dataset=args.dataset_split,
        action_space=args.action_space, gui_render=False,
        max_steps=args.max_steps,
    )
    num_episodes = builder.get_episode_num()
    print(f"可用 episode 总数: {num_episodes}")
    
    episode_infos = []
    success_count = 0
    episode_idx = args.seed_start
    consecutive_failures = 0
    
    while success_count < args.num_random_episodes:
        # Wrap around if we exceed available episodes
        actual_idx = episode_idx % num_episodes
        
        ep_start = time.time()
        try:
            env, builder = create_robomme_env(
                task, actual_idx, dataset=args.dataset_split,
                action_space=args.action_space, max_steps=args.max_steps,
            )
            frames, ep_info = run_episode(
                env, task, args.max_steps, args.image_size,
                args.extra_frames_after_success,
            )
            env.close()
            
            # Get initial state info
            meta = get_episode_metadata_safe(builder, task, actual_idx)
            ep_info["episode_idx"] = actual_idx
            ep_info["seed"] = meta.get("seed")
            ep_info["difficulty"] = meta.get("difficulty")
            
            if ep_info["success"]:
                for frame in frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                episode_infos.append(ep_info)
                success_count += 1
                consecutive_failures = 0
                
                elapsed = time.time() - ep_start
                seed_str = f" seed={meta.get('seed', '?')}" if meta.get("seed") else ""
                print(f"  [R] Episode {start_ep_idx + success_count:4d} | Frames: {ep_info['num_frames']:4d} | Success{seed_str} | {elapsed:.1f}s")
            else:
                consecutive_failures += 1
                elapsed = time.time() - ep_start
                print(f"  [R] FAILED (ep_idx={actual_idx}) | Frames: {ep_info['num_frames']:4d} | {elapsed:.1f}s")
            
        except Exception as e:
            consecutive_failures += 1
            print(f"  [R] ERROR (ep_idx={actual_idx}): {e}")
        
        episode_idx += 1
        
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"\n警告: 连续失败 {MAX_CONSECUTIVE_FAILURES} 次，跳过并继续...")
            consecutive_failures = 0
    
    print(f"阶段1 完成: 成功 {success_count}/{args.num_random_episodes}")
    return episode_infos


# ─── Phase 2: Uniform collection ─────────────────────────────────────────

def phase_uniform(args, dataset, task, start_ep_idx=0):
    """Phase 2: Uniform grid-based collection."""
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    
    print(f"\n{'='*60}")
    print(f"阶段2: 均匀采样采集 ({args.num_uniform_episodes} 个 episode)")
    print(f"{'='*60}")
    
    # Get task bounds
    bounds_config = TASK_OBJECT_BOUNDS.get(task, TASK_OBJECT_BOUNDS["PickXtimes"])
    bounds = bounds_config["bounds"]
    num_objects = bounds_config["num_objects"]
    
    # Generate grid points
    grid_points = generate_grid_points(bounds, GRID_RESOLUTION, num_objects)
    print(f"生成 {len(grid_points)} 个网格点 (分辨率: {GRID_RESOLUTION})")
    
    # Shuffle grid points for better coverage
    rng = np.random.RandomState(123)
    rng.shuffle(grid_points)
    
    # Limit to requested number
    target_configs = grid_points[:args.num_uniform_episodes]
    print(f"目标采集 {len(target_configs)} 个 episode")
    
    builder = BenchmarkEnvBuilder(
        env_id=task, dataset=args.dataset_split,
        action_space=args.action_space, gui_render=False,
        max_steps=args.max_steps,
    )
    
    episode_infos = []
    success_count = 0
    total_attempts = 0
    config_idx = 0
    use_seed_fallback = False
    fallback_idx = 0
    
    perturbation_rng = np.random.RandomState(456)
    
    while success_count < args.num_uniform_episodes:
        if config_idx >= len(target_configs):
            if not use_seed_fallback:
                print(f"  网格点已用完，切换到基于 episode_idx 的随机采样...")
                use_seed_fallback = True
            # Generate more random episode indices
            num_episodes = builder.get_episode_num()
            target_configs.extend([None] * (args.num_uniform_episodes - success_count))
        
        target_config = target_configs[config_idx]
        config_idx += 1
        total_attempts += 1
        
        collected = False
        consecutive_failures = 0
        
        for retry in range(MAX_PERTURB_RETRIES + 1):
            try:
                if target_config is not None and retry == 0:
                    # Try to find episode matching this grid point
                    ep_idx, actual_positions = find_episode_for_grid_point(
                        task, target_config, builder, max_attempts=10
                    )
                    if ep_idx is None:
                        # Fall back to random episode
                        ep_idx = perturbation_rng.randint(0, builder.get_episode_num())
                        actual_positions = None
                else:
                    # Random episode index
                    ep_idx = perturbation_rng.randint(0, builder.get_episode_num())
                    actual_positions = None
                
                env, builder = create_robomme_env(
                    task, ep_idx, dataset=args.dataset_split,
                    action_space=args.action_space, max_steps=args.max_steps,
                )
                frames, ep_info = run_episode(
                    env, task, args.max_steps, args.image_size,
                    args.extra_frames_after_success,
                )
                env.close()
                
                if ep_info["success"]:
                    for frame in frames:
                        dataset.add_frame(frame)
                    dataset.save_episode()
                    
                    meta = get_episode_metadata_safe(builder, task, ep_idx)
                    ep_info["episode_idx"] = ep_idx
                    ep_info["seed"] = meta.get("seed")
                    ep_info["difficulty"] = meta.get("difficulty")
                    ep_info["_sampling_method"] = "fallback_random" if use_seed_fallback else "uniform_grid"
                    
                    episode_infos.append(ep_info)
                    success_count += 1
                    
                    obj_str = ""
                    if ep_info.get("object_positions"):
                        obj_str = f" | objects: {len(ep_info['object_positions'])} positions recorded"
                    print(f"  [U] Episode {start_ep_idx + success_count:4d} | Frames: {ep_info['num_frames']:4d} | Success{obj_str}")
                    collected = True
                    break
                else:
                    consecutive_failures += 1
                    print(f"  [U] Episode FAILED (ep_idx={ep_idx}), skipping...")
                    if consecutive_failures >= 3:
                        break
            
            except Exception as e:
                consecutive_failures += 1
                print(f"  [U] ERROR (ep_idx={ep_idx}): {e}")
                if consecutive_failures >= 3:
                    break
        
        if not collected and use_seed_fallback:
            fallback_idx += 1
    
    print(f"阶段2 完成: 成功 {success_count}/{args.num_uniform_episodes} (总尝试 {total_attempts})")
    fallback_count = sum(1 for info in episode_infos if info.get("_sampling_method") == "fallback_random")
    grid_count = success_count - fallback_count
    print(f"  - 网格采样成功: {grid_count} 个")
    print(f"  - Fallback 随机采样成功: {fallback_count} 个")
    return episode_infos


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="两阶段采集 RoboMME 数据集：随机 + Uniform 均匀采样",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", type=str, default="PickXtimes",
                        help="RoboMME task name (e.g., PickXtimes, BinFill, MoveCube)")
    parser.add_argument("--num-random-episodes", type=int, default=300,
                        help="阶段1: 随机采集的 episode 数量 (默认: 300)")
    parser.add_argument("--num-uniform-episodes", type=int, default=100,
                        help="阶段2: Uniform 均匀采样的 episode 数量 (默认: 100)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed-start", type=int, default=0,
                        help="起始 episode index (默认: 0)")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--extra-frames-after-success", type=int, default=EXTRA_FRAMES_AFTER_SUCCESS)
    parser.add_argument("--dataset-split", type=str, default="test",
                        help="Dataset split: train, val, or test")
    parser.add_argument("--action-space", type=str, default="joint_angle",
                        help="Action space: joint_angle (8-D) or ee_pose (7-D)")
    parser.add_argument("--streaming-encoding", action="store_true", default=False)
    parser.add_argument("--encoder-threads", type=int, default=None)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--batch-encoding-size", type=int, default=1)
    parser.add_argument("--vcodec", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = f"./outputs/robomme_{args.task}"
    if args.repo_id is None:
        args.repo_id = f"work2/robomme_{args.task}"
    
    output_dir = Path(args.output_dir)
    is_resume = output_dir.exists() and (output_dir / "episode_initial_states.json").exists()
    
    print("=" * 80)
    print("RoboMME 两阶段数据采集 (随机 + Uniform)")
    print("=" * 80)
    print(f"任务: {args.task}")
    print(f"阶段1 随机采集: {args.num_random_episodes} episodes")
    print(f"阶段2 Uniform采集: {args.num_uniform_episodes} episodes")
    print(f"输出目录: {args.output_dir}")
    print(f"Repo ID: {args.repo_id}")
    print(f"FPS: {args.fps}")
    print(f"图像分辨率: {args.image_size}x{args.image_size}")
    print(f"Episode起始: {args.seed_start}")
    print(f"数据集split: {args.dataset_split}")
    print(f"动作空间: {args.action_space}")
    print(f"流式编码: {'是' if args.streaming_encoding else '否'}")
    print(f"模式: {'Resume (追加到已有数据集)' if is_resume else '从头开始'}")
    print("=" * 80)
    
    if args.num_random_episodes == 0 and args.num_uniform_episodes == 0:
        print("错误: 两个阶段的 episode 数量都为 0，无需采集。")
        sys.exit(1)
    
    # Validate task name
    from robomme.env_record_wrapper import BenchmarkEnvBuilder
    valid_tasks = BenchmarkEnvBuilder(env_id=args.task, dataset=args.dataset_split).get_task_list()
    if args.task not in valid_tasks:
        print(f"错误: 无效的任务名 '{args.task}'。可用任务: {valid_tasks}")
        sys.exit(1)
    
    # Create/load dataset
    if is_resume:
        print(f"\n加载已有LeRobot数据集 (resume)...")
        dataset = LeRobotDataset.resume(repo_id=args.repo_id, root=args.output_dir)
        existing_episodes = dataset.num_episodes
        print(f"已有 {existing_episodes} 个 episode")
    else:
        if output_dir.exists():
            print(f"警告: 输出目录已存在，将删除并重建: {args.output_dir}")
            shutil.rmtree(output_dir)
        print(f"\n创建LeRobot数据集（视频格式）...")
        dataset = create_dataset(
            args.repo_id, args.output_dir, args.fps, args.image_size,
            streaming_encoding=args.streaming_encoding, encoder_threads=args.encoder_threads,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
            batch_encoding_size=args.batch_encoding_size, vcodec=args.vcodec,
        )
        existing_episodes = 0
        print(f"数据集创建成功: {args.output_dir}")
    
    all_episode_infos = []
    start_time = time.time()
    
    # Phase 1: Random collection
    if args.num_random_episodes > 0:
        random_infos = phase_random(args, dataset, args.task, start_ep_idx=existing_episodes)
        all_episode_infos.extend(random_infos)
        existing_episodes += len(random_infos)
    
    # Phase 2: Uniform collection
    if args.num_uniform_episodes > 0:
        uniform_infos = phase_uniform(args, dataset, args.task, start_ep_idx=existing_episodes)
        all_episode_infos.extend(uniform_infos)
    
    # Finalize dataset
    print("\n" + "-" * 80)
    print("正在保存数据集...")
    dataset.finalize()
    
    # Save metadata
    save_episode_metadata(args.output_dir, all_episode_infos, args.task)
    
    total_time = time.time() - start_time
    total_success = len(all_episode_infos)
    
    print("\n" + "=" * 80)
    print("采集完成！")
    print("=" * 80)
    print(f"本次新增总Episode: {total_success}")
    print(f"数据集总Episode: {dataset.num_episodes}")
    print(f"总用时: {total_time:.1f}s")
    print(f"数据集路径: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()