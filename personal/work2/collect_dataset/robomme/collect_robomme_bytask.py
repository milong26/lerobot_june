#!/usr/bin/env python
"""
RoboMME Expert Dataset Collector

使用 RoboMME 官方的 FailAwarePandaArmMotionPlanningSolver 采集 expert demonstration。

核心执行链:
  seed=N
  -> gym.make(task, seed=N)
  -> env.reset()
  -> FailAwarePandaArmMotionPlanningSolver(env)
  -> 遍历 env.unwrapped.task_list
  -> solve_callable(env, planner) 内部调用 planner.move_to_pose_with_screw() -> env.step()
  -> EpisodeRecorder wrapper 拦截每次 env.step() 记录 (action, obs, reward, success)
  -> env.unwrapped.evaluate() 检查 success
  -> 成功才写入 LeRobotDataset

两阶段采集:
  阶段1 — 随机采样: seed=0,1,2,... 递增，使用 expert planner 采集 N 个成功 episode
  阶段2 — 均匀采样: 对可动物体 spawn region 进行网格化采样 M 个 episode

使用示例:
    python collect_robomme_bytask.py \
        --task PickXtimes \
        --num-random-episodes 300 \
        --num-uniform-episodes 100 \
        --output-dir personal/work2/dataset_view_robomme/PickXtimes/ \
        --repo-id work2/robomme_PickXtimes
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

nvidia_icd = "/usr/share/vulkan/icd.d/nvidia_icd.json"
if os.path.exists(nvidia_icd):
    os.environ["VK_ICD_FILENAMES"] = nvidia_icd

os.environ["SAPIEN_VULKAN_DEVICE_INDEX"] = "0"

import gymnasium as gym
import numpy as np

import robomme.robomme_env  # 触发 robomme 环境注册

from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASK_DESCRIPTIONS = {
    "BinFill": "Fill the target bin with the correct number of cubes",
    "PickXtimes": "Pick the indicated cube the specified number of times",
    "SwingXtimes": "Swing the object the specified number of times",
    "StopCube": "Grasp and stop the moving cube",
    "VideoUnmask": "Pick the cube shown in the reference video",
    "VideoUnmaskSwap": "Pick the cube matching the reference video after a swap",
    "ButtonUnmask": "Press the button indicated by the reference",
    "ButtonUnmaskSwap": "Press the correct button after objects are swapped",
    "PickHighlight": "Pick the highlighted cube",
    "VideoRepick": "Repick the cube shown in the reference video",
    "VideoPlaceButton": "Place the cube on the button shown in the video",
    "VideoPlaceOrder": "Place cubes in the order shown in the video",
    "MoveCube": "Move the cube to the target location",
    "InsertPeg": "Insert the peg into the target hole",
    "PatternLock": "Unlock the pattern by pressing buttons in sequence",
    "RouteStick": "Route the stick through the required waypoints",
}

TASK_SPAWN_CONFIGS = {
    "PickXtimes": {
        "cube_region_center": [-0.1, 0.0],
        "cube_region_half_size": 0.2,
        "target_region_center": [-0.1, 0.0],
        "target_region_half_size": 0.2,
        "z": 0.02,
        "difficulty_configs": {
            "easy": {"num_colors": 1, "num_cubes_range": (1, 3)},
            "medium": {"num_colors": 3, "num_cubes_range": (1, 3)},
            "hard": {"num_colors": 3, "num_cubes_range": (4, 5)},
        },
    },
}

GRID_RESOLUTION = {"x": 5, "y": 5}
MAX_PERTURB_RETRIES = 4
EXTRA_FRAMES_AFTER_SUCCESS = 10
MAX_CONSECUTIVE_FAILURES = 50


def create_raw_env(task, seed, max_steps=300):
    from robomme.env_record_wrapper import BenchmarkEnvBuilder

    builder = BenchmarkEnvBuilder(
        env_id=task,
        dataset="test",
        action_space="joint_angle",
        gui_render=False,
        max_steps=max_steps,
    )
    # 使用 episode_idx=seed 来让 builder 从元数据中解析配置
    # 但如果元数据中没有对应的 episode，会使用默认配置
    env = builder.make_env_for_episode(episode_idx=seed, max_steps=max_steps)
    return env


def get_state_from_obs(obs):
    joint_state = obs.get("joint_state_list", [])
    gripper_state = obs.get("gripper_state_list", [])

    if isinstance(joint_state, list) and len(joint_state) > 0:
        joint_state = joint_state[-1] if isinstance(joint_state[0], list) else joint_state
    joint_arr = np.asarray(joint_state, dtype=np.float32).flatten()

    if isinstance(gripper_state, list) and len(gripper_state) > 0:
        gripper_state = gripper_state[-1] if isinstance(gripper_state[0], list) else gripper_state
    gripper_arr = np.asarray(gripper_state, dtype=np.float32).flatten()

    state = np.concatenate([joint_arr, gripper_arr])

    if len(state) < 8:
        state = np.pad(state, (0, 8 - len(state)), mode="constant")
    elif len(state) > 8:
        state = state[:8]

    return state


class EpisodeRecorder:
    def __init__(self, env, image_size=256):
        self.env = env
        self.image_size = image_size
        self.frames = []
        self.success_detected = False
        self.frames_after_success = 0
        self.original_step = None
        self.initial_obj_positions = None

    def _extract_images(self, obs):
        pixels = obs.get("pixels", obs)
        front_rgb = None
        wrist_rgb = None

        if "image" in pixels:
            front_rgb = np.asarray(pixels["image"], dtype=np.uint8)
        elif "front_rgb_list" in obs:
            front_list = obs["front_rgb_list"]
            front_rgb = np.asarray(
                front_list[-1] if isinstance(front_list, list) else front_list,
                dtype=np.uint8,
            )

        if "wrist_image" in pixels:
            wrist_rgb = np.asarray(pixels["wrist_image"], dtype=np.uint8)
        elif "wrist_rgb_list" in obs:
            wrist_list = obs["wrist_rgb_list"]
            wrist_rgb = np.asarray(
                wrist_list[-1] if isinstance(wrist_list, list) else wrist_list,
                dtype=np.uint8,
            )

        return front_rgb, wrist_rgb

    def _resize_if_needed(self, img):
        if img is None:
            return None
        if img.shape[:2] == (self.image_size, self.image_size):
            return img
        from PIL import Image
        return np.array(
            Image.fromarray(img).resize(
                (self.image_size, self.image_size), Image.BILINEAR
            )
        )

    def _recording_step(self, action):
        obs, reward, terminated, truncated, info = self.original_step(action)

        front_rgb, wrist_rgb = self._extract_images(obs)

        if front_rgb is not None and wrist_rgb is not None:
            front_rgb = self._resize_if_needed(front_rgb)
            wrist_rgb = self._resize_if_needed(wrist_rgb)

            state = get_state_from_obs(obs)

            action_arr = np.asarray(action, dtype=np.float32).flatten()
            if len(action_arr) < 8:
                action_arr = np.pad(action_arr, (0, 8 - len(action_arr)), mode="constant")
            elif len(action_arr) > 8:
                action_arr = action_arr[:8]

            frame = {
                "observation.images.image": front_rgb,
                "observation.images.wrist_image": wrist_rgb,
                "observation.state": state,
                "action": action_arr,
                "next.reward": np.array([float(reward)], dtype=np.float32),
                "next.success": np.array([info.get("success", False)], dtype=bool),
            }
            self.frames.append(frame)

        status = info.get("status", "ongoing")
        is_success = status == "success"

        if is_success and not self.success_detected:
            self.success_detected = True
            self.frames_after_success = 0

        if self.success_detected:
            self.frames_after_success += 1

        return obs, reward, terminated, truncated, info

    def __enter__(self):
        self.original_step = self.env.step
        self.env.step = self._recording_step
        return self

    def __exit__(self, *args):
        self.env.step = self.original_step

    def is_success(self):
        return self.success_detected

    def get_frames(self):
        return self.frames


def get_object_positions_from_env(env):
    positions = {}
    try:
        inner = env
        while hasattr(inner, "env") and not hasattr(inner, "scene"):
            inner = inner.env

        if hasattr(inner, "scene") and inner.scene is not None:
            scene = inner.scene
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


def run_episode_with_planner(env, task, max_steps=300, image_size=256, extra_frames=10):
    """
    运行一个 episode。
    
    使用 BenchmarkEnvBuilder 创建的环境已经包含 DemonstrationWrapper，
    在 reset() 时会自动生成 demonstration trajectory。
    我们需要拦截 DemonstrationWrapper._step_batch(action) 调用，同时记录 action 和 obs。
    """
    initial_obj_positions = get_object_positions_from_env(env)

    # 找到 DemonstrationWrapper
    demo_wrapper = env
    while hasattr(demo_wrapper, "env") and not hasattr(demo_wrapper, "demonstration_data"):
        demo_wrapper = demo_wrapper.env
    
    # 拦截 DemonstrationWrapper 的 _step_batch 方法
    collected_steps = []
    original_step_batch = demo_wrapper._step_batch
    
    def _recording_step_batch(action):
        result = original_step_batch(action)
        obs_batch, reward_batch, terminated_batch, truncated_batch, info_batch = result
        
        # 将 batch 转换为单个步骤
        import torch
        batch_size = int(reward_batch.numel()) if hasattr(reward_batch, 'numel') else 0
        
        for step_idx in range(batch_size):
            step_obs = {}
            step_info = {}
            
            for key in obs_batch:
                val = obs_batch[key]
                if isinstance(val, torch.Tensor):
                    step_obs[key] = val[step_idx].cpu().numpy() if val.ndim > 0 else val.cpu().numpy()
                elif isinstance(val, list) and len(val) > step_idx:
                    step_obs[key] = val[step_idx]
                else:
                    step_obs[key] = val
            
            for key in info_batch:
                val = info_batch[key]
                if isinstance(val, torch.Tensor):
                    step_info[key] = val[step_idx].cpu().numpy() if val.ndim > 0 else val.cpu().numpy()
                elif isinstance(val, list) and len(val) > step_idx:
                    step_info[key] = val[step_idx]
                else:
                    step_info[key] = val
            
            collected_steps.append({
                "action": action,
                "obs": step_obs,
                "reward": float(reward_batch[step_idx].cpu().numpy()) if hasattr(reward_batch, 'cpu') else float(reward_batch[step_idx]),
                "terminated": bool(terminated_batch[step_idx].cpu().numpy()) if hasattr(terminated_batch, 'cpu') else bool(terminated_batch[step_idx]),
                "truncated": bool(truncated_batch[step_idx].cpu().numpy()) if hasattr(truncated_batch, 'cpu') else bool(truncated_batch[step_idx]),
                "info": step_info,
            })
        
        return result
    
    demo_wrapper._step_batch = _recording_step_batch

    try:
        # reset() 会触发 DemonstrationWrapper 自动生成 demonstration trajectory
        obs, info = env.reset()
        
        # 检查 demonstration 是否成功（从 DemonstrationWrapper 获取）
        episode_success = getattr(demo_wrapper, "episode_success", False)
        num_demo_steps = len(collected_steps)
        
        print(f"  [D] episode_success={episode_success}, collected_steps={num_demo_steps}")
        
        if not episode_success or num_demo_steps == 0:
            return [], {"success": False, "num_frames": 0, "object_positions": {}}
        
        # 处理收集到的步骤
        with EpisodeRecorder(env, image_size) as recorder:
            recorder.initial_obj_positions = initial_obj_positions
            
            for step_data in collected_steps:
                action = step_data["action"]
                obs = step_data["obs"]
                info = step_data["info"]
                
                # 使用实际的 action 执行 recording step
                recorder._recording_step(action)
                
                # 检查是否成功
                status = info.get("status", "ongoing")
                if status == "success" and not recorder.success_detected:
                    recorder.success_detected = True
                    recorder.frames_after_success = 0
                
                if recorder.success_detected:
                    recorder.frames_after_success += 1
        
        return recorder.get_frames(), {
            "success": recorder.is_success(),
            "num_frames": len(recorder.get_frames()),
            "object_positions": recorder.initial_obj_positions,
        }
    finally:
        demo_wrapper._step_batch = original_step_batch


def create_dataset(repo_id, output_dir, fps=10, image_size=256,
                   streaming_encoding=False, encoder_threads=None,
                   image_writer_processes=0, image_writer_threads=8,
                   batch_encoding_size=1, vcodec=None):
    from lerobot.configs.video import rgb_encoder_defaults, RGBEncoderConfig

    features = {
        "observation.images.image": {
            "dtype": "video",
            "shape": (3, image_size, image_size),
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist_image": {
            "dtype": "video",
            "shape": (3, image_size, image_size),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {"dtype": "float32", "shape": (8,)},
        "action": {"dtype": "float32", "shape": (8,)},
        "next.reward": {"dtype": "float32", "shape": (1,)},
        "next.success": {"dtype": "bool", "shape": (1,)},
    }

    rgb_enc = RGBEncoderConfig(vcodec=vcodec) if vcodec else rgb_encoder_defaults()
    total_writer_threads = image_writer_threads * 2 if image_writer_threads > 0 else 0

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type="robomme",
        use_videos=True,
        image_writer_processes=image_writer_processes,
        image_writer_threads=total_writer_threads,
        batch_encoding_size=batch_encoding_size,
        rgb_encoder=rgb_enc,
        encoder_threads=encoder_threads,
        streaming_encoding=streaming_encoding,
    )


def save_episode_metadata(output_dir, all_episode_infos, task_name):
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

    print(f"\nEpisode 初始环境信息已保存到: {metadata_file}")


def generate_grid_points(task_config, resolution):
    region_center = task_config["cube_region_center"]
    region_half = task_config["cube_region_half_size"]
    z = task_config.get("z", 0.02)

    x_min = region_center[0] - region_half
    x_max = region_center[0] + region_half
    y_min = region_center[1] - region_half
    y_max = region_center[1] + region_half

    x_points = np.linspace(x_min, x_max, resolution["x"])
    y_points = np.linspace(y_min, y_max, resolution["y"])

    grid_points = []
    for x in x_points:
        for y in y_points:
            grid_points.append([x, y, z])

    return grid_points


def set_object_positions(env, positions):
    try:
        inner = env
        while hasattr(inner, "env") and not hasattr(inner, "scene"):
            inner = inner.env

        if hasattr(inner, "scene") and inner.scene is not None:
            scene = inner.scene
            if hasattr(scene, "actors"):
                actors = scene.actors
                if actors is not None and len(actors) >= len(positions):
                    for i, pos in enumerate(positions):
                        actor = actors[i]
                        if hasattr(actor, "set_pose"):
                            from sapien import Pose
                            new_pose = Pose(p=pos)
                            actor.set_pose(new_pose)
                    return True
    except Exception as e:
        print(f"  Warning: Could not set object positions: {e}")
    return False


def phase_random(args, dataset, task, start_ep_idx=0):
    print(f"\n{'='*60}")
    print(f"阶段1: 随机采集 ({args.num_random_episodes} 个 episode)")
    print(f"{'='*60}")

    episode_infos = []
    success_count = 0
    seed = args.seed_start
    consecutive_failures = 0

    while success_count < args.num_random_episodes:
        ep_start = time.time()
        try:
            env = create_raw_env(task, seed=seed, max_steps=args.max_steps)

            frames, ep_info = run_episode_with_planner(
                env, task, args.max_steps, args.image_size, args.extra_frames_after_success
            )
            env.close()

            if ep_info["success"]:
                for frame in frames:
                    dataset.add_frame(frame)
                dataset.save_episode()

                ep_info["seed"] = seed
                ep_info["episode_idx"] = start_ep_idx + success_count
                ep_info["difficulty"] = seed % 3
                episode_infos.append(ep_info)
                success_count += 1
                consecutive_failures = 0

                elapsed = time.time() - ep_start
                print(
                    f"  [R] Episode {start_ep_idx + success_count:4d} | "
                    f"Frames: {ep_info['num_frames']:4d} | "
                    f"Success (seed={seed}) | {elapsed:.1f}s"
                )
            else:
                consecutive_failures += 1
                elapsed = time.time() - ep_start
                print(
                    f"  [R] FAILED (seed={seed}) | "
                    f"Frames: {ep_info['num_frames']:4d} | {elapsed:.1f}s"
                )

        except Exception as e:
            consecutive_failures += 1
            print(f"  [R] ERROR (seed={seed}): {e}")

        seed += 1

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"\n警告: 连续失败 {MAX_CONSECUTIVE_FAILURES} 次，跳过并继续...")
            consecutive_failures = 0

    print(f"阶段1 完成: 成功 {success_count}/{args.num_random_episodes}")
    return episode_infos


def phase_uniform(args, dataset, task, start_ep_idx=0):
    print(f"\n{'='*60}")
    print(f"阶段2: 均匀采样采集 ({args.num_uniform_episodes} 个 episode)")
    print(f"{'='*60}")

    task_config = TASK_SPAWN_CONFIGS.get(task, TASK_SPAWN_CONFIGS["PickXtimes"])
    grid_points = generate_grid_points(task_config, GRID_RESOLUTION)
    print(f"生成 {len(grid_points)} 个网格点 (分辨率: {GRID_RESOLUTION})")

    rng = np.random.RandomState(123)
    rng.shuffle(grid_points)

    target_configs = grid_points[:args.num_uniform_episodes]
    print(f"目标采集 {len(target_configs)} 个 episode")

    episode_infos = []
    success_count = 0
    total_attempts = 0
    config_idx = 0
    use_seed_fallback = False

    perturbation_rng = np.random.RandomState(456)

    while success_count < args.num_uniform_episodes:
        if config_idx >= len(target_configs):
            if not use_seed_fallback:
                print(f"  网格点已用完，切换到基于 seed 的随机采样...")
                use_seed_fallback = True
            target_configs.extend([None] * (args.num_uniform_episodes - success_count))

        target_config = target_configs[config_idx]
        config_idx += 1
        total_attempts += 1

        consecutive_failures = 0

        for retry in range(MAX_PERTURB_RETRIES + 1):
            try:
                seed = perturbation_rng.randint(0, 10000)
                env = create_raw_env(task, seed=seed, max_steps=args.max_steps)

                if target_config is not None and retry == 0:
                    env.reset()
                    positions = get_object_positions_from_env(env)
                    pos_list = list(positions.values())
                    if pos_list:
                        set_object_positions(env, [target_config])

                frames, ep_info = run_episode_with_planner(
                    env, task, args.max_steps, args.image_size, args.extra_frames_after_success
                )
                env.close()

                if ep_info["success"]:
                    for frame in frames:
                        dataset.add_frame(frame)
                    dataset.save_episode()

                    ep_info["seed"] = seed
                    ep_info["episode_idx"] = start_ep_idx + success_count
                    ep_info["_sampling_method"] = (
                        "fallback_random" if use_seed_fallback else "uniform_grid"
                    )

                    episode_infos.append(ep_info)
                    success_count += 1

                    print(
                        f"  [U] Episode {start_ep_idx + success_count:4d} | "
                        f"Frames: {ep_info['num_frames']:4d} | "
                        f"Success (seed={seed})"
                    )
                    break
                else:
                    consecutive_failures += 1
                    print(f"  [U] FAILED (seed={seed}), retry {retry+1}/{MAX_PERTURB_RETRIES}")
                    if consecutive_failures >= 3:
                        break

            except Exception as e:
                consecutive_failures += 1
                print(f"  [U] ERROR (seed={seed}): {e}")
                if consecutive_failures >= 3:
                    break

    print(f"阶段2 完成: 成功 {success_count}/{args.num_uniform_episodes} (总尝试 {total_attempts})")
    fallback_count = sum(
        1 for info in episode_infos if info.get("_sampling_method") == "fallback_random"
    )
    grid_count = success_count - fallback_count
    print(f"  - 网格采样成功: {grid_count} 个")
    print(f"  - Fallback 随机采样成功: {fallback_count} 个")
    return episode_infos


def main():
    parser = argparse.ArgumentParser(
        description="RoboMME Expert Dataset Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task", type=str, default="PickXtimes", help="RoboMME task name"
    )
    parser.add_argument(
        "--num-random-episodes",
        type=int,
        default=300,
        help="Phase 1: number of random episodes (default: 300)",
    )
    parser.add_argument(
        "--num-uniform-episodes",
        type=int,
        default=100,
        help="Phase 2: number of uniform episodes (default: 100)",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed-start", type=int, default=0, help="Starting seed (default: 0)")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument(
        "--extra-frames-after-success",
        type=int,
        default=EXTRA_FRAMES_AFTER_SUCCESS,
    )
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
    print("RoboMME Expert Dataset Collector (Random + Uniform)")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Phase 1 Random: {args.num_random_episodes} episodes")
    print(f"Phase 2 Uniform: {args.num_uniform_episodes} episodes")
    print(f"Output dir: {args.output_dir}")
    print(f"Repo ID: {args.repo_id}")
    print(f"FPS: {args.fps}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Seed start: {args.seed_start}")
    print(f"Mode: {'Resume' if is_resume else 'Fresh'}")
    print("=" * 80)

    if args.num_random_episodes == 0 and args.num_uniform_episodes == 0:
        print("Error: Both episode counts are 0, nothing to collect.")
        sys.exit(1)

    if is_resume:
        print(f"\nLoading existing dataset (resume)...")
        dataset = LeRobotDataset.resume(repo_id=args.repo_id, root=args.output_dir)
        existing_episodes = dataset.num_episodes
        print(f"Existing episodes: {existing_episodes}")
    else:
        if output_dir.exists():
            print(f"Warning: Output dir exists, will recreate: {args.output_dir}")
            shutil.rmtree(output_dir)
        print(f"\nCreating LeRobot dataset (video format)...")
        dataset = create_dataset(
            args.repo_id,
            args.output_dir,
            args.fps,
            args.image_size,
            streaming_encoding=args.streaming_encoding,
            encoder_threads=args.encoder_threads,
            image_writer_processes=args.image_writer_processes,
            image_writer_threads=args.image_writer_threads,
            batch_encoding_size=args.batch_encoding_size,
            vcodec=args.vcodec,
        )
        existing_episodes = 0
        print(f"Dataset created: {args.output_dir}")

    all_episode_infos = []
    start_time = time.time()

    if args.num_random_episodes > 0:
        random_infos = phase_random(args, dataset, args.task, start_ep_idx=existing_episodes)
        all_episode_infos.extend(random_infos)
        existing_episodes += len(random_infos)

    if args.num_uniform_episodes > 0:
        uniform_infos = phase_uniform(args, dataset, args.task, start_ep_idx=existing_episodes)
        all_episode_infos.extend(uniform_infos)

    print("\n" + "-" * 80)
    print("Saving dataset...")
    dataset.finalize()

    save_episode_metadata(args.output_dir, all_episode_infos, args.task)

    total_time = time.time() - start_time
    total_success = len(all_episode_infos)

    print("\n" + "=" * 80)
    print("Collection complete!")
    print("=" * 80)
    print(f"Total episodes: {total_success}")
    print(f"Dataset episodes: {dataset.num_episodes}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Dataset path: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()