#!/usr/bin/env python
"""
RoboMME Expert Dataset Collector

使用 RoboMME 官方的 FailAwarePandaArmMotionPlanningSolver 采集 expert demonstration。

核心执行链:
  seed=N
  -> gym.make(task, seed=N)
  -> env.reset()
  -> RobommeRecordWrapper 拦截 env.step()
  -> FailAwarePandaArmMotionPlanningSolver(env)
  -> 遍历 env.unwrapped.task_list
  -> solve_callable(env, planner) 内部调用 planner.move_to_pose_with_screw() -> env.step()
  -> env.unwrapped.evaluate() 检查 success
  -> 成功轨迹写入 LeRobotDataset

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
    """创建原始 RoboMME 环境，使用 gym.make 而不是 BenchmarkEnvBuilder。
    
    这样 seed=0,1,2,... 是真正独立的随机环境，不受官方 test split 限制。
    """
    env = gym.make(
        task,
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        seed=seed,
    )
    return env


def get_state_from_obs(obs):
    """从 RoboMME observation 提取 LeRobot state 格式 (joint + gripper, 8维)。"""
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


def extract_images_from_obs(obs, image_size=256):
    """从 RoboMME observation 提取 front/wrist 图像。"""
    front_rgb = None
    wrist_rgb = None

    if "front_rgb_list" in obs:
        front_list = obs["front_rgb_list"]
        front_rgb = np.asarray(
            front_list[-1] if isinstance(front_list, list) else front_list,
            dtype=np.uint8,
        )

    if "wrist_rgb_list" in obs:
        wrist_list = obs["wrist_rgb_list"]
        wrist_rgb = np.asarray(
            wrist_list[-1] if isinstance(wrist_list, list) else wrist_list,
            dtype=np.uint8,
        )

    if front_rgb is not None and front_rgb.shape[:2] != (image_size, image_size):
        from PIL import Image
        front_rgb = np.array(
            Image.fromarray(front_rgb).resize(
                (image_size, image_size), Image.BILINEAR
            )
        )

    if wrist_rgb is not None and wrist_rgb.shape[:2] != (image_size, image_size):
        from PIL import Image
        wrist_rgb = np.array(
            Image.fromarray(wrist_rgb).resize(
                (image_size, image_size), Image.BILINEAR
            )
        )

    return front_rgb, wrist_rgb


class StepRecorder:
    """轻量级 wrapper，拦截 env.step() 记录 (action, obs, info)。
    
    用于在 planner 执行过程中收集完整的 trajectory 数据。
    """
    def __init__(self, env, image_size=256):
        self.env = env
        self.image_size = image_size
        self.steps = []
        self.original_step = None
        self.initial_obj_positions = None

    def _recording_step(self, action):
        obs, reward, terminated, truncated, info = self.original_step(action)
        
        front_rgb, wrist_rgb = extract_images_from_obs(obs, self.image_size)
        state = get_state_from_obs(obs)
        
        action_arr = np.asarray(action, dtype=np.float32).flatten()
        if len(action_arr) < 8:
            action_arr = np.pad(action_arr, (0, 8 - len(action_arr)), mode="constant")
        elif len(action_arr) > 8:
            action_arr = action_arr[:8]

        self.steps.append({
            "front_rgb": front_rgb,
            "wrist_rgb": wrist_rgb,
            "state": state,
            "action": action_arr,
            "reward": float(reward),
            "info": info,
        })
        
        return obs, reward, terminated, truncated, info

    def __enter__(self):
        self.original_step = self.env.step
        self.env.step = self._recording_step
        return self

    def __exit__(self, *args):
        self.env.step = self.original_step

    def get_steps(self):
        return self.steps


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
    """运行一个 episode，使用 RoboMME 官方 expert planner。
    
    核心流程:
      env.reset()
      -> 记录初始物体状态
      -> 创建 FailAwarePandaArmMotionPlanningSolver
      -> 遍历 env.unwrapped.task_list
      -> task_entry["solve"](env, planner)
      -> env.unwrapped.evaluate() 检查 success
      -> StepRecorder 自动截获 planner 内部 env.step(action)
    """
    from mani_skill.examples.motionplanning.panda.motionplanner import (
        PandaArmMotionPlanningSolver,
    )

    try:
        from robomme.robomme_env.utils.planner_fail_safe import (
            FailAwarePandaArmMotionPlanningSolver,
            ScrewPlanFailure,
        )
    except Exception:
        FailAwarePandaArmMotionPlanningSolver = PandaArmMotionPlanningSolver
        ScrewPlanFailure = RuntimeError

    obs, info = env.reset()
    initial_obj_positions = get_object_positions_from_env(env)

    planner = FailAwarePandaArmMotionPlanningSolver(
        env,
        debug=False,
        vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=False,
        print_env_info=False,
    )

    task_list = getattr(env.unwrapped, "task_list", [])
    if not task_list:
        print("  Warning: No task_list found in environment")
        return [], {"success": False, "num_frames": 0, "object_positions": {}}

    with StepRecorder(env, image_size) as recorder:
        recorder.initial_obj_positions = initial_obj_positions

        for task_idx, task_entry in enumerate(task_list):
            solve_callable = task_entry.get("solve")
            if not callable(solve_callable):
                continue

            try:
                solve_callable(env, planner)
            except ScrewPlanFailure:
                print(f"  Task {task_idx} failed with ScrewPlanFailure")
                break
            except Exception as e:
                print(f"  Task {task_idx} error: {e}")
                break

    steps = recorder.get_steps()
    
    # 检查 success
    success = False
    if steps:
        last_info = steps[-1]["info"]
        status = last_info.get("status", "ongoing")
        success = status == "success"
        
        # 或者使用 env.unwrapped.evaluate()
        try:
            eval_result = env.unwrapped.evaluate(solve_complete_eval=True)
            success = bool(eval_result.get("success", False))
        except Exception:
            pass

    return steps, {
        "success": success,
        "num_frames": len(steps),
        "object_positions": initial_obj_positions,
    }


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


def steps_to_frames(steps, task_description, image_size=256):
    """将 StepRecorder 收集的 steps 转换为 LeRobotDataset frames。"""
    frames = []
    for step in steps:
        front_rgb = step["front_rgb"]
        wrist_rgb = step["wrist_rgb"]
        
        if front_rgb is None or wrist_rgb is None:
            continue
        
        frame = {
            "observation.images.image": front_rgb,
            "observation.images.wrist_image": wrist_rgb,
            "observation.state": step["state"],
            "action": step["action"],
            "next.reward": np.array([step["reward"]], dtype=np.float32),
            "next.success": np.array([step["info"].get("success", False)], dtype=bool),
            "task": task_description,
        }
        frames.append(frame)
    return frames


def phase_random(args, dataset, task, start_ep_idx=0):
    print(f"\n{'='*60}")
    print(f"阶段1: 随机采集 ({args.num_random_episodes} 个 episode)")
    print(f"{'='*60}")

    episode_infos = []
    success_count = 0
    seed = args.seed_start
    consecutive_failures = 0
    task_description = TASK_DESCRIPTIONS.get(task, task)

    while success_count < args.num_random_episodes:
        ep_start = time.time()
        try:
            env = create_raw_env(task, seed=seed, max_steps=args.max_steps)

            steps, ep_info = run_episode_with_planner(
                env, task, args.max_steps, args.image_size, args.extra_frames_after_success
            )
            env.close()

            if ep_info["success"]:
                frames = steps_to_frames(steps, task_description, args.image_size)
                
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
    task_description = TASK_DESCRIPTIONS.get(task, task)

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

                steps, ep_info = run_episode_with_planner(
                    env, task, args.max_steps, args.image_size, args.extra_frames_after_success
                )
                env.close()

                if ep_info["success"]:
                    frames = steps_to_frames(steps, task_description, args.image_size)
                    
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