#!/usr/bin/env python
"""
RoboMME expert trajectory collector -> LeRobotDataset.

Execution:
  seed -> raw RoboMME env -> official task_list solve() + FailAware planner
  -> record real planner actions/raw camera observations
  -> evaluate success -> commit ONLY successful trajectories to LeRobotDataset.

The random phase is intentionally independent from BenchmarkEnvBuilder/test splits.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
_nvidia_icd = "/usr/share/vulkan/icd.d/nvidia_icd.json"
if os.path.exists(_nvidia_icd):
    os.environ["VK_ICD_FILENAMES"] = _nvidia_icd
os.environ["SAPIEN_VULKAN_DEVICE_INDEX"] = os.environ.get("SAPIEN_VULKAN_DEVICE_INDEX", "0")

import gymnasium as gym
import numpy as np
import torch

import robomme.robomme_env  # noqa: F401 - registers RoboMME gym envs
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from robomme_task_config import (
    TASK_CONFIG_ADAPTERS,
    apply_task_configuration,
    extract_task_configuration,
    generate_uniform_configurations,
    get_uniform_spec,
)

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

MAX_PERTURB_RETRIES = 4
EXTRA_FRAMES_AFTER_SUCCESS = 10
MAX_CONSECUTIVE_FAILURES = 50
SCREW_MAX_ATTEMPTS = 3
RRT_MAX_ATTEMPTS = 3


def _to_bool(value) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().any().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def create_raw_env(task, seed, difficulty=None):
    """Create the same raw env style used by RoboMME's official dataset generator."""
    kwargs = dict(
        obs_mode="rgb+depth+segmentation",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        reward_mode="dense",
        seed=int(seed),
    )
    if difficulty is not None:
        kwargs["difficulty"] = difficulty
    return gym.make(task, **kwargs)


def _resize_rgb(rgb, image_size=256):
    from PIL import Image

    arr = _to_numpy(rgb)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.dtype != np.uint8:
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.floating) and arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.shape[:2] != (image_size, image_size):
        arr = np.asarray(Image.fromarray(arr).resize((image_size, image_size), Image.BILINEAR))
    return arr


def extract_images_from_raw_obs(obs, image_size=256):
    """Read raw ManiSkill camera tensors used by RoboMME's official RecordWrapper."""
    try:
        sensor_data = obs["sensor_data"]
        front = sensor_data["base_camera"]["rgb"]
        wrist = sensor_data["hand_camera"]["rgb"]
        return _resize_rgb(front, image_size), _resize_rgb(wrist, image_size)
    except Exception:
        # Fallback for Benchmark/DemonstrationWrapper observations.
        front = obs.get("front_rgb_list") if isinstance(obs, dict) else None
        wrist = obs.get("wrist_rgb_list") if isinstance(obs, dict) else None
        if front is None or wrist is None:
            return None, None
        if isinstance(front, list):
            front = front[-1]
        if isinstance(wrist, list):
            wrist = wrist[-1]
        return _resize_rgb(front, image_size), _resize_rgb(wrist, image_size)


def extract_robot_state(env):
    """Match LeRobot RoboMME eval wrapper: 7 arm joints + one gripper scalar."""
    qpos = _to_numpy(env.unwrapped.agent.robot.qpos).reshape(-1).astype(np.float32)
    arm = qpos[:7]
    gripper = qpos[7:8] if qpos.size >= 8 else np.zeros(1, dtype=np.float32)
    state = np.concatenate([arm, gripper]).astype(np.float32)
    if state.size < 8:
        state = np.pad(state, (0, 8 - state.size)).astype(np.float32)
    return state[:8]


def normalize_joint_action(action):
    arr = _to_numpy(action).reshape(-1).astype(np.float32)
    if arr.size < 8:
        arr = np.pad(arr, (0, 8 - arr.size), constant_values=-1.0)
    return arr[:8].astype(np.float32)


class ExpertTrajectoryRecorder(gym.Wrapper):
    """Record the actual env.step(action) calls made by RoboMME's expert planner."""

    def __init__(self, env, image_size=256):
        super().__init__(env)
        self.image_size = image_size
        self.steps = []
        self.last_action = None

    def reset(self, **kwargs):
        self.steps = []
        self.last_action = None
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        front_rgb, wrist_rgb = extract_images_from_raw_obs(obs, self.image_size)
        if front_rgb is not None and wrist_rgb is not None:
            action_arr = normalize_joint_action(action)
            self.steps.append(
                {
                    "front_rgb": front_rgb,
                    "wrist_rgb": wrist_rgb,
                    "state": extract_robot_state(self),
                    "action": action_arr,
                    "reward": float(_to_numpy(reward).reshape(-1)[0]),
                    "success": _to_bool(info.get("success", False)),
                    "info": dict(info),
                }
            )
            self.last_action = action_arr.copy()
        return obs, reward, terminated, truncated, info


def _patch_planner_screw_to_rrt(planner):
    """Use RoboMME official dataset-generation fallback: 3x screw -> 3x RRT."""
    original_screw = planner.move_to_pose_with_screw
    original_rrt = planner.move_to_pose_with_RRTStar

    def _move_screw_then_rrt(*args, **kwargs):
        from robomme.robomme_env.utils.planner_fail_safe import ScrewPlanFailure

        for _ in range(SCREW_MAX_ATTEMPTS):
            try:
                result = original_screw(*args, **kwargs)
            except ScrewPlanFailure:
                continue
            if isinstance(result, int) and result == -1:
                continue
            return result

        for _ in range(RRT_MAX_ATTEMPTS):
            try:
                result = original_rrt(*args, **kwargs)
            except Exception:
                continue
            if isinstance(result, int) and result == -1:
                continue
            return result
        return -1

    planner.move_to_pose_with_screw = _move_screw_then_rrt


def _make_planner(env, task):
    from robomme.robomme_env.utils.planner_fail_safe import (
        FailAwarePandaArmMotionPlanningSolver,
        FailAwarePandaStickMotionPlanningSolver,
    )

    # Monkey-patch PandaStickMotionPlanningSolver.__init__ to fix upstream bug:
    # it passes visualize_target_grasp_pose to BaseMotionPlanningSolver which doesn't accept it
    from mani_skill.examples.motionplanning.panda.motionplanner_stick import (
        PandaStickMotionPlanningSolver,
    )
    if not hasattr(PandaStickMotionPlanningSolver, "_orig_init_patched"):
        _orig_init = PandaStickMotionPlanningSolver.__init__

        def _fixed_init(
            self,
            env,
            debug=False,
            vis=True,
            base_pose=None,
            visualize_target_grasp_pose=True,
            print_env_info=True,
            joint_vel_limits=0.9,
            joint_acc_limits=0.9,
        ):
            # Call parent __init__ without visualize_target_grasp_pose
            from mani_skill.examples.motionplanning.base_motionplanner.motionplanner import (
                BaseMotionPlanningSolver,
            )

            BaseMotionPlanningSolver.__init__(
                self,
                env,
                debug=debug,
                vis=vis,
                base_pose=base_pose,
                print_env_info=print_env_info,
                joint_vel_limits=joint_vel_limits,
                joint_acc_limits=joint_acc_limits,
            )

        PandaStickMotionPlanningSolver.__init__ = _fixed_init
        PandaStickMotionPlanningSolver._orig_init_patched = True

    common = dict(
        debug=False,
        vis=False,
        base_pose=env.unwrapped.agent.robot.pose,
        print_env_info=False,
    )
    if task in ("PatternLock", "RouteStick"):
        planner = FailAwarePandaStickMotionPlanningSolver(env, joint_vel_limits=0.3, **common)
    else:
        planner = FailAwarePandaArmMotionPlanningSolver(env, **common)
    _patch_planner_screw_to_rrt(planner)
    return planner


def _mark_planner_failure(env):
    u = env.unwrapped
    try:
        u.failureflag = torch.tensor([True])
        u.successflag = torch.tensor([False])
        u.current_task_failure = True
    except Exception:
        pass


def _evaluate(env):
    try:
        result = env.unwrapped.evaluate(solve_complete_eval=True)
    except Exception:
        return False, False, {}
    return _to_bool(result.get("success", False)), _to_bool(result.get("fail", False)), result


def run_episode_with_planner(
    raw_env,
    task,
    image_size=256,
    extra_frames=10,
    uniform_config=None,
):
    """
    Run RoboMME's own expert solver and return the real successful trajectory.

    If uniform_config is provided, it is injected AFTER reset and BEFORE planner
    creation via apply_task_configuration().
    """
    from robomme.env_record_wrapper import FailsafeTimeout
    from robomme.robomme_env.utils.SceneGenerationError import SceneGenerationError
    from robomme.robomme_env.utils.planner_fail_safe import ScrewPlanFailure

    env = ExpertTrajectoryRecorder(raw_env, image_size=image_size)
    try:
        env.reset()

        # Extract initial configuration AFTER reset, BEFORE any planner action
        initial_configuration = extract_task_configuration(env, task)

        # If uniform config provided, inject it now
        if uniform_config is not None:
            applied = apply_task_configuration(env, task, uniform_config)
            # Re-read configuration after injection to capture actual injected state
            if applied:
                initial_configuration = extract_task_configuration(env, task)

        difficulty = getattr(env.unwrapped, "difficulty", None)
        planner = _make_planner(env, task)

        tasks = list(getattr(env.unwrapped, "task_list", []) or [])
        if not tasks:
            return [], {
                "success": False,
                "num_frames": 0,
                "initial_configuration": initial_configuration,
                "difficulty": difficulty,
                "reason": "no_task_list",
            }

        episode_success = False
        failure_reason = None

        for task_idx, task_entry in enumerate(tasks):
            solve_callable = task_entry.get("solve")
            if not callable(solve_callable):
                continue

            _evaluate(env)
            screw_failed = False
            try:
                solve_result = solve_callable(env, planner)
                if isinstance(solve_result, int) and solve_result == -1:
                    screw_failed = True
                    _mark_planner_failure(env)
            except ScrewPlanFailure:
                screw_failed = True
                _mark_planner_failure(env)
            except FailsafeTimeout:
                failure_reason = f"failsafe_timeout_task_{task_idx}"
                break

            success, failed, _ = _evaluate(env)
            if success:
                episode_success = True
                break
            if screw_failed or failed:
                failure_reason = (
                    f"screw_and_rrt_failed_task_{task_idx}" if screw_failed else f"task_failed_{task_idx}"
                )
                break
        else:
            episode_success, _, _ = _evaluate(env)

        final_success, _, _ = _evaluate(env)
        episode_success = episode_success or final_success

        if episode_success and extra_frames > 0 and env.last_action is not None:
            tail_recorded = 0
            for _ in range(extra_frames):
                try:
                    env.step(env.last_action.copy())
                    tail_recorded += 1
                except Exception:
                    break
        else:
            tail_recorded = 0

        return list(env.steps), {
            "success": bool(episode_success),
            "num_frames": len(env.steps),
            "initial_configuration": initial_configuration,
            "difficulty": str(difficulty) if difficulty is not None else None,
            "tail_frames": tail_recorded,
            "reason": failure_reason,
        }

    except SceneGenerationError:
        return [], {
            "success": False,
            "num_frames": 0,
            "initial_configuration": {},
            "difficulty": None,
            "reason": "scene_generation_error",
        }


def steps_to_frames(steps, task_description):
    frames = []
    for step in steps:
        if step["front_rgb"] is None or step["wrist_rgb"] is None:
            continue
        frames.append(
            {
                "observation.images.image": step["front_rgb"],
                "observation.images.wrist_image": step["wrist_rgb"],
                "observation.state": step["state"],
                "action": step["action"],
                "next.reward": np.array([step["reward"]], dtype=np.float32),
                "next.success": np.array([step["success"]], dtype=bool),
                "task": task_description,
            }
        )
    return frames


def commit_successful_episode(dataset, frames):
    """Transactional LeRobot commit: failed/empty trajectories never become episodes."""
    if not frames:
        return False
    if dataset.has_pending_frames():
        dataset.clear_episode_buffer(delete_images=True)
    try:
        for frame in frames:
            dataset.add_frame(frame)
        if not dataset.has_pending_frames():
            return False
        dataset.save_episode()
        return True
    except Exception:
        if dataset.has_pending_frames():
            dataset.clear_episode_buffer(delete_images=True)
        raise


def create_dataset(
    repo_id,
    output_dir,
    fps=10,
    image_size=256,
    streaming_encoding=False,
    encoder_threads=None,
    image_writer_processes=0,
    image_writer_threads=8,
    batch_encoding_size=1,
    vcodec=None,
):
    from lerobot.configs.video import RGBEncoderConfig, rgb_encoder_defaults

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


def phase_random(args, dataset, task, start_ep_idx=0):
    print(f"\n{'=' * 60}")
    print(f"Phase 1: Random collection ({args.num_random_episodes} episodes)")
    print("=" * 60)

    infos = []
    success_count = 0
    seed = args.seed_start
    consecutive_failures = 0
    task_description = TASK_DESCRIPTIONS.get(task, task)
    env_recreate_interval = 50
    episodes_since_recreate = 0

    # Create env and recreate every 50 episodes to avoid Vulkan resource exhaustion
    raw_env = create_raw_env(task, seed, difficulty=args.difficulty)

    while success_count < args.num_random_episodes:
        started = time.time()
        try:
            raw_env.unwrapped.seed = int(seed)
            steps, ep_info = run_episode_with_planner(
                raw_env,
                task,
                image_size=args.image_size,
                extra_frames=args.extra_frames_after_success,
            )

            if ep_info["success"]:
                frames = steps_to_frames(steps, task_description)
                if not frames:
                    raise RuntimeError(
                        "Expert reported success but recorder produced 0 valid RGB frames. "
                        "Expected raw obs sensor_data.base_camera/hand_camera."
                    )
                commit_successful_episode(dataset, frames)

                ep_info["seed"] = seed
                ep_info["episode_idx"] = start_ep_idx + success_count
                ep_info["_sampling_method"] = "random"
                infos.append(ep_info)
                success_count += 1
                consecutive_failures = 0
                print(
                    f"  [R] Episode {start_ep_idx + success_count:4d} | "
                    f"Frames: {len(frames):4d} | Success (seed={seed}) | "
                    f"{time.time() - started:.1f}s"
                )
            else:
                consecutive_failures += 1
                print(
                    f"  [R] FAILED (seed={seed}) | Frames: {ep_info['num_frames']:4d} | "
                    f"reason={ep_info.get('reason')} | {time.time() - started:.1f}s"
                )

        except Exception as exc:
            consecutive_failures += 1
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer(delete_images=True)
            print(f"  [R] ERROR (seed={seed}): {type(exc).__name__}: {exc}")

        episodes_since_recreate += 1
        if episodes_since_recreate >= env_recreate_interval:
            if raw_env is not None:
                try:
                    raw_env.close()
                except Exception:
                    pass
            raw_env = create_raw_env(task, seed + 1, difficulty=args.difficulty)
            episodes_since_recreate = 0

        seed += 1
        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            print(f"Warning: {MAX_CONSECUTIVE_FAILURES} consecutive failures, continuing.")
            consecutive_failures = 0

    if raw_env is not None:
        try:
            raw_env.close()
        except Exception:
            pass

    print(f"Phase 1 complete: {success_count}/{args.num_random_episodes} successful")
    return infos


def phase_uniform(args, dataset, task, start_ep_idx=0):
    print(f"\n{'=' * 60}")
    print(f"Phase 2: Uniform sampling ({args.num_uniform_episodes} episodes)")
    print("=" * 60)

    if args.num_uniform_episodes <= 0:
        return []

    task_description = TASK_DESCRIPTIONS.get(task, task)
    rng = np.random.RandomState(456)
    fallback_seed = max(args.seed_start + args.num_random_episodes, 10000)

    # Get uniform spec for this task
    raw_env = create_raw_env(task, args.seed_start, difficulty=args.difficulty)
    try:
        raw_env.reset()
        uniform_spec = get_uniform_spec(raw_env, task)
    finally:
        raw_env.close()

    if uniform_spec is None:
        print(f"  No uniform spec for task={task}, falling back to random collection.")
        return _phase_fallback_random(args, dataset, task, start_ep_idx, fallback_seed, task_description)

    print(f"  Uniform spec: {uniform_spec['num_objects']} objects, "
          f"grid {uniform_spec['grid_resolution']['x']}x{uniform_spec['grid_resolution']['y']}")

    # Generate uniform configurations
    uniform_configs = generate_uniform_configurations(
        uniform_spec, args.num_uniform_episodes, rng
    )
    print(f"  Generated {len(uniform_configs)} uniform configurations")

    infos = []
    success_count = 0
    config_idx = 0
    env_recreate_interval = 50
    episodes_since_recreate = 0

    # Create env and recreate every 50 episodes to avoid Vulkan resource exhaustion
    raw_env = create_raw_env(task, int(rng.randint(0, 1_000_000)), difficulty=args.difficulty)

    while success_count < args.num_uniform_episodes:
        uniform_cfg = uniform_configs[config_idx] if config_idx < len(uniform_configs) else None
        config_idx += 1

        success_this_config = False
        for retry in range(MAX_PERTURB_RETRIES + 1):
            seed = int(rng.randint(0, 1_000_000))
            candidate = None
            method = "fallback_random"
            if uniform_cfg is not None:
                candidate = uniform_cfg
                method = "uniform_grid" if retry == 0 else "uniform_perturbed"

            try:
                raw_env.unwrapped.seed = int(seed)
                steps, ep_info = run_episode_with_planner(
                    raw_env,
                    task,
                    image_size=args.image_size,
                    extra_frames=args.extra_frames_after_success,
                    uniform_config=candidate,
                )
                if not ep_info["success"]:
                    continue

                frames = steps_to_frames(steps, task_description)
                if not frames:
                    continue
                commit_successful_episode(dataset, frames)
                ep_info["seed"] = seed
                ep_info["episode_idx"] = start_ep_idx + success_count
                ep_info["_sampling_method"] = method
                ep_info["uniform_config"] = candidate
                infos.append(ep_info)
                success_count += 1
                success_this_config = True
                print(
                    f"  [U] Episode {start_ep_idx + success_count:4d} | "
                    f"Frames: {len(frames):4d} | {method} | seed={seed}"
                )
                break
            except Exception as exc:
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer(delete_images=True)
                print(f"  [U] ERROR (seed={seed}): {type(exc).__name__}: {exc}")

        if success_this_config:
            episodes_since_recreate += 1
            if episodes_since_recreate >= env_recreate_interval:
                if raw_env is not None:
                    try:
                        raw_env.close()
                    except Exception:
                        pass
                raw_env = create_raw_env(task, int(rng.randint(0, 1_000_000)), difficulty=args.difficulty)
                episodes_since_recreate = 0
            continue

        # If all configs exhausted or failed, fall back to random
        while success_count < args.num_uniform_episodes:
            try:
                raw_env.unwrapped.seed = int(fallback_seed)
                steps, ep_info = run_episode_with_planner(
                    raw_env,
                    task,
                    image_size=args.image_size,
                    extra_frames=args.extra_frames_after_success,
                )
                if ep_info["success"]:
                    frames = steps_to_frames(steps, task_description)
                    if frames:
                        commit_successful_episode(dataset, frames)
                        ep_info["seed"] = fallback_seed
                        ep_info["episode_idx"] = start_ep_idx + success_count
                        ep_info["_sampling_method"] = "fallback_random"
                        infos.append(ep_info)
                        success_count += 1
                        print(
                            f"  [U] Episode {start_ep_idx + success_count:4d} | "
                            f"Frames: {len(frames):4d} | fallback_random | seed={fallback_seed}"
                        )
            except Exception as exc:
                if dataset.has_pending_frames():
                    dataset.clear_episode_buffer(delete_images=True)
                print(f"  [U] FALLBACK ERROR (seed={fallback_seed}): {type(exc).__name__}: {exc}")

            episodes_since_recreate += 1
            if episodes_since_recreate >= env_recreate_interval:
                if raw_env is not None:
                    try:
                        raw_env.close()
                    except Exception:
                        pass
                raw_env = create_raw_env(task, int(rng.randint(0, 1_000_000)), difficulty=args.difficulty)
                episodes_since_recreate = 0

            fallback_seed += 1
            break

    if raw_env is not None:
        try:
            raw_env.close()
        except Exception:
            pass

    print(f"Phase 2 complete: {success_count}/{args.num_uniform_episodes} successful")
    return infos


def _phase_fallback_random(args, dataset, task, start_ep_idx, fallback_seed, task_description):
    """Fallback random collection when uniform spec is not available."""
    infos = []
    success_count = 0
    env_recreate_interval = 50
    episodes_since_recreate = 0

    raw_env = create_raw_env(task, fallback_seed, difficulty=args.difficulty)

    while success_count < args.num_uniform_episodes:
        try:
            raw_env.unwrapped.seed = int(fallback_seed)
            steps, ep_info = run_episode_with_planner(
                raw_env,
                task,
                image_size=args.image_size,
                extra_frames=args.extra_frames_after_success,
            )
            if ep_info["success"]:
                frames = steps_to_frames(steps, task_description)
                if frames:
                    commit_successful_episode(dataset, frames)
                    ep_info["seed"] = fallback_seed
                    ep_info["episode_idx"] = start_ep_idx + success_count
                    ep_info["_sampling_method"] = "fallback_random"
                    infos.append(ep_info)
                    success_count += 1
                    print(
                        f"  [U] Episode {start_ep_idx + success_count:4d} | "
                        f"Frames: {len(frames):4d} | fallback_random | seed={fallback_seed}"
                    )
        except Exception as exc:
            if dataset.has_pending_frames():
                dataset.clear_episode_buffer(delete_images=True)
            print(f"  [U] FALLBACK ERROR (seed={fallback_seed}): {type(exc).__name__}: {exc}")

        episodes_since_recreate += 1
        if episodes_since_recreate >= env_recreate_interval:
            if raw_env is not None:
                try:
                    raw_env.close()
                except Exception:
                    pass
            raw_env = create_raw_env(task, fallback_seed + 1, difficulty=args.difficulty)
            episodes_since_recreate = 0

        fallback_seed += 1

    if raw_env is not None:
        try:
            raw_env.close()
        except Exception:
            pass

    print(f"Phase 2 complete (fallback): {success_count}/{args.num_uniform_episodes} successful")
    return infos


def save_episode_metadata(output_dir, all_episode_infos, task_name):
    path = Path(output_dir) / "episode_initial_states.json"
    metadata = {"task": task_name, "num_episodes": len(all_episode_infos), "episodes": []}

    for i, info in enumerate(all_episode_infos):
        ep = {
            "episode_index": i,
            "success": bool(info.get("success", False)),
            "num_frames": int(info.get("num_frames", 0)),
            "seed": info.get("seed"),
            "difficulty": info.get("difficulty"),
            "sampling_method": info.get("_sampling_method"),
            "tail_frames": int(info.get("tail_frames", 0)),
            "initial_configuration": info.get("initial_configuration", {}),
        }
        if info.get("uniform_config") is not None:
            ep["uniform_config"] = info["uniform_config"]
        metadata["episodes"].append(ep)

    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Episode initial states saved to: {path}")


def inspect_config(args):
    """Inspect mode: extract and print task configurations without collecting data."""
    task = args.task
    num_seeds = args.inspect_seeds
    print(f"\n{'=' * 60}")
    print(f"Inspect mode: task={task}, seeds=0..{num_seeds - 1}")
    print("=" * 60)

    if task not in TASK_CONFIG_ADAPTERS:
        print(f"ERROR: No adapter for task={task}")
        sys.exit(1)

    for seed in range(num_seeds):
        print(f"\n--- Seed {seed} ---")
        raw_env = None
        try:
            raw_env = create_raw_env(task, seed, difficulty=args.difficulty)
            raw_env.reset()
            config = extract_task_configuration(raw_env, task)

            print(f"  Difficulty: {config['task_config'].get('difficulty')}")
            print(f"  Movable objects ({len(config['movable_objects'])}):")
            for obj in config["movable_objects"]:
                print(f"    - {obj['name']}: pos={obj['pose']['position'] if obj['pose'] else 'N/A'}")
            print(f"  Randomized targets ({len(config['randomized_targets'])}):")
            for tgt in config["randomized_targets"]:
                print(f"    - {tgt['name']}: pos={tgt['pose']['position'] if tgt['pose'] else 'N/A'}")
            print(f"  Articulations ({len(config['articulations'])}):")
            for art in config["articulations"]:
                print(f"    - {art.get('name', 'unknown')}")
            print(f"  Task config: {json.dumps(config['task_config'], indent=4, default=str)}")

            # Check uniform spec
            uniform_spec = get_uniform_spec(raw_env, task)
            if uniform_spec:
                print(f"  Uniform spec: {uniform_spec['num_objects']} objects, "
                      f"grid {uniform_spec['grid_resolution']['x']}x{uniform_spec['grid_resolution']['y']}")
            else:
                print("  Uniform spec: None (task does not support uniform spatial sampling)")

        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}")
        finally:
            if raw_env is not None:
                try:
                    raw_env.close()
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description="RoboMME expert -> LeRobotDataset collector")
    parser.add_argument("--task", type=str, default="PickXtimes")
    parser.add_argument("--num-random-episodes", type=int, default=300)
    parser.add_argument("--num-uniform-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--extra-frames-after-success", type=int, default=EXTRA_FRAMES_AFTER_SUCCESS)
    parser.add_argument("--streaming-encoding", action="store_true", default=False)
    parser.add_argument("--encoder-threads", type=int, default=None)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--batch-encoding-size", type=int, default=1)
    parser.add_argument("--vcodec", type=str, default=None)
    parser.add_argument("--inspect-config-only", action="store_true", default=False,
                        help="Only inspect task configuration, do not collect data")
    parser.add_argument("--inspect-seeds", type=int, default=3,
                        help="Number of seeds to inspect in inspect mode")
    parser.add_argument("--difficulty", type=str, default=None, choices=["easy", "medium", "hard"],
                        help="Fix difficulty level for all episodes. If not set, difficulty is derived from seed %% 3.")
    args = parser.parse_args()

    # Inspect mode
    if args.inspect_config_only:
        inspect_config(args)
        return

    if args.output_dir is None:
        args.output_dir = f"personal/work2/dataset_view_robomme/{args.task}"
    if args.repo_id is None:
        args.repo_id = f"work2/robomme_{args.task}"

    output_dir = Path(args.output_dir)
    is_resume = output_dir.exists() and (output_dir / "meta" / "info.json").exists()

    print("=" * 80)
    print("RoboMME Expert Dataset Collector")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Difficulty: {args.difficulty or 'auto (seed % 3)'}")
    print(f"Random: {args.num_random_episodes}, Uniform: {args.num_uniform_episodes}")
    print(f"Output: {args.output_dir}")
    print(f"Seed start: {args.seed_start}")
    print(f"Mode: {'Resume' if is_resume else 'Fresh'}")
    print("=" * 80)

    if args.num_random_episodes == 0 and args.num_uniform_episodes == 0:
        print("Nothing to collect.")
        sys.exit(0)

    if is_resume:
        dataset = LeRobotDataset.resume(repo_id=args.repo_id, root=args.output_dir)
        existing_episodes = dataset.num_episodes
    else:
        if output_dir.exists():
            shutil.rmtree(output_dir)
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

    all_infos = []
    started = time.time()

    if args.num_random_episodes > 0:
        random_infos = phase_random(args, dataset, args.task, start_ep_idx=existing_episodes)
        all_infos.extend(random_infos)
        existing_episodes += len(random_infos)

    if args.num_uniform_episodes > 0:
        uniform_infos = phase_uniform(args, dataset, args.task, start_ep_idx=existing_episodes)
        all_infos.extend(uniform_infos)

    dataset.finalize()
    save_episode_metadata(args.output_dir, all_infos, args.task)

    print("=" * 80)
    print(f"Collection complete: {len(all_infos)} successful episodes")
    print(f"Elapsed: {time.time() - started:.1f}s")
    print(f"Dataset: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()