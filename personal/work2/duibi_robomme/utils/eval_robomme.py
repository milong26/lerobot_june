#!/usr/bin/env python
"""
Robomme evaluation script for SmolVLA policy.

Evaluates a trained SmolVLA checkpoint on a single Robomme task by running
rollouts in the Robomme simulation environment and computing success rate.

Usage:
    python eval_robomme.py \
        --checkpoint-path /path/to/checkpoint \
        --task MoveCube_easy \
        --n-episodes 20 \
        --eval-seeds 0,1,2,3,4 \
        --output-dir /path/to/eval_output \
        --gpu-id 0
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
_nvidia_icd = "/usr/share/vulkan/icd.d/nvidia_icd.json"
if os.path.exists(_nvidia_icd):
    os.environ["VK_ICD_FILENAMES"] = _nvidia_icd
os.environ["SAPIEN_VULKAN_DEVICE_INDEX"] = os.environ.get("SAPIEN_VULKAN_DEVICE_INDEX", "0")
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import gymnasium as gym

import robomme.robomme_env  # noqa: F401 - registers RoboMME gym envs

TASK_NAME_MAP = {
    "MoveCube_easy": "MoveCube",
    "PatternLock_medium": "PatternLock",
    "RouteStick_hard": "RouteStick",
}

DIFFICULTY_MAP = {
    "MoveCube_easy": "easy",
    "PatternLock_medium": "medium",
    "RouteStick_hard": "hard",
}


def create_raw_env(task, seed, difficulty=None):
    """Create a raw Robomme environment."""
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


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


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
    """Extract front and wrist camera images from raw observation."""
    try:
        sensor_data = obs["sensor_data"]
        front = sensor_data["base_camera"]["rgb"]
        wrist = sensor_data["hand_camera"]["rgb"]
        return _resize_rgb(front, image_size), _resize_rgb(wrist, image_size)
    except Exception:
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
    """Extract robot state: 7 arm joints + 1 gripper scalar."""
    qpos = _to_numpy(env.unwrapped.agent.robot.qpos).reshape(-1).astype(np.float32)
    arm = qpos[:7]
    gripper = qpos[7:8] if qpos.size >= 8 else np.zeros(1, dtype=np.float32)
    state = np.concatenate([arm, gripper]).astype(np.float32)
    if state.size < 8:
        state = np.pad(state, (0, 8 - state.size)).astype(np.float32)
    return state[:8]


def _to_bool(value):
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().bool().any().item())
    if isinstance(value, np.ndarray):
        return bool(np.any(value))
    return bool(value)


def load_policy(checkpoint_path, device):
    """Load SmolVLA policy from checkpoint."""
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies import make_policy

    print(f"Loading policy from: {checkpoint_path}")

    policy = make_policy(
        policy_path=checkpoint_path,
        device=device,
    )
    policy.eval()
    print(f"Policy loaded on device: {device}")
    return policy


def run_episode(policy, env, max_steps=300, image_size=256):
    """Run a single evaluation episode."""
    obs, info = env.reset()
    front_rgb, wrist_rgb = extract_images_from_raw_obs(obs, image_size)

    episode_steps = 0
    episode_success = False

    for step in range(max_steps):
        state = extract_robot_state(env)

        obs_dict = {
            "observation.images.camera1": torch.from_numpy(front_rgb.transpose(2, 0, 1)).unsqueeze(0).to(policy.device),
            "observation.images.camera2": torch.from_numpy(wrist_rgb.transpose(2, 0, 1)).unsqueeze(0).to(policy.device),
            "observation.state": torch.from_numpy(state).unsqueeze(0).to(policy.device),
        }

        with torch.no_grad():
            action = policy.select_action(obs_dict)

        action_np = action.detach().cpu().numpy().squeeze(0).astype(np.float32)

        obs, reward, terminated, truncated, info = env.step(action_np)
        episode_steps += 1

        success = _to_bool(info.get("success", False)) or _to_bool(info.get("is_success", False))
        status = info.get("status", "ongoing")
        if status == "success":
            success = True

        if success:
            episode_success = True
            break

        if terminated or truncated:
            break

        front_rgb, wrist_rgb = extract_images_from_raw_obs(obs, image_size)
        if front_rgb is None or wrist_rgb is None:
            break

    return episode_success, episode_steps


def evaluate_task(policy, task_name, eval_seeds, n_episodes_per_seed, gpu_id, output_dir, max_steps=300):
    """Evaluate a policy on a single Robomme task."""
    robomme_task = TASK_NAME_MAP.get(task_name, task_name)
    difficulty = DIFFICULTY_MAP.get(task_name, None)

    print(f"\n{'='*60}")
    print(f"Evaluating task: {task_name} (Robomme task: {robomme_task}, difficulty: {difficulty})")
    print(f"Eval seeds: {eval_seeds}")
    print(f"Episodes per seed: {n_episodes_per_seed}")
    print(f"Max steps: {max_steps}")
    print(f"{'='*60}")

    episode_results = []
    total_episodes = 0
    total_success = 0

    for seed in eval_seeds:
        env = create_raw_env(robomme_task, seed, difficulty=difficulty)

        for ep_idx in range(n_episodes_per_seed):
            ep_seed = seed * 1000 + ep_idx
            env.unwrapped.seed = int(ep_seed)

            try:
                success, steps = run_episode(policy, env, max_steps=max_steps)
            except Exception as e:
                print(f"  ERROR episode {ep_idx} (seed={ep_seed}): {type(e).__name__}: {e}")
                success = False
                steps = 0

            total_episodes += 1
            if success:
                total_success += 1

            result = {
                "episode_index": ep_idx,
                "seed": ep_seed,
                "eval_seed": seed,
                "success": success,
                "steps": steps,
            }
            episode_results.append(result)

            status_str = "SUCCESS" if success else "FAIL"
            print(f"  [{status_str}] Episode {ep_idx} (seed={ep_seed}): {steps} steps")

        try:
            env.close()
        except Exception:
            pass

    success_rate = total_success / total_episodes if total_episodes > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Task {task_name} complete")
    print(f"  Episodes: {total_episodes}")
    print(f"  Success: {total_success}")
    print(f"  Success rate: {success_rate:.4f}")
    print(f"{'='*60}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_result = {
        "task": task_name,
        "robomme_task": robomme_task,
        "difficulty": difficulty,
        "eval_seeds": eval_seeds,
        "n_episodes_per_seed": n_episodes_per_seed,
        "num_episodes": total_episodes,
        "num_success": total_success,
        "success_rate": success_rate,
        "episode_results": episode_results,
    }

    result_file = output_dir / "eval_result.json"
    with open(result_file, "w") as f:
        json.dump(eval_result, f, indent=2)
    print(f"  Result saved to: {result_file}")

    return eval_result


def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA on Robomme task")
    parser.add_argument("--checkpoint-path", type=str, required=True,
                       help="Path to SmolVLA checkpoint directory")
    parser.add_argument("--task", type=str, required=True,
                       choices=["MoveCube_easy", "PatternLock_medium", "RouteStick_hard"])
    parser.add_argument("--n-episodes", type=int, default=20,
                       help="Number of evaluation episodes per seed")
    parser.add_argument("--eval-seeds", type=str, default="0,1,2,3,4",
                       help="Comma-separated evaluation seeds")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=300)
    args = parser.parse_args()

    eval_seeds = [int(s) for s in args.eval_seeds.split(",")]
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"

    policy = load_policy(args.checkpoint_path, device)

    evaluate_task(
        policy=policy,
        task_name=args.task,
        eval_seeds=eval_seeds,
        n_episodes_per_seed=args.n_episodes,
        gpu_id=args.gpu_id,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()