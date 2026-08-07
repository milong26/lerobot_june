#!/usr/bin/env python
"""
Meta-World demonstration data generator for SIC framework.
Generates LeRobot format datasets with dual-camera (top + wrist) views.

Uses different seeds for each episode to get varied obj/goal positions,
following the original collect_metaworld_dataset.py approach.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import mujoco
import metaworld
import metaworld.policies as policies

os.environ["HF_LEROBOT_HOME"] = str(Path(__file__).parent / "data")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from personal.work2.mw_common.obs_utils import ENV_STATE_LAYOUT

CONFIG_PATH = Path(__file__).parent.parent.parent / "src" / "lerobot" / "envs" / "metaworld_config.json"
with open(CONFIG_PATH) as f:
    METAWORLD_CONFIG = json.load(f)

TASK_DESCRIPTIONS = METAWORLD_CONFIG.get("TASK_DESCRIPTIONS", {})


def get_expert_policy(task_name):
    policy_class_name = f"Sawyer{task_name.replace('-', ' ').title().replace(' ', '')}Policy"
    try:
        policy_class = getattr(policies, policy_class_name)
        return policy_class(), policy_class_name
    except AttributeError:
        print(f"Error: Cannot find expert policy {policy_class_name} for task {task_name}")
        sys.exit(1)


def resize_image(image, target_size):
    from PIL import Image
    img = Image.fromarray(image)
    img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img)


def render_dual_camera(env_top, env_wrist, image_size=224):
    top_image = env_top.render()
    if top_image is not None:
        top_image = np.flip(top_image, (0, 1))
        top_image = resize_image(top_image, image_size)
    wrist_image = env_wrist.render()
    if wrist_image is not None:
        wrist_image = resize_image(wrist_image, image_size)
    return top_image, wrist_image


def sync_env_state(src_env, dst_env):
    dst_env.data.qpos[:] = src_env.data.qpos[:]
    dst_env.data.qvel[:] = src_env.data.qvel[:]
    dst_env.data.ctrl[:] = src_env.data.ctrl[:]
    mujoco.mj_forward(dst_env.model, dst_env.data)


def create_metaworld_env(task_name, seed=None, camera_name="corner2"):
    mt1 = metaworld.MT1(task_name, seed=seed)
    env = mt1.train_classes[task_name](render_mode="rgb_array", camera_name=camera_name)
    task = mt1.train_tasks[0]
    env.set_task(task)
    env._freeze_rand_vec = True
    return env


def run_episode(env_top, env_wrist, expert_policy, task_name, max_steps=500, image_size=224):
    obs, info = env_top.reset()
    env_wrist.reset()
    sync_env_state(env_top, env_wrist)

    obj_pose = env_top.obj_init_pos.copy()
    goal_pose = env_top.goal.copy()

    frames = []
    success_flags = []

    for step in range(max_steps):
        action = expert_policy.get_action(obs)
        obs, reward, terminated, truncated, info = env_top.step(action)
        sync_env_state(env_top, env_wrist)

        top_image, wrist_image = render_dual_camera(env_top, env_wrist, image_size)
        if top_image is None or wrist_image is None:
            continue

        frame = {
            "observation.images.top": top_image,
            "observation.images.wrist": wrist_image,
            "observation.state": obs[:4].copy().astype(np.float32),
            "observation.environment_state": obs.copy().astype(np.float32),
            "action": action.copy().astype(np.float32),
            "next.reward": np.array([reward], dtype=np.float32),
            "next.success": np.array([info.get("success", 0)], dtype=bool),
        }
        frames.append(frame)
        success_flags.append(info.get("success", 0))

        if terminated or truncated:
            break

    return frames, {
        "obj_init_pos": obj_pose,
        "goal_pose": goal_pose,
        "success": any(success_flags),
        "num_frames": len(frames),
    }


def create_dataset(repo_id, output_dir, fps=30, image_size=224):
    features = {
        "observation.images.top": {
            "dtype": "image",
            "shape": (3, image_size, image_size),
            "names": ["channels", "height", "width"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (3, image_size, image_size),
            "names": ["channels", "height", "width"],
        },
        "observation.state": {"dtype": "float32", "shape": (4,)},
        "observation.environment_state": {"dtype": "float32", "shape": (39,), "names": ["keypoints"]},
        "action": {"dtype": "float32", "shape": (4,), "names": {"axes": ["x", "y", "z", "gripper"]}},
        "next.reward": {"dtype": "float32", "shape": (1,)},
        "next.success": {"dtype": "bool", "shape": (1,)},
    }
    return LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=features, root=output_dir,
        robot_type="metaworld", use_videos=True,
    )


def save_episode_metadata(output_dir, episode_infos, task_name, config_map):
    metadata_file = Path(output_dir) / "episode_initial_states.json"
    metadata = {
        "task": task_name,
        "num_episodes": len(episode_infos),
        "config_map": {str(k): list(v) for k, v in config_map.items()},
        "episodes": [],
    }
    for i, info in enumerate(episode_infos):
        metadata["episodes"].append({
            "episode_index": i,
            "obj_init_pos": info["obj_init_pos"].tolist(),
            "goal_pose": info["goal_pose"].tolist(),
            "success": bool(info["success"]),
            "num_frames": info["num_frames"],
        })
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Meta-World Data Generator for SIC Framework")
    parser.add_argument("--task", type=str, default="pick-place-v3")
    parser.add_argument("--num-episodes", type=int, default=72)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--save-config-map", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = "personal/work3/data/b0_dataset"
    if args.repo_id is None:
        args.repo_id = "b0_dataset"

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    expert_policy, policy_name = get_expert_policy(args.task)

    print("=" * 80)
    print("Meta-World Data Generator for SIC Framework")
    print("=" * 80)
    print(f"Task: {args.task}")
    print(f"Target episodes: {args.num_episodes}")
    print(f"Output dir: {args.output_dir}")
    print(f"Repo ID: {args.repo_id}")
    print(f"FPS: {args.fps}")
    print(f"Image size: {args.image_size}")
    print(f"Seed range: {args.seed_start} ~ {args.seed_start + args.num_episodes - 1}")
    print(f"Expert policy: {policy_name}")
    print("=" * 80)

    dataset = create_dataset(args.repo_id, args.output_dir, args.fps, args.image_size)

    episode_infos = []
    config_map = {}
    success_count = 0
    fail_count = 0
    start_time = time.time()

    print(f"\nGenerating {args.num_episodes} episodes...")
    print("-" * 80)

    for ep_idx in range(args.num_episodes):
        ep_start = time.time()
        seed = args.seed_start + ep_idx

        pos_id = ep_idx // 8
        rot_id = ep_idx % 8
        config_map[ep_idx] = (pos_id, rot_id)

        try:
            env_top = create_metaworld_env(args.task, seed=seed, camera_name="corner2")
            env_wrist = create_metaworld_env(args.task, seed=seed, camera_name="behindGripper")
        except Exception as e:
            print(f"  Episode {ep_idx}: Failed to create env ({e})")
            fail_count += 1
            continue

        frames, ep_info = run_episode(env_top, env_wrist, expert_policy, args.task, args.max_steps, args.image_size)
        env_top.close()
        env_wrist.close()

        if ep_info["num_frames"] == 0:
            print(f"  Episode {ep_idx:3d}/{args.num_episodes} | No frames rendered | Seed: {seed}")
            fail_count += 1
            continue

        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()

        episode_infos.append({**ep_info, "included": True})
        if ep_info["success"]:
            success_count += 1

        print(
            f"  Episode {ep_idx:3d}/{args.num_episodes} | "
            f"Frames: {ep_info['num_frames']:4d} | "
            f"Success: {'Y' if ep_info['success'] else 'N'} | "
            f"Pos={pos_id}, Rot={rot_id} | "
            f"Seed: {seed} | "
            f"Time: {time.time() - ep_start:.2f}s"
        )

    dataset.finalize()

    save_episode_metadata(args.output_dir, episode_infos, args.task, config_map)

    if args.save_config_map:
        os.makedirs(os.path.dirname(args.save_config_map), exist_ok=True)
        with open(args.save_config_map, 'w') as f:
            json.dump({str(k): list(v) for k, v in config_map.items()}, f, indent=2)
        print(f"\nConfig map saved to {args.save_config_map}")

    total_time = time.time() - start_time
    included = sum(1 for e in episode_infos if e.get("included", True))
    print("\n" + "=" * 80)
    print("Generation Complete!")
    print("=" * 80)
    print(f"Target episodes: {args.num_episodes}")
    print(f"Included episodes: {included}")
    print(f"Expert success: {success_count}")
    print(f"Failed/skipped: {fail_count}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Dataset path: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()