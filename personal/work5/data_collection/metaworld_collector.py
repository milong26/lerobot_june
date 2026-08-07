"""MetaWorld demo collector using scripted policy with state injection."""

import os
import sys
import time
import numpy as np
import mujoco
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import metaworld
import metaworld.policies as policies

_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from personal.work2.mw_common.state_injection import make_env_with_fixed_state, validate_pick_place_pair
from personal.work2.mw_common.task_ranges import KNOWN_RANGES


def get_expert_policy(task_name):
    policy_class_name = f"Sawyer{task_name.replace('-', ' ').title().replace(' ', '')}Policy"
    try:
        policy_class = getattr(policies, policy_class_name)
        return policy_class()
    except AttributeError:
        raise RuntimeError(f"Cannot find expert policy {policy_class_name} for task {task_name}")


def resize_image(image, target_size):
    from PIL import Image
    img = Image.fromarray(image)
    img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img)


def sync_env_state(src_env, dst_env):
    dst_env.data.qpos[:] = src_env.data.qpos[:]
    dst_env.data.qvel[:] = src_env.data.qvel[:]
    dst_env.data.ctrl[:] = src_env.data.ctrl[:]
    mujoco.mj_forward(dst_env.model, dst_env.data)


def run_episode(env_top, env_wrist, expert_policy, max_steps=500, image_size=224):
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

        top_image = env_top.render()
        if top_image is not None:
            top_image = np.flip(top_image, (0, 1))
            top_image = resize_image(top_image, image_size)

        wrist_image = env_wrist.render()
        if wrist_image is not None:
            wrist_image = resize_image(wrist_image, image_size)

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


def collect_demo(obj_pos, goal_pos, task_name="pick-place-v3", max_steps=500,
                 image_size=224, seed=42, require_success=True):
    """Collect a single demo with specified object and goal positions."""
    rand_vec = np.concatenate([obj_pos, goal_pos])

    if not validate_pick_place_pair(obj_pos[:2], goal_pos[:2]):
        return None, None

    try:
        env_top, _, _ = make_env_with_fixed_state(
            task_name, rand_vec, seed=seed, camera_name="corner2")
        env_wrist, _, _ = make_env_with_fixed_state(
            task_name, rand_vec, seed=seed, camera_name="behindGripper")
    except Exception as e:
        print(f"  Failed to create env: {e}")
        return None, None

    expert_policy = get_expert_policy(task_name)
    frames, ep_info = run_episode(env_top, env_wrist, expert_policy, max_steps, image_size)
    env_top.close()
    env_wrist.close()

    if require_success and not ep_info["success"]:
        return None, None

    return frames, ep_info


def generate_grid_positions(n_per_axis=5, task_name="pick-place-v3"):
    """Generate a uniform grid of object positions within the task workspace."""
    info = KNOWN_RANGES[task_name]
    obj_low = np.array(info["obj_low"])
    obj_high = np.array(info["obj_high"])
    goal_low = np.array(info["goal_low"])
    goal_high = np.array(info["goal_high"])

    obj_x = np.linspace(obj_low[0], obj_high[0], n_per_axis)
    obj_y = np.linspace(obj_low[1], obj_high[1], n_per_axis)
    obj_z = np.array([obj_low[2]])

    goal_x = np.linspace(goal_low[0], goal_high[0], n_per_axis)
    goal_y = np.linspace(goal_low[1], goal_high[1], n_per_axis)
    goal_z = np.linspace(goal_low[2], goal_high[2], n_per_axis)

    positions = []
    for ox in obj_x:
        for oy in obj_y:
            for oz in obj_z:
                obj = np.array([ox, oy, float(oz)])
                for gx in goal_x:
                    for gy in goal_y:
                        for gz in goal_z:
                            goal = np.array([gx, gy, gz])
                            if validate_pick_place_pair(obj[:2], goal[:2]):
                                positions.append((obj, goal))
                                break  # one goal per obj position for B0
                        if positions and np.allclose(positions[-1][0], obj):
                            break
                    if positions and np.allclose(positions[-1][0], obj):
                        break

    # If we have more than needed, take evenly spaced subset
    if len(positions) > n_per_axis * n_per_axis:
        indices = np.linspace(0, len(positions) - 1, n_per_axis * n_per_axis, dtype=int)
        positions = [positions[i] for i in indices]

    return positions


def generate_random_positions(n, task_name="pick-place-v3", seed=42, exclude_grid=None):
    """Generate random object/goal positions, optionally excluding grid positions."""
    info = KNOWN_RANGES[task_name]
    obj_low = np.array(info["obj_low"])
    obj_high = np.array(info["obj_high"])
    goal_low = np.array(info["goal_low"])
    goal_high = np.array(info["goal_high"])

    rng = np.random.default_rng(seed)
    positions = []
    max_attempts = n * 50
    attempts = 0

    exclude_set = set()
    if exclude_grid:
        for obj, goal in exclude_grid:
            exclude_set.add((round(obj[0], 3), round(obj[1], 3)))

    while len(positions) < n and attempts < max_attempts:
        obj = rng.uniform(obj_low, obj_high)
        goal = rng.uniform(goal_low, goal_high)
        attempts += 1

        if validate_pick_place_pair(obj[:2], goal[:2]):
            key = (round(obj[0], 3), round(obj[1], 3))
            if key not in exclude_set:
                positions.append((obj, goal))

    return positions