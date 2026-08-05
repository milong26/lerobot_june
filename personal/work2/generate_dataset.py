#!/usr/bin/env python
"""
Meta-World 示范数据生成器（重构版）。
见 SPEC.md 4.7 节。

相比原版 collect_metaworld_dataset.py 的主要改动:
- 支持可插拔采样策略 (--strategy uniform/grid/boundary/distance_stratified)
- 使用 mw_common/state_injection.py 精确指定 (obj_pos, goal_pos)
- --require-success (默认 True): expert 失败的 episode 不写入数据集
- 断点续采: 已存在数据集时追加而非删除重建
- ENV_STATE_DESCRIPTION 引用 mw_common.obs_utils
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
import mujoco
import metaworld
import metaworld.policies as policies

os.environ["HF_LEROBOT_HOME"] = str(Path(__file__).parent / "outputs")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from personal.work2.mw_common.obs_utils import ENV_STATE_LAYOUT
from personal.work2.mw_common.state_injection import make_env_with_fixed_state, validate_pick_place_pair
from personal.work2.sampling_strategies import get_strategy, STRATEGY_MAP

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
        print(f"错误: 找不到任务 {task_name} 的专家策略 {policy_class_name}")
        sys.exit(1)


def resize_image(image, target_size):
    from PIL import Image
    img = Image.fromarray(image)
    img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img)


def render_dual_camera(env_top, env_wrist, image_size=480):
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


def run_episode(env_top, env_wrist, expert_policy, task_name, max_steps=500, image_size=480):
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

        task_description = TASK_DESCRIPTIONS.get(task_name, task_name)
        frame = {
            "observation.images.top": top_image,
            "observation.images.wrist": wrist_image,
            "observation.state": obs[:4].copy().astype(np.float32),
            "observation.environment_state": obs.copy().astype(np.float32),
            "action": action.copy().astype(np.float32),
            "next.reward": np.array([reward], dtype=np.float32),
            "next.success": np.array([info.get("success", 0)], dtype=bool),
            "task": task_description,
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


def create_dataset(repo_id, output_dir, fps=80, image_size=480):
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
        "task": {"dtype": "string", "shape": (1,)},
    }
    return LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=features, root=output_dir,
        robot_type="metaworld", use_videos=True,
    )


def load_existing_dataset(output_dir):
    """尝试加载已有数据集，返回 (dataset, num_episodes) 或 (None, 0)。"""
    try:
        ds = LeRobotDataset(root=str(output_dir))
        return ds, ds.num_episodes
    except Exception:
        return None, 0


def save_episode_metadata(output_dir, episode_infos, task_name):
    metadata_file = Path(output_dir) / "episode_initial_states.json"
    metadata = {
        "task": task_name,
        "num_episodes": len(episode_infos),
        "env_state_layout": {k: f"slice({v.start}, {v.stop})" for k, v in ENV_STATE_LAYOUT.items()},
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
    parser = argparse.ArgumentParser(description="Meta-World 示范数据生成器")
    parser.add_argument("--task", type=str, default="pick-place-v3")
    parser.add_argument("--strategy", type=str, default="uniform", choices=list(STRATEGY_MAP.keys()))
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--fps", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--require-success", action="store_true", default=True,
                        help="只收录 expert 成功的 episode (默认开启)")
    parser.add_argument("--no-require-success", action="store_true", default=False,
                        help="允许收录失败的 episode")
    args = parser.parse_args()

    require_success = args.require_success and not args.no_require_success

    if args.output_dir is None:
        args.output_dir = f"personal/work2/generated/{args.strategy}_{args.num_episodes}"
    if args.repo_id is None:
        task_short = args.task.replace("-v3", "").replace("-", "_")
        args.repo_id = f"lerobot/metaworld_{task_short}_{args.strategy}"

    output_dir = Path(args.output_dir)

    # 断点续采: 加载已有数据集
    existing_ds, existing_eps = load_existing_dataset(output_dir)
    start_ep = existing_eps
    if existing_eps > 0:
        print(f"检测到已有数据集: {existing_eps} episodes, 将从 episode {existing_eps} 开始追加")
    elif output_dir.exists():
        import shutil
        shutil.rmtree(output_dir)

    rng = np.random.default_rng(args.seed)
    strategy = get_strategy(args.strategy, task_name=args.task)
    samples = strategy.sample(args.num_episodes, rng)

    expert_policy, policy_name = get_expert_policy(args.task)

    print("=" * 80)
    print("Meta-World 数据生成 (重构版)")
    print("=" * 80)
    print(f"任务: {args.task}")
    print(f"策略: {args.strategy}")
    print(f"目标 episode 数: {args.num_episodes}")
    print(f"输出目录: {args.output_dir}")
    print(f"Repo ID: {args.repo_id}")
    print(f"Require success: {require_success}")
    print(f"Expert policy: {policy_name}")
    print("=" * 80)

    if existing_eps == 0:
        dataset = create_dataset(args.repo_id, args.output_dir, args.fps, args.image_size)
    else:
        dataset = existing_ds

    episode_infos = []
    if existing_eps > 0:
        metadata_file = output_dir / "episode_initial_states.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                old_meta = json.load(f)
            episode_infos = old_meta.get("episodes", [])

    success_count = 0
    fail_count = 0
    start_time = time.time()

    print(f"\n开始生成 {args.num_episodes} 个episode (从 #{start_ep} 开始)...")
    print("-" * 80)

    for ep_idx in range(start_ep, args.num_episodes):
        ep_start = time.time()
        obj, goal = samples[ep_idx]
        rand_vec = np.concatenate([obj, goal])

        if not validate_pick_place_pair(obj[:2], goal[:2]):
            print(f"  Episode {ep_idx}: 跳过 (obj/goal 不满足平面距离约束)")
            fail_count += 1
            continue

        try:
            env_top, _, _ = make_env_with_fixed_state(args.task, rand_vec, seed=args.seed, camera_name="corner2")
            env_wrist, _, _ = make_env_with_fixed_state(args.task, rand_vec, seed=args.seed, camera_name="behindGripper")
        except Exception as e:
            print(f"  Episode {ep_idx}: 创建环境失败 ({e})")
            fail_count += 1
            continue

        frames, ep_info = run_episode(env_top, env_wrist, expert_policy, args.task, args.max_steps, args.image_size)
        env_top.close()
        env_wrist.close()

        if require_success and not ep_info["success"]:
            print(
                f"  Episode {ep_idx:3d}/{args.num_episodes} | "
                f"Expert 失败, 不收录 | "
                f"Obj: [{obj[0]:.3f}, {obj[1]:.3f}, {obj[2]:.3f}] | "
                f"Time: {time.time() - ep_start:.2f}s"
            )
            episode_infos.append({**ep_info, "obj_init_pos": obj, "goal_pose": goal, "included": False})
            fail_count += 1
            continue

        for frame in frames:
            dataset.add_frame(frame)
        dataset.save_episode()

        episode_infos.append({**ep_info, "obj_init_pos": obj, "goal_pose": goal, "included": True})
        if ep_info["success"]:
            success_count += 1

        print(
            f"  Episode {ep_idx:3d}/{args.num_episodes} | "
            f"Frames: {ep_info['num_frames']:4d} | "
            f"Success: {'✓' if ep_info['success'] else '✗'} | "
            f"Obj: [{obj[0]:.3f}, {obj[1]:.3f}, {obj[2]:.3f}] | "
            f"Time: {time.time() - ep_start:.2f}s"
        )

    dataset.finalize()
    save_episode_metadata(args.output_dir, episode_infos, args.task)

    total_time = time.time() - start_time
    included = sum(1 for e in episode_infos if e.get("included", True))
    print("\n" + "=" * 80)
    print("生成完成!")
    print("=" * 80)
    print(f"目标 episode 数: {args.num_episodes}")
    print(f"实际收录 episode 数: {included}")
    print(f"Expert 成功: {success_count}")
    print(f"失败/跳过: {fail_count}")
    print(f"总用时: {total_time:.1f}s")
    print(f"数据集路径: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()