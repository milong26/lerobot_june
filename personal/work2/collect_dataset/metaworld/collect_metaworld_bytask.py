#!/usr/bin/env python
"""
两阶段采集 Meta-World 数据集：
  阶段1 — 随机采样：通过 seed 递增随机采集 N 个 episode
  阶段2 — 均匀采样：对 reset randomization space 进行 Latin Hypercube 均匀覆盖采样 M 个 episode

两阶段共享同一个 LeRobot 数据集（resume 模式追加），最终生成统一的 episode_initial_states.json。

使用示例:
    # 阶段1 采集300个随机episode + 阶段2 采集100个uniform episode
    python collect_metaworld_bytask.py \
        --task coffee-push-v3 \
        --num-random-episodes 300 \
        --num-uniform-episodes 100 \
        --output-dir personal/work2/dataset_view/coffee_push_corner3/ \
        --repo-id work2/coffee_push_corner3 \
        --seed-start 0 \
        --streaming-encoding \
        --encoder-threads 2

    # 只采集随机阶段
    python collect_metaworld_bytask.py \
        --task pick-place-v3 \
        --num-random-episodes 200 \
        --num-uniform-episodes 0 \
        --output-dir ./outputs/pick_place \
        --repo-id work2/pick_place

    # 只采集uniform阶段（resume已有数据集）
    python collect_metaworld_bytask.py \
        --task pick-place-v3 \
        --num-random-episodes 0 \
        --num-uniform-episodes 100 \
        --output-dir ./outputs/pick_place \
        --repo-id work2/pick_place
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

CONFIG_PATH = Path(__file__).parent.parent.parent.parent.parent / "src" / "lerobot" / "envs" / "metaworld_config.json"
with open(CONFIG_PATH) as f:
    METAWORLD_CONFIG = json.load(f)

TASK_DESCRIPTIONS = METAWORLD_CONFIG.get("TASK_DESCRIPTIONS", {})

MAX_REJECT_ATTEMPTS = 200
MAX_ATTEMPTS_PER_EPISODE = 50
EXTRA_FRAMES_AFTER_SUCCESS = 10


class RandVecExhaustedError(RuntimeError):
    pass


# ─── 环境 & 任务解析 ────────────────────────────────────────────────

def create_metaworld_env(task_name, seed=None, camera_name="corner"):
    mt1 = metaworld.MT1(task_name, seed=seed)
    env = mt1.train_classes[task_name](render_mode="rgb_array", camera_name=camera_name)
    task = mt1.train_tasks[0]
    env.set_task(task)
    env._freeze_rand_vec = True
    return env


def introspect_task_space(task_name, seed=42):
    env = create_metaworld_env(task_name, seed=seed)
    env._freeze_rand_vec = False
    space = env._random_reset_space
    assert space is not None, f"{task_name} 的 _random_reset_space 为 None"
    low = space.low.copy()
    high = space.high.copy()
    dim = space.shape[0]
    has_obj = hasattr(env, "obj_init_pos") and env.obj_init_pos is not None
    obj_pos_dim = len(env.obj_init_pos) if has_obj else 0
    has_goal = hasattr(env, "goal") and env.goal is not None
    goal_pos_dim = len(env.goal) if has_goal else 0
    env.reset()
    env.close()
    return {
        "space": space, "low": low, "high": high, "dim": dim,
        "has_obj": has_obj, "obj_pos_dim": obj_pos_dim,
        "has_goal": has_goal, "goal_pos_dim": goal_pos_dim,
    }


def sync_env_state(src_env, dst_env):
    dst_env.data.qpos[:] = src_env.data.qpos[:]
    dst_env.data.qvel[:] = src_env.data.qvel[:]
    dst_env.data.ctrl[:] = src_env.data.ctrl[:]
    mujoco.mj_forward(dst_env.model, dst_env.data)


def render_dual_camera(env_top, env_wrist, image_size=480):
    top_image = env_top.render()
    if top_image is not None:
        top_image = np.flip(top_image, (0, 1))
        top_image = resize_image(top_image, image_size)
    wrist_image = env_wrist.render()
    if wrist_image is not None:
        wrist_image = resize_image(wrist_image, image_size)
    return top_image, wrist_image


def resize_image(image, target_size):
    from PIL import Image
    return np.array(Image.fromarray(image).resize((target_size, target_size), Image.BILINEAR))


def get_obj_pose_from_env(env):
    if hasattr(env, "obj_init_pos") and env.obj_init_pos is not None:
        return env.obj_init_pos.copy()
    return None


def get_goal_pose_from_env(env):
    if hasattr(env, "goal") and env.goal is not None:
        return env.goal.copy()
    return None


def get_rand_vec_from_env(env):
    if hasattr(env, "_random_reset_space") and env._random_reset_space is not None:
        if hasattr(env, "_get_state_rand_vec"):
            try:
                return env._get_state_rand_vec().copy()
            except Exception:
                pass
    return None


def inject_rand_vec(env, rand_vec, max_reject_attempts=MAX_REJECT_ATTEMPTS):
    env._freeze_rand_vec = False
    env.seeded_rand_vec = False
    vec = np.asarray(rand_vec, dtype=np.float64).copy()
    state = {"calls": 0}
    def _patched():
        state["calls"] += 1
        if state["calls"] > max_reject_attempts:
            raise RandVecExhaustedError(
                f"rand_vec 被任务内部拒绝采样循环拒绝超过 {max_reject_attempts} 次，"
                "该初始状态对当前 task 不合法。"
            )
        return vec.copy()
    env._get_state_rand_vec = _patched


def set_task_state_directly(env, rand_vec):
    inject_rand_vec(env, rand_vec)


def perturb_rand_vec(rand_vec, low, high, scale=0.05, rng=None,
                     existing_rand_vecs=None, invalid_rand_vecs=None,
                     num_candidates=20):
    if rng is None:
        rng = np.random.RandomState()
    vec = rand_vec.copy()
    dim = len(vec)
    range_sizes = high - low

    if existing_rand_vecs is None or len(existing_rand_vecs) == 0:
        for i in range(dim):
            delta = rng.uniform(-scale * range_sizes[i], scale * range_sizes[i])
            vec[i] = np.clip(vec[i] + delta, low[i], high[i])
        return vec

    existing_arr = np.array(existing_rand_vecs)
    invalid_arr = np.array(invalid_rand_vecs) if invalid_rand_vecs and len(invalid_rand_vecs) > 0 else None

    candidates = []
    for _ in range(num_candidates):
        cand = vec.copy()
        for i in range(dim):
            delta = rng.uniform(-scale * range_sizes[i], scale * range_sizes[i])
            cand[i] = np.clip(cand[i] + delta, low[i], high[i])
        candidates.append(cand)

    def normalized_min_dist(cand, ref_arr):
        if ref_arr is None or len(ref_arr) == 0:
            return 0.0
        diff = (ref_arr - cand) / range_sizes
        dists = np.sqrt(np.sum(diff ** 2, axis=1))
        return np.min(dists)

    best_score = -1.0
    best_cand = vec.copy()

    for cand in candidates:
        dist_valid = normalized_min_dist(cand, existing_arr)
        dist_invalid = normalized_min_dist(cand, invalid_arr)
        score = dist_valid
        if invalid_arr is not None and len(invalid_arr) > 0:
            score += 0.5 * min(dist_invalid, 1.0)
        if score > best_score:
            best_score = score
            best_cand = cand.copy()

    return best_cand


def try_collect_episode(env_top, env_wrist, expert_policy, task_name, rand_vec,
                        max_steps, image_size, extra_frames_after_success):
    set_task_state_directly(env_top, rand_vec)
    set_task_state_directly(env_wrist, rand_vec)
    env_top.reset()
    env_wrist.reset()
    sync_env_state(env_top, env_wrist)
    frames, ep_info = run_episode(
        env_top, env_wrist, expert_policy, task_name,
        max_steps, image_size, extra_frames_after_success,
    )
    ep_info["rand_vec"] = rand_vec
    return frames, ep_info


# ─── Uniform 采样 ────────────────────────────────────────────────────

def latin_hypercube_sample(dim, n_samples, low, high, rng):
    samples = np.zeros((n_samples, dim))
    for d in range(dim):
        edges = np.linspace(low[d], high[d], n_samples + 1)
        points = rng.uniform(edges[:-1], edges[1:])
        rng.shuffle(points)
        samples[:, d] = points
    return samples


def compute_uniform_samples(task_space_info, num_samples, existing_rand_vecs=None, task_name=None):
    low, high, dim = task_space_info["low"], task_space_info["high"], task_space_info["dim"]
    existing_set = set()
    if existing_rand_vecs:
        for s in existing_rand_vecs:
            existing_set.add(tuple(np.round(s, 4)))

    rng = np.random.RandomState(42)
    samples = []

    for _ in range(num_samples * 200):
        if len(samples) >= num_samples:
            break
        lhc = latin_hypercube_sample(dim, 1, low, high, rng)
        vec = lhc[0]
        key = tuple(np.round(vec, 4))
        if key in existing_set:
            continue
        samples.append(vec.copy())
        existing_set.add(key)

    return samples


# ─── Episode 运行 ────────────────────────────────────────────────────

def run_episode(env_top, env_wrist, expert_policy, task_name,
                max_steps=500, image_size=480, extra_frames_after_success=EXTRA_FRAMES_AFTER_SUCCESS):
    obs, info = env_top.reset()
    env_wrist.reset()
    sync_env_state(env_top, env_wrist)
    obj_pose = get_obj_pose_from_env(env_top)
    goal_pose = get_goal_pose_from_env(env_top)
    rand_vec = get_rand_vec_from_env(env_top)

    frames, success_flags = [], []
    success_detected, frames_after_success = False, 0

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
            "task": TASK_DESCRIPTIONS.get(task_name, task_name),
        }
        frames.append(frame)
        success_flags.append(info.get("success", 0))

        if info.get("success", 0) and not success_detected:
            success_detected = True
            frames_after_success = 0
            print(f"  >>> Success at step {step}, collecting {extra_frames_after_success} more frames...")
        if success_detected:
            frames_after_success += 1
            if frames_after_success >= extra_frames_after_success:
                break
        if terminated or truncated:
            break

    return frames, {
        "obj_init_pos": obj_pose, "goal_pose": goal_pose, "rand_vec": rand_vec,
        "success": any(success_flags), "num_frames": len(frames),
    }


# ─── 数据集 I/O ──────────────────────────────────────────────────────

def create_dataset(repo_id, output_dir, fps=80, image_size=480,
                   streaming_encoding=False, encoder_threads=None,
                   image_writer_processes=0, image_writer_threads=8,
                   batch_encoding_size=1, vcodec=None):
    from lerobot.configs.video import rgb_encoder_defaults, RGBEncoderConfig
    features = {
        "observation.images.top": {"dtype": "video", "shape": (3, image_size, image_size), "names": ["channels", "height", "width"]},
        "observation.images.wrist": {"dtype": "video", "shape": (3, image_size, image_size), "names": ["channels", "height", "width"]},
        "observation.state": {"dtype": "float32", "shape": (4,)},
        "observation.environment_state": {"dtype": "float32", "shape": (39,), "names": ["keypoints"]},
        "action": {"dtype": "float32", "shape": (4,), "names": {"axes": ["x", "y", "z", "gripper"]}},
        "next.reward": {"dtype": "float32", "shape": (1,)},
        "next.success": {"dtype": "bool", "shape": (1,)},
    }
    rgb_enc = RGBEncoderConfig(vcodec=vcodec) if vcodec else rgb_encoder_defaults()
    total_writer_threads = image_writer_threads * 2 if image_writer_threads > 0 else 0
    return LeRobotDataset.create(
        repo_id=repo_id, fps=fps, features=features, root=output_dir,
        robot_type="metaworld", use_videos=True,
        image_writer_processes=image_writer_processes, image_writer_threads=total_writer_threads,
        batch_encoding_size=batch_encoding_size, rgb_encoder=rgb_enc,
        encoder_threads=encoder_threads, streaming_encoding=streaming_encoding,
    )


def load_existing_states(metadata_file):
    if not Path(metadata_file).exists():
        return []
    with open(metadata_file, "r") as f:
        old = json.load(f)
    return [np.array(ep["rand_vec"]) for ep in old.get("episodes", []) if "rand_vec" in ep]


def save_episode_metadata(output_dir, all_episode_infos, task_name, task_space_info=None):
    metadata_file = Path(output_dir) / "episode_initial_states.json"
    env_state_structure = {}
    if task_space_info:
        env_state_structure = {
            "reset_space_dim": task_space_info["dim"],
            "reset_space_low": task_space_info["low"].tolist(),
            "reset_space_high": task_space_info["high"].tolist(),
            "has_obj": task_space_info["has_obj"],
            "obj_pos_dim": task_space_info["obj_pos_dim"],
            "has_goal": task_space_info["has_goal"],
            "goal_pos_dim": task_space_info["goal_pos_dim"],
        }
    metadata = {
        "task": task_name, "num_episodes": len(all_episode_infos),
        "env_state_structure": env_state_structure, "episodes": [],
    }
    for i, info in enumerate(all_episode_infos):
        ep = {"episode_index": i, "success": bool(info["success"]), "num_frames": info["num_frames"]}
        if info.get("obj_init_pos") is not None:
            ep["obj_init_pos"] = info["obj_init_pos"].tolist()
        if info.get("goal_pose") is not None:
            ep["goal_pose"] = info["goal_pose"].tolist()
        if info.get("rand_vec") is not None:
            ep["rand_vec"] = info["rand_vec"].tolist()
        metadata["episodes"].append(ep)
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nEpisode初始环境信息已保存到: {metadata_file}")


# ─── 阶段1: 随机采集 ─────────────────────────────────────────────────

def phase_random(args, dataset, expert_policy, task_space_info, start_ep_idx=0):
    print(f"\n{'='*60}")
    print(f"阶段1: 随机采集 ({args.num_random_episodes} 个 episode)")
    print(f"{'='*60}")
    seed_start = args.seed_start
    episode_infos = []
    success_count = 0
    ep_idx = 0

    while ep_idx < args.num_random_episodes:
        ep_start = time.time()
        seed = seed_start + ep_idx
        env_top = create_metaworld_env(args.task, seed=seed, camera_name="corner")
        env_wrist = create_metaworld_env(args.task, seed=seed, camera_name="gripperPOV")
        frames, ep_info = run_episode(
            env_top, env_wrist, expert_policy, args.task,
            args.max_steps, args.image_size, args.extra_frames_after_success,
        )
        env_top.close()
        env_wrist.close()

        if ep_info["success"]:
            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            episode_infos.append(ep_info)
            success_count += 1
            elapsed = time.time() - ep_start
            obj_str = ""
            if ep_info.get("obj_init_pos") is not None:
                p = ep_info["obj_init_pos"]
                obj_str = f" | obj: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]"
            print(f"  [R] Episode {start_ep_idx + ep_idx + 1:4d} | Frames: {ep_info['num_frames']:4d} | Success{obj_str} | {elapsed:.1f}s")
            ep_idx += 1
        else:
            elapsed = time.time() - ep_start
            print(f"  [R] FAILED (seed={seed}) | Frames: {ep_info['num_frames']:4d} | {elapsed:.1f}s")

    print(f"阶段1 完成: 成功 {success_count}/{args.num_random_episodes}")
    return episode_infos


# ─── 阶段2: Uniform 采集 ─────────────────────────────────────────────

def phase_uniform(args, dataset, expert_policy, task_space_info, start_ep_idx=0):
    print(f"\n{'='*60}")
    print(f"阶段2: 均匀采样采集 ({args.num_uniform_episodes} 个 episode)")
    print(f"{'='*60}")

    metadata_file = Path(args.output_dir) / "episode_initial_states.json"
    existing_rand_vecs = load_existing_states(metadata_file)
    print(f"已有 {len(existing_rand_vecs)} 个 rand_vec 记录")

    print(f"生成 {args.num_uniform_episodes} 个 Latin Hypercube 采样点...")
    grid_centers = compute_uniform_samples(task_space_info, args.num_uniform_episodes, existing_rand_vecs, args.task)
    print(f"共 {len(grid_centers)} 个目标初始状态")

    episode_infos = []
    success_count = 0
    total_attempts = 0
    sample_idx = 0
    use_seed_fallback = False
    fallback_seed = args.seed_start + args.num_random_episodes

    perturbation_rng = np.random.RandomState(123)

    invalid_rand_vecs = []
    invalid_failure_threshold = 3

    while success_count < args.num_uniform_episodes:
        if sample_idx >= len(grid_centers):
            if not use_seed_fallback:
                print(f"  LHC 采样点已用完，切换到基于 seed 的随机采样...")
                use_seed_fallback = True
            new_samples = compute_uniform_samples(
                task_space_info, args.num_uniform_episodes - success_count, existing_rand_vecs, args.task
            )
            if not new_samples:
                print("错误: 无法生成新的不重复采样点，终止采集。")
                break
            grid_centers.extend(new_samples)
            print(f"新增 {len(new_samples)} 个采样点")

        rand_vec = grid_centers[sample_idx]
        sample_idx += 1
        total_attempts += 1

        collected = False
        consecutive_failures = 0
        current_vec = rand_vec

        for retry in range(5):
            try:
                env_top = create_metaworld_env(args.task, seed=42, camera_name="corner")
                env_wrist = create_metaworld_env(args.task, seed=42, camera_name="gripperPOV")
                if retry == 0:
                    current_vec = rand_vec
                else:
                    current_vec = perturb_rand_vec(
                        current_vec, task_space_info["low"], task_space_info["high"],
                        rng=perturbation_rng,
                        existing_rand_vecs=existing_rand_vecs,
                        invalid_rand_vecs=invalid_rand_vecs,
                        num_candidates=20,
                    )
                frames, ep_info = try_collect_episode(
                    env_top, env_wrist, expert_policy, args.task, current_vec,
                    args.max_steps, args.image_size, args.extra_frames_after_success,
                )
                env_top.close()
                env_wrist.close()
            except RandVecExhaustedError:
                consecutive_failures += 1
                if retry < 4:
                    print(f"  [U] 采样点被拒绝，尝试自适应扰动 ({retry+1}/4)...")
                    continue
                else:
                    print(f"  [U] SKIPPED: 自适应扰动4次后仍不合法，跳过")
                    invalid_rand_vecs.append(rand_vec.copy())
                    break

            if ep_info["success"]:
                for frame in frames:
                    dataset.add_frame(frame)
                dataset.save_episode()
                ep_info["_sampling_method"] = "fallback_random" if use_seed_fallback else "latin_hypercube"
                episode_infos.append(ep_info)
                success_count += 1
                existing_rand_vecs.append(current_vec.copy())
                obj_str = ""
                if ep_info.get("obj_init_pos") is not None:
                    p = ep_info["obj_init_pos"]
                    obj_str = f" | obj: [{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]"
                print(f"  [U] Episode {start_ep_idx + success_count:4d} | Frames: {ep_info['num_frames']:4d} | Success{obj_str}")
                collected = True
                break
            else:
                consecutive_failures += 1
                print(f"  [U] Episode FAILED (no success), skipping...")
                if consecutive_failures >= invalid_failure_threshold:
                    invalid_rand_vecs.append(rand_vec.copy())
                break

        if not collected and use_seed_fallback:
            fallback_seed += 1
            new_vec = np.random.RandomState(fallback_seed).uniform(
                task_space_info["low"], task_space_info["high"]
            )
            grid_centers.append(new_vec)

    print(f"阶段2 完成: 成功 {success_count}/{args.num_uniform_episodes} (总尝试 {total_attempts})")
    fallback_count = sum(1 for info in episode_infos if info.get("_sampling_method") == "fallback_random")
    lhc_count = success_count - fallback_count
    print(f"  - LHC 采样成功: {lhc_count} 个")
    print(f"  - Fallback 随机采样成功: {fallback_count} 个")
    print(f"  - 标记为非法的采样点: {len(invalid_rand_vecs)} 个")
    return episode_infos, fallback_count


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="两阶段采集 Meta-World 数据集：随机 + Uniform 均匀采样",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", type=str, default="pick-place-v3")
    parser.add_argument("--num-random-episodes", type=int, default=300,
                        help="阶段1: 随机采集的 episode 数量 (默认: 300)")
    parser.add_argument("--num-uniform-episodes", type=int, default=100,
                        help="阶段2: Uniform 均匀采样的 episode 数量 (默认: 100)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--repo-id", type=str, default=None)
    parser.add_argument("--fps", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=480)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-attempts-per-episode", type=int, default=MAX_ATTEMPTS_PER_EPISODE)
    parser.add_argument("--extra-frames-after-success", type=int, default=EXTRA_FRAMES_AFTER_SUCCESS)
    parser.add_argument("--streaming-encoding", action="store_true", default=False)
    parser.add_argument("--encoder-threads", type=int, default=None)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--batch-encoding-size", type=int, default=1)
    parser.add_argument("--vcodec", type=str, default=None)

    args = parser.parse_args()

    if args.output_dir is None:
        task_short = args.task.replace("-v3", "").replace("-", "_")
        args.output_dir = f"./outputs/metaworld_{task_short}"
    if args.repo_id is None:
        task_short = args.task.replace("-v3", "").replace("-", "_")
        args.repo_id = f"lerobot/metaworld_{task_short}"

    output_dir = Path(args.output_dir)
    is_resume = output_dir.exists() and (output_dir / "episode_initial_states.json").exists()

    print("=" * 80)
    print("Meta-World 两阶段数据采集 (随机 + Uniform)")
    print("=" * 80)
    print(f"任务: {args.task}")
    print(f"阶段1 随机采集: {args.num_random_episodes} episodes")
    print(f"阶段2 Uniform采集: {args.num_uniform_episodes} episodes")
    print(f"输出目录: {args.output_dir}")
    print(f"Repo ID: {args.repo_id}")
    print(f"FPS: {args.fps}")
    print(f"图像分辨率: {args.image_size}x{args.image_size}")
    print(f"Seed起始: {args.seed_start}")
    print(f"流式编码: {'是' if args.streaming_encoding else '否'}")
    if args.streaming_encoding:
        print(f"  编码器线程: {args.encoder_threads or 'auto'}")
        print(f"  视频编码器: {args.vcodec or 'libsvtav1(默认)'}")
    print(f"模式: {'Resume (追加到已有数据集)' if is_resume else '从头开始'}")
    print("=" * 80)

    if args.num_random_episodes == 0 and args.num_uniform_episodes == 0:
        print("错误: 两个阶段的 episode 数量都为 0，无需采集。")
        sys.exit(1)

    policy_class_name = f"Sawyer{args.task.replace('-', ' ').title().replace(' ', '')}Policy"
    try:
        expert_policy = getattr(policies, policy_class_name)()
        print(f"\n使用专家策略: {policy_class_name}")
    except AttributeError:
        print(f"\n错误: 找不到任务 {args.task} 的专家策略 {policy_class_name}")
        sys.exit(1)

    print("\n解析任务的 reset randomization space...")
    task_space_info = introspect_task_space(args.task)
    print(f"  Reset space dim: {task_space_info['dim']}")
    print(f"  Has obj: {task_space_info['has_obj']} (dim={task_space_info['obj_pos_dim']})")
    print(f"  Has goal: {task_space_info['has_goal']} (dim={task_space_info['goal_pos_dim']})")
    print(f"  Space low: {task_space_info['low'].tolist()}")
    print(f"  Space high: {task_space_info['high'].tolist()}")

    # 创建/加载数据集
    if is_resume:
        print(f"\n加载已有LeRobot数据集 (resume)...")
        dataset = LeRobotDataset.resume(repo_id=args.repo_id, root=args.output_dir)
        existing_episodes = dataset.num_episodes
        print(f"已有 {existing_episodes} 个 episode")
    else:
        if output_dir.exists():
            print(f"警告: 输出目录已存在，将删除并重建: {args.output_dir}")
            import shutil
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

    # 阶段1: 随机采集
    if args.num_random_episodes > 0:
        random_infos = phase_random(args, dataset, expert_policy, task_space_info, start_ep_idx=existing_episodes)
        all_episode_infos.extend(random_infos)
        existing_episodes += len(random_infos)

    # 阶段2: Uniform 采集
    fallback_count = 0
    if args.num_uniform_episodes > 0:
        uniform_infos, fallback_count = phase_uniform(args, dataset, expert_policy, task_space_info, start_ep_idx=existing_episodes)
        all_episode_infos.extend(uniform_infos)

    # 完成数据集
    print("\n" + "-" * 80)
    print("正在保存数据集...")
    dataset.finalize()

    # 保存 metadata
    save_episode_metadata(args.output_dir, all_episode_infos, args.task, task_space_info)

    total_time = time.time() - start_time
    total_success = len(all_episode_infos)
    print("\n" + "=" * 80)
    print("采集完成！")
    print("=" * 80)
    print(f"阶段1 成功: {len([i for i in all_episode_infos if i.get('_phase') == 'random']) if any('_phase' in i for i in all_episode_infos) else args.num_random_episodes}")
    print(f"阶段2 成功: {len([i for i in all_episode_infos if i.get('_phase') == 'uniform']) if any('_phase' in i for i in all_episode_infos) else args.num_uniform_episodes}")
    if fallback_count > 0:
        print(f"  └─ 其中 Fallback 随机采样: {fallback_count} 个 (LHC 采样点耗尽后回退)")
    print(f"本次新增总Episode: {total_success}")
    print(f"数据集总Episode: {dataset.num_episodes}")
    print(f"总用时: {total_time:.1f}s")
    print(f"数据集路径: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()