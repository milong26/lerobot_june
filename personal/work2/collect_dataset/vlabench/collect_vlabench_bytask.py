#!/usr/bin/env python3
"""Collect VLABench expert demonstrations into one LeRobot dataset.

Default task is ``select_fruit``.  The collector has two phases:
1. seeded random resets (300 successful episodes);
2. spatial grid/combinations of movable entities (100 successful episodes).

Only successful episodes are saved.  The initial simulator configuration of
every saved episode is written to episode_initial_states.json.
"""
import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
from scipy.spatial.transform import Rotation

# Import these modules for their registration side effects.  VLABench's
# registry is populated by decorators when robot/task modules are imported.
import VLABench.robots  # noqa: F401
import VLABench.tasks  # noqa: F401
from VLABench.envs import load_env
from VLABench.utils.utils import get_logger

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

EXTRA_STEPS = 10
DEFAULT_TASK = "select_fruit"


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def make_env(task, seed, reset_wait_step, num_objects, episode_config=None):
    # VLABench uses numpy/random during task construction; seed both so the
    # seed phase is reproducible even though load_env has no seed argument.
    np.random.seed(seed)
    random.seed(seed)
    return load_env(task, robot="franka", random_init=episode_config is None,
                    episode_config=episode_config, num_objects=num_objects, reset_wait_step=reset_wait_step,
                    run_mode="train")


def camera_indices(env):
    names = []
    for i in range(env.physics.model.ncam):
        try:
            names.append(env.physics.model.camera(i).name)
        except Exception:
            names.append("")
    global_i = next((i for i, n in enumerate(names) if n == "forward"), None)
    wrist_i = next((i for i, n in enumerate(names) if "wrist" in n.lower()), None)
    if global_i is None or wrist_i is None or global_i == wrist_i:
        raise RuntimeError(f"required fixed cameras forward+wrist not found: {names}")
    return global_i, wrist_i, names


def entity_state(env):
    result = {}
    for name, entity in env.task.entities.items():
        state = {"name": name, "class": type(entity).__name__}
        try:
            body = entity.mjcf_model.worldbody
            bound = env.physics.bind(body)
            state["position"] = np.asarray(bound.xpos).copy()
            state["quaternion"] = np.asarray(bound.xquat).copy()
        except Exception:
            continue
        result[name] = jsonable(state)
    return result


def movable_entities(env):
    target_container = getattr(env.task.config_manager, "target_container", None)
    out = []
    for name, entity in env.task.entities.items():
        is_container = name == target_container or hasattr(entity, "contain")
        is_object = hasattr(entity, "get_xpos") and hasattr(entity, "set_pose")
        if is_object and not is_container:
            out.append((name, entity))
    return out


def set_entity_xy(env, entity, xy):
    try:
        pos = np.asarray(entity.get_xpos(env.physics), dtype=np.float64).copy()
        quat = np.asarray(entity.get_xqaut(env.physics), dtype=np.float64).copy()
        pos[:2] = xy
        entity.set_pose(env.physics, pos, quat)
        env.physics.forward()
        return True
    except Exception:
        return False


def grid_candidates(env, grid_size, xy_radius, max_candidates, seed, cell_jitter):
    entities = movable_entities(env)
    if not entities:
        return []
    rng = np.random.RandomState(seed)
    edges = np.linspace(-xy_radius, xy_radius, grid_size + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    cell_width = edges[1] - edges[0]
    offsets = np.array([(x, y) for x in centers for y in centers], dtype=np.float64)
    rng.shuffle(offsets)
    jitter_limit = 0.5 * cell_width * cell_jitter
    offsets += rng.uniform(-jitter_limit, jitter_limit, size=offsets.shape)
    bases = [np.asarray(e.get_xpos(env.physics), dtype=np.float64).copy() for _, e in entities]
    n_points = len(offsets)
    total = n_points ** len(entities)
    count = min(max_candidates, total)
    combination_ids = np.unique(np.linspace(0, total - 1, count, dtype=np.int64))
    candidates = []
    for combination_id in combination_ids:
        value, placement = int(combination_id), []
        for i, (name, _) in enumerate(entities):
            point_i, value = value % n_points, value // n_points
            placement.append((name, bases[i][:2] + offsets[point_i]))
        candidates.append(placement)
    return candidates


class EpisodeStepLimit(RuntimeError):
    pass


def run_expert(env, max_steps, extra_steps):
    observations, actions, success = [], [], False
    original_step, step_count = env.step, 0
    def limited_step(action=None):
        nonlocal step_count
        if action is not None:
            step_count += 1
            if step_count > max_steps:
                raise EpisodeStepLimit(f"expert exceeded {max_steps} steps")
        return original_step(action)
    env.step = limited_step
    try:
        for skill in env.get_expert_skill_sequence() or []:
            obs, waypoints, stage_ok, task_ok = skill(env)
            observations.extend(obs or []); actions.extend(waypoints or [])
            if task_ok:
                success = True; break
            if not stage_ok:
                break
    finally:
        env.step = original_step
    if not success or not observations or len(observations) != len(actions):
        return None
    for _ in range(extra_steps):
        last = np.asarray(actions[-1])
        quat_xyzw = Rotation.from_euler("xyz", last[3:6]).as_quat()
        quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        ok, qpos = env.robot.get_qpos_from_ee_pos(env.physics, last[:3], quat)
        if not ok:
            qpos = env.robot.get_qpos(env.physics)
        env.physics.data.ctrl[:] = np.concatenate([qpos, last[-2:]])
        env.physics.step()
        observations.append(env.get_observation()); actions.append(last.copy())
    return observations, actions


def create_dataset(root, repo_id, image_size, fps):
    # Re-open an initialized dataset after an interrupted collection.
    info_file = root / "meta" / "info.json"
    if info_file.exists():
        # resume() opens a local dataset directly in write mode.  Calling the
        # normal constructor may try to resolve repo_id on Hugging Face when
        # no parquet episodes exist yet, which is wrong for local collection.
        return LeRobotDataset.resume(
            repo_id=repo_id,
            root=str(root),
            image_writer_processes=0,
            image_writer_threads=4,
        )
    if root.exists() and not any(root.iterdir()):
        # LeRobotDataset.create requires the root itself not to exist.
        root.rmdir()
    return LeRobotDataset.create(
        repo_id=repo_id, root=str(root), robot_type="franka", fps=fps,
        use_videos=True, image_writer_processes=0, image_writer_threads=4,
        features={
            "observation.images.global": {"dtype": "video", "shape": (3, image_size, image_size), "names": ["channels", "height", "width"]},
            "observation.images.wrist": {"dtype": "video", "shape": (3, image_size, image_size), "names": ["channels", "height", "width"]},
            "observation.state": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
            "action": {"dtype": "float32", "shape": (7,), "names": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]},
        })


def add_episode(dataset, observations, actions, gi, wi, size, instruction, robot_origin):
    from PIL import Image
    for obs, action in zip(observations, actions):
        rgb = np.asarray(obs["rgb"])
        def image(i):
            return np.asarray(Image.fromarray(rgb[i]).resize((size, size)))
        ee = np.asarray(obs["ee_state"], dtype=np.float64).reshape(-1)
        pos = ee[:3] - robot_origin
        quat_xyzw = np.array([ee[4], ee[5], ee[6], ee[3]])
        euler = Rotation.from_quat(quat_xyzw).as_euler("xyz")
        state = np.concatenate([pos, euler, [float(ee[7] > 0.5)]]).astype(np.float32)
        raw_action = np.asarray(action, dtype=np.float32).reshape(-1)
        raw_action[:3] -= robot_origin
        act = np.concatenate([raw_action[:6], [float(raw_action[-1] > 0.03)]]).astype(np.float32)
        dataset.add_frame({"observation.images.global": image(gi),
                           "observation.images.wrist": image(wi),
                           "observation.state": state, "action": act,
                           "task": instruction})
    dataset.save_episode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--random-episodes", type=int, default=300)
    p.add_argument("--uniform-episodes", type=int, default=100)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--output-root", default="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view_vlabench")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--image-size", type=int, default=480)
    p.add_argument("--num-objects", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--grid-size", type=int, default=12)
    p.add_argument("--grid-radius", type=float, default=0.08)
    p.add_argument("--cell-jitter", type=float, default=0.8,
                   help="fraction of half-cell width used for in-cell stratified jitter")
    p.add_argument("--uniform-retries", type=int, default=5)
    p.add_argument("--uniform-seed", type=int, default=42)
    p.add_argument("--perturb-scale", type=float, default=0.15)
    p.add_argument("--max-consecutive-failures", type=int, default=200)
    p.add_argument("--reset-wait-step", type=int, default=10)
    args = p.parse_args()
    if not 0.0 <= args.cell_jitter <= 1.0:
        p.error("--cell-jitter must be in [0, 1]")
    available_cells = args.grid_size ** (2 * args.num_objects)
    if args.uniform_episodes > available_cells:
        p.error(f"uniform episodes require {args.uniform_episodes} cells, only {available_cells} combinations available")
    root = Path(args.output_root) / args.task
    repo_id = args.repo_id or f"vlabench_{args.task}"
    meta_path = root / "episode_initial_states.json"
    records = json.loads(meta_path.read_text()) if meta_path.exists() else {"task": args.task, "num_objects": args.num_objects, "episodes": []}
    if records.get("num_objects", args.num_objects) != args.num_objects:
        raise RuntimeError("existing dataset uses a different --num-objects value")
    records.setdefault("num_objects", args.num_objects)
    records.setdefault("uniform_sampling", {
        "method": "stratified_grid_with_in_cell_jitter",
        "grid_size": args.grid_size, "grid_radius": args.grid_radius,
        "cell_jitter": args.cell_jitter, "uniform_seed": args.uniform_seed,
        "available_cell_combinations": available_cells,
        "selected_training_cells": args.uniform_episodes,
        "held_out_cell_combinations": available_cells - args.uniform_episodes,
    })
    dataset = create_dataset(root, repo_id, args.image_size, args.fps)
    infos = records.setdefault("episodes", [])
    saved = len(infos)
    if dataset.meta.total_episodes != saved:
        raise RuntimeError(f"LeRobot has {dataset.meta.total_episodes} episodes but JSON has {saved}; repair before resume")
    logger = get_logger()

    def collect(seed, mode, placements=None, episode_config=None):
        nonlocal saved
        env = None
        try:
            env = make_env(args.task, seed, args.reset_wait_step, args.num_objects, episode_config)
            if placements:
                for name, xy in placements:
                    if name not in env.task.entities or not set_entity_xy(env, env.task.entities[name], xy):
                        return False
            initial = {"seed": seed, "sampling_method": mode, "entities": entity_state(env),
                       "episode_config": jsonable(env.save()), "episode_index": saved}
            result = run_expert(env, args.max_steps, EXTRA_STEPS)
            if result is None:
                return False
            obs, acts = result
            gi, wi, camera_names = camera_indices(env)
            instruction = env.task.get_instruction() or args.task
            robot_origin = np.asarray(env.get_robot_frame_position(), dtype=np.float64)
            add_episode(dataset, obs, acts, gi, wi, args.image_size, instruction, robot_origin)
            initial.update({"success": True, "num_frames": len(obs), "post_success_steps": EXTRA_STEPS,
                            "camera_names": camera_names, "instruction": instruction})
            infos.append(jsonable(initial)); saved += 1
            records["num_episodes"] = saved
            meta_path.write_text(json.dumps(jsonable(records), indent=2))
            print(f"saved episode {saved}/{args.random_episodes + args.uniform_episodes} seed={seed} mode={mode}")
            return True
        except Exception as exc:
            logger.warning("seed %s failed: %s", seed, exc)
            return False
        finally:
            if env is not None:
                env.close()

    random_saved = sum(e.get("sampling_method") == "seed_random" for e in infos)
    uniform_saved = sum(e.get("sampling_method") in {"uniform_grid", "uniform_perturbed"} for e in infos)
    target_total = args.random_episodes + args.uniform_episodes
    prior_seeds = [int(e["seed"]) for e in infos if "seed" in e]
    seed = max([args.seed_start - 1] + prior_seeds) + 1
    failed = 0
    while random_saved < args.random_episodes and saved < target_total:
        if collect(seed, "seed_random"):
            random_saved += 1; failed = 0
        else:
            failed += 1
            if failed >= args.max_consecutive_failures:
                raise RuntimeError(f"random collection failed {failed} consecutive times; aborting")
        seed += 1

    if saved < target_total and uniform_saved < args.uniform_episodes:
        probe = make_env(args.task, args.uniform_seed, args.reset_wait_step, args.num_objects)
        baseline_config = jsonable(probe.save())
        candidates = grid_candidates(probe, args.grid_size, args.grid_radius,
                                      args.uniform_episodes, args.uniform_seed, args.cell_jitter)
        probe.close()
        for candidate in candidates:
            if uniform_saved >= args.uniform_episodes or saved >= target_total:
                break
            ok = False
            for retry in range(args.uniform_retries):
                rng = np.random.RandomState(args.uniform_seed + seed * 997 + retry)
                placement = candidate if retry == 0 else [
                    (name, xy + rng.normal(0, args.grid_radius * args.perturb_scale, 2))
                    for name, xy in candidate]
                mode = "uniform_grid" if retry == 0 else "uniform_perturbed"
                if collect(seed, mode, placement, baseline_config):
                    uniform_saved += 1; ok = True; break
            if not ok:
                print("skip grid candidate after perturbation retries")
            seed += 1

    failed = 0
    while saved < target_total:
        if collect(seed, "seed_fallback"):
            failed = 0
        else:
            failed += 1
            if failed >= args.max_consecutive_failures:
                raise RuntimeError(f"fallback failed {failed} consecutive times; aborting")
        seed += 1
    dataset.finalize()
    print(f"done: {saved} successful episodes -> {root}")


if __name__ == "__main__":
    main()
