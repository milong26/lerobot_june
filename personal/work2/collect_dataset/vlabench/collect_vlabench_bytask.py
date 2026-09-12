#!/usr/bin/env python3
"""Collect VLABench expert demonstrations into one LeRobot dataset.

Default task is ``select_fruit``.  The collector has two phases:
1. seeded random resets (300 successful episodes);
2. spatial grid/combinations of movable entities (100 successful episodes).

Only successful episodes are saved.  The initial simulator configuration of
every saved episode is written to episode_initial_states.json.
"""
import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

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


def make_env(task, seed, reset_wait_step):
    # VLABench uses numpy/random during task construction; seed both so the
    # seed phase is reproducible even though load_env has no seed argument.
    np.random.seed(seed)
    random.seed(seed)
    return load_env(task, robot="franka", random_init=True,
                    reset_wait_step=reset_wait_step, run_mode="train")


def camera_indices(env):
    names = []
    for i in range(env.physics.model.ncam):
        try:
            names.append(env.physics.model.camera(i).name)
        except Exception:
            names.append("")
    global_i = next((i for i, n in enumerate(names) if n == "forward"),
                    min(2, len(names) - 1))
    wrist_i = next((i for i, n in enumerate(names) if "wrist" in n),
                   len(names) - 1)
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
    # Entities with is_grasped are the task's manipulable objects.  Containers
    # and fixed scenery are intentionally excluded from spatial randomization.
    out = []
    for name, entity in env.task.entities.items():
        if hasattr(entity, "is_grasped") and hasattr(entity, "mjcf_model"):
            out.append((name, entity))
    return out


def set_entity_xy(env, entity, xy):
    try:
        body = entity.mjcf_model.worldbody
        pos = np.asarray(body.pos, dtype=np.float64).copy()
        pos[:2] = xy
        body.pos = pos
        env.step()
        return True
    except Exception:
        return False


def grid_candidates(env, grid_size, xy_radius, max_candidates, seed):
    entities = movable_entities(env)
    if not entities:
        return []
    rng = np.random.RandomState(seed)
    axes = np.linspace(-xy_radius, xy_radius, grid_size)
    offsets = np.array([(x, y) for x in axes for y in axes], dtype=np.float64)
    rng.shuffle(offsets)
    bases = []
    for _, ent in entities:
        try:
            bases.append(np.asarray(ent.mjcf_model.worldbody.pos, dtype=np.float64).copy())
        except Exception:
            bases.append(None)
    candidates = []
    # Combination: each movable object receives an independently selected grid
    # offset.  Limit the Cartesian product to keep planning tractable.
    for _ in range(max_candidates * 4):
        choice = rng.randint(0, len(offsets), size=len(entities))
        candidates.append([(entities[i][0], bases[i][:2] + offsets[choice[i]])
                           for i in range(len(entities)) if bases[i] is not None])
        if len(candidates) >= max_candidates:
            break
    return candidates


def run_expert(env, task, extra_steps):
    observations, actions = [], []
    success = False
    for skill in env.get_expert_skill_sequence() or []:
        obs, waypoints, stage_ok, task_ok = skill(env)
        observations.extend(obs or [])
        actions.extend(waypoints or [])
        if task_ok:
            success = True
            break
        if not stage_ok:
            break
    if not success or not observations or len(observations) != len(actions):
        return None
    # The expert ends on the success transition. Hold the last command for 10
    # simulator steps and record those frames as requested.
    for _ in range(extra_steps):
        last = np.asarray(actions[-1])
        ok, qpos = env.robot.get_qpos_from_ee_pos(env.physics, last[:3],
                                                   env.robot.get_end_effector_quat(env.physics))
        if not ok:
            break
        ts = env.step(np.concatenate([qpos, last[-2:]]))
        observations.append(env.get_observation())
        actions.append(last.copy())
        if ts.last():
            break
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
            "observation.state": {"dtype": "float32", "shape": (16,)},
            "action": {"dtype": "float32", "shape": (9,)},
        })


def add_episode(dataset, observations, actions, gi, wi, size, instruction):
    from PIL import Image
    for obs, action in zip(observations, actions):
        rgb = np.asarray(obs["rgb"])
        def image(i):
            return np.asarray(Image.fromarray(rgb[i]).resize((size, size)))
        state = np.asarray(obs["q_state"], dtype=np.float32).reshape(-1)
        state = np.pad(state, (0, max(0, 16 - len(state))))[:16]
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        act = np.pad(act, (0, max(0, 9 - len(act))))[:9]
        dataset.add_frame({"observation.images.global": image(gi),
                           "observation.images.wrist": image(wi),
                           "observation.state": state, "action": act}, task=instruction)
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
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--grid-size", type=int, default=5)
    p.add_argument("--grid-radius", type=float, default=0.08)
    p.add_argument("--uniform-retries", type=int, default=5)
    p.add_argument("--reset-wait-step", type=int, default=10)
    args = p.parse_args()
    root = Path(args.output_root) / args.task
    repo_id = args.repo_id or f"vlabench_{args.task}"
    meta_path = root / "episode_initial_states.json"
    records = json.loads(meta_path.read_text()) if meta_path.exists() else {"task": args.task, "episodes": []}
    dataset = create_dataset(root, repo_id, args.image_size, args.fps)
    logger = get_logger()
    saved = len(records.get("episodes", []))
    infos = records.setdefault("episodes", [])

    def collect(seed, mode, placements=None):
        nonlocal saved
        env = None
        try:
            env = make_env(args.task, seed, args.reset_wait_step)
            if placements:
                for name, xy in placements:
                    set_entity_xy(env, env.task.entities[name], xy)
            initial = {"seed": seed, "sampling_method": mode,
                       "entities": entity_state(env),
                       "episode_index": saved}
            result = run_expert(env, args.task, EXTRA_STEPS)
            if result is None:
                return False
            obs, acts = result
            gi, wi, camera_names = camera_indices(env)
            instruction = env.task.get_instruction()
            add_episode(dataset, obs, acts, gi, wi, args.image_size, instruction)
            initial.update({"success": True, "num_frames": len(obs), "camera_names": camera_names,
                            "instruction": instruction})
            infos.append(jsonable(initial)); saved += 1
            meta_path.write_text(json.dumps(jsonable(records), indent=2))
            print(f"saved episode {saved}/400 seed={seed} mode={mode}")
            return True
        except Exception as exc:
            logger.warning("seed %s failed: %s", seed, exc)
            return False
        finally:
            if env is not None:
                env.close()

    seed = args.seed_start
    while saved < args.random_episodes:
        collect(seed, "seed_random"); seed += 1
    probe = make_env(args.task, seed, args.reset_wait_step)
    candidates = grid_candidates(probe, args.grid_size, args.grid_radius,
                                  args.uniform_episodes * 3, seed)
    probe.close()
    uniform_saved = 0
    for candidate in candidates:
        if uniform_saved >= args.uniform_episodes: break
        ok = False
        for retry in range(args.uniform_retries):
            perturbed = [(n, xy + np.random.RandomState(seed + retry).normal(0, args.grid_radius * .15, 2))
                         for n, xy in candidate] if retry else candidate
            if collect(seed, "uniform_grid" if retry == 0 else "uniform_perturbed", perturbed):
                uniform_saved += 1; ok = True; break
        seed += 1
        if not ok: print("skip invalid grid candidate")
    while saved < args.random_episodes + args.uniform_episodes:
        if collect(seed, "seed_fallback"): pass
        seed += 1
    dataset.consolidate(run_compute_stats=True)
    print(f"done: {saved} successful episodes -> {root}")


if __name__ == "__main__":
    main()
