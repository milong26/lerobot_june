"""
RoboMME task configuration adapters.

Each adapter extracts and applies the episode-level randomized configuration
for a specific RoboMME task, based on the actual source code of that task.

Public API:
    extract_task_configuration(env, task) -> dict
    apply_task_configuration(env, task, config) -> bool
    get_uniform_spec(env, task) -> dict | None
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# JSON-safe conversion helpers
# ---------------------------------------------------------------------------

def _to_serializable(obj: Any) -> Any:
    """Recursively convert torch/numpy/Pose objects to JSON-safe types."""
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, torch.Tensor):
        val = obj.detach().cpu()
        if val.numel() == 1:
            return _py_scalar(val.item())
        return _to_serializable(val.numpy())
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return _py_scalar(obj.item())
        return [_to_serializable(x) for x in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    # Sapien Pose
    if hasattr(obj, "p") and hasattr(obj, "q"):
        return {
            "position": _to_serializable(obj.p),
            "quaternion": _to_serializable(obj.q),
        }
    return _py_scalar(obj)


def _py_scalar(v):
    """Convert numpy scalar to Python native."""
    if hasattr(v, "item"):
        return v.item()
    return v


def _actor_pose(actor) -> Optional[Dict]:
    """Extract position + quaternion from a Sapien actor."""
    if actor is None:
        return None
    pose = getattr(actor, "pose", None)
    if pose is None and hasattr(actor, "get_pose"):
        pose = actor.get_pose()
    if pose is None:
        return None
    return {
        "position": _to_serializable(pose.p),
        "quaternion": _to_serializable(pose.q),
    }


def _actor_velocity(actor) -> Optional[Dict]:
    """Extract linear and angular velocity if available."""
    if actor is None:
        return None
    try:
        lv = getattr(actor, "linear_velocity", None)
        av = getattr(actor, "angular_velocity", None)
        out = {}
        if lv is not None:
            out["linear_velocity"] = _to_serializable(lv)
        if av is not None:
            out["angular_velocity"] = _to_serializable(av)
        return out if out else None
    except Exception:
        return None


def _articulation_state(art) -> Optional[Dict]:
    """Extract qpos/qvel from an articulation."""
    if art is None:
        return None
    out = {"name": getattr(art, "name", str(art))}
    try:
        out["qpos"] = _to_serializable(art.qpos)
    except Exception:
        pass
    try:
        out["qvel"] = _to_serializable(art.qvel)
    except Exception:
        pass
    return out


def _scene_state_snapshot(env) -> Dict:
    """Full scene state via get_state_dict() or actor/articulation summary."""
    u = env.unwrapped
    # Prefer ManiSkill's built-in state dict
    try:
        sd = u.get_state_dict()
        return _to_serializable(sd)
    except Exception:
        pass

    summary: Dict[str, Any] = {"actors": [], "articulations": []}
    scene = getattr(u, "scene", None)
    if scene is not None:
        for i, actor in enumerate(getattr(scene, "actors", []) or []):
            name = getattr(actor, "name", None) or f"actor_{i}"
            entry = {"name": str(name), "pose": _actor_pose(actor)}
            vel = _actor_velocity(actor)
            if vel:
                entry["velocity"] = vel
            summary["actors"].append(entry)
    for art in getattr(u, "scene", None).__dict__.get("articulations", []) if getattr(u, "scene", None) else []:
        s = _articulation_state(art)
        if s:
            summary["articulations"].append(s)
    return summary


# ---------------------------------------------------------------------------
# Per-task adapters
# ---------------------------------------------------------------------------

def _adapter_pickxtimes(env, task: str) -> Dict:
    u = env.unwrapped
    movable = []
    targets = []
    articulations = []
    task_config = {}

    all_cubes = list(getattr(u, "all_cubes", []) or [])
    for i, cube in enumerate(all_cubes):
        entry = {
            "name": getattr(cube, "name", None) or f"cube_{i}",
            "pose": _actor_pose(cube),
        }
        vel = _actor_velocity(cube)
        if vel:
            entry["velocity"] = vel
        movable.append(entry)

    target_cube = getattr(u, "target_cube", None)
    target = getattr(u, "target", None)
    button = getattr(u, "button", None)

    if target_cube is not None:
        targets.append({
            "name": "target_cube",
            "pose": _actor_pose(target_cube),
        })
    if target is not None:
        targets.append({
            "name": "target",
            "pose": _actor_pose(target),
        })
    if button is not None:
        targets.append({
            "name": "button",
            "pose": _actor_pose(button),
        })

    task_config["target_color_name"] = getattr(u, "target_color_name", None)
    task_config["num_repeats"] = int(getattr(u, "num_repeats", 0))
    task_config["difficulty"] = getattr(u, "difficulty", None)
    task_config["num_cubes"] = len(all_cubes)
    task_config["red_cubes"] = len(getattr(u, "red_cubes", []) or [])
    task_config["blue_cubes"] = len(getattr(u, "blue_cubes", []) or [])
    task_config["green_cubes"] = len(getattr(u, "green_cubes", []) or [])

    return {
        "movable_objects": movable,
        "randomized_targets": targets,
        "articulations": articulations,
        "task_config": task_config,
        "scene_state": _scene_state_snapshot(env),
    }


def _adapter_binfill(env, task: str) -> Dict:
    u = env.unwrapped
    movable = []
    targets = []
    articulations = []
    task_config = {}

    for color_name in ["red_cubes", "blue_cubes", "green_cubes"]:
        cubes = list(getattr(u, color_name, []) or [])
        for i, cube in enumerate(cubes):
            entry = {
                "name": getattr(cube, "name", None) or f"{color_name}_{i}",
                "pose": _actor_pose(cube),
            }
            vel = _actor_velocity(cube)
            if vel:
                entry["velocity"] = vel
            movable.append(entry)

    board = getattr(u, "board_with_hole", None)
    button = getattr(u, "button", None)
    if board is not None:
        targets.append({"name": "board_with_hole", "pose": _actor_pose(board)})
    if button is not None:
        targets.append({"name": "button", "pose": _actor_pose(button)})

    task_config["difficulty"] = getattr(u, "difficulty", None)
    task_config["red_target"] = int(getattr(u, "red_cubes_target_number", 0))
    task_config["blue_target"] = int(getattr(u, "blue_cubes_target_number", 0))
    task_config["green_target"] = int(getattr(u, "green_cubes_target_number", 0))
    task_config["red_spawn"] = int(getattr(u, "red_cubes_spawn_number", 0))
    task_config["blue_spawn"] = int(getattr(u, "blue_cubes_spawn_number", 0))
    task_config["green_spawn"] = int(getattr(u, "green_cubes_spawn_number", 0))
    task_config["dynamic"] = bool(getattr(u, "dynamic", False))

    return {
        "movable_objects": movable,
        "randomized_targets": targets,
        "articulations": articulations,
        "task_config": task_config,
        "scene_state": _scene_state_snapshot(env),
    }


def _adapter_swingxtimes(env, task: str) -> Dict:
    u = env.unwrapped
    movable = []
    targets = []
    articulations = []
    task_config = {}

    all_cubes = list(getattr(u, "all_cubes", []) or [])
    for i, cube in enumerate(all_cubes):
        entry = {
            "name": getattr(cube, "name", None) or f"cube_{i}",
            "pose": _actor_pose(cube),
        }
        vel = _actor_velocity(cube)
        if vel:
            entry["velocity"] = vel
        movable.append(entry)

    target_cube = getattr(u, "target_cube", None)
    target_right = getattr(u, "target_right", None)
    target_left = getattr(u, "target_left", None)
    button = getattr(u, "button", None)

    if target_cube is not None:
        targets.append({"name": "target_cube", "pose": _actor_pose(target_cube)})
    if target_right is not None:
        targets.append({"name": "target_right", "pose": _actor_pose(target_right)})
    if target_left is not None:
        targets.append({"name": "target_left", "pose": _actor_pose(target_left)})
    if button is not None:
        targets.append({"name": "button", "pose": _actor_pose(button)})

    task_config["target_color_name"] = getattr(u, "target_color_name", None)
    task_config["num_repeats"] = int(getattr(u, "num_repeats", 0))
    task_config["difficulty"] = getattr(u, "difficulty", None)
    task_config["num_cubes"] = len(all_cubes)

    return {
        "movable_objects": movable,
        "randomized_targets": targets,
        "articulations": articulations,
        "task_config": task_config,
        "scene_state": _scene_state_snapshot(env),
    }


def _adapter_stopcube(env, task: str) -> Dict:
    u = env.unwrapped
    movable = []
    targets = []
    articulations = []
    task_config = {}

    cube = getattr(u, "cube", None)
    if cube is not None:
        entry = {"name": "cube", "pose": _actor_pose(cube)}
        vel = _actor_velocity(cube)
        if vel:
            entry["velocity"] = vel
        movable.append(entry)

    target = getattr(u, "target", None)
    button = getattr(u, "button", None)
    if target is not None:
        targets.append({"name": "target", "pose": _actor_pose(target)})
    if button is not None:
        targets.append({"name": "button", "pose": _actor_pose(button)})

    task_config["difficulty"] = getattr(u, "difficulty", None)
    task_config["stop_time"] = int(getattr(u, "stop_time", 0))
    task_config["move_interval"] = int(getattr(u, "move_interval", 0))
    task_config["start_pos_xy"] = _to_serializable(getattr(u, "start_pos_xy", None))
    task_config["end_pos_xy"] = _to_serializable(getattr(u, "end_pos_xy", None))

    return {
        "movable_objects": movable,
        "randomized_targets": targets,
        "articulations": articulations,
        "task_config": task_config,
        "scene_state": _scene_state_snapshot(env),
    }


def _adapter_videounmask(env, task: str) -> Dict:
    u = env.unwrapped
    movable = []
    targets = []
    articulations = []
    task_config = {}

    bins_list = list(getattr(u, "spawned_bins", []) or [])
    for i, b in enumerate(bins_list):
        entry = {"name": getattr(b, "name", None) or f"bin_{i}", "pose": _actor_pose(b)}
        targets.append(entry)

    # cubes under bins
    for color_name in ["red_cubes", "green_cubes", "blue_cubes"]:
        cubes = list(getattr(u, color_name, []) or [])
        for i, c in enumerate(cubes):
            entry = {
                "name": getattr(c, "name", None) or f"{color_name}_{i}",
                "pose": _actor_pose(c),
            }
            vel = _actor_velocity(c)
            if vel:
                entry["velocity"] = vel
            movable.append(entry)

    task_config["difficulty"] = getattr(u, "difficulty", None)
    task_config["num_bins"] = len(bins_list)
    task_config["num_pick"] = int(getattr(u, "num_repeats", 0))
    task_config["color_names"] = list(getattr(u, "color_names", []) or [])
    task_config["target_colors"] = list(getattr(u, "target_colors", []) or [])

    return {
        "movable_objects": movable,
        "randomized_targets": targets,
        "articulations": articulations,
        "task_config": task_config,
        "scene_state": _scene_state_snapshot(env),
    }


def _adapter_videounmaskswap(env, task: str) -> Dict:
    u = env.unwrapped
    movable = []
    targets = []
    articulations = []
    task_config = {}

    bins_list = list(getattr(u, "spawned_bins", []) or [])
    for i, b in enumerate(bins_list):
        entry = {"name": getattr(b, "name", None) or f"bin_{i}", "pose": _actor_pose(b)}
        targets.append(entry)

    for color_name in ["red_cubes", "green_cubes", "blue_cubes"]:
        cubes = list(getattr(u, color_name, []) or [])
        for i, c in enumerate(cubes):
            entry = {
                "name": getattr(c, "name", None) or f"{color_name}_{i}",
                "pose": _actor_pose(c),
            }
            vel = _actor_velocity(c)
            if vel:
                entry["velocity"] = vel
            movable.append(entry)

    task_config["difficulty"] = getattr(u, "difficulty", None)
    task_config["num_bins"] = len(bins_list)
    task_config["swap_times"] = int(getattr(u, "swap_times", 0))
    task_config["pick_times"] = int(getattr(u, "pick_times", 0))
    task_config["target_colors"] = list(getattr(u, "target_colors", []) or [])

    return {
        "movable_objects": movable,
        "randomized_targets": targets,
        "articulations": articulations,
        "task_config": task_config,
        "scene_state": _scene_state_snapshot(env),
    }


def _adapter_buttonunmask(env, task: str) -> Dict:
    u = env.unwrapped
    movable = []
    targets = []
    articulations = []
    task_config = {}

    bins_list = list(getattr(u, "spawned_bins", []) or [])
    for i, b in enumerate(bins_list):
        entry = {"name": getattr(b, "name", None) or f"bin_{i}", "pose": _actor_pose(b)}
        targets.append(entry)

    for color_name in ["red_cubes", "green_cubes", "blue_cubes"]:
        cubes = list(getattr(u, color_name, []) or [])
        for i, c in enumerate(cubes):
            entry = {
                "name": getattr(c, "name", None) or f"{color_name}_{i}",
                "pose": _actor_pose(c),
            }
            vel = _actor_velocity(c)
            if vel:
                entry["velocity"] = vel
            movable.append(entry)

    button_left = getattr(u, "button_left", None)
    button_right = getattr(u, "button_right", None)
    if button_left is not None:
        targets.append({"name": "button_left", "pose": _actor_pose(button_left)})
    if button_right is not None:
        targets.append({"name": "button_right", "pose": _actor_pose(button_right)})

    task_config["difficulty"] = getattr(u, "difficulty", None)
    task_config["num_bins"] = len(bins_list)
    task_config["num_repeats"] = int(getattr(u, "num_repeats", 0))
    task_config["target_colors"] = list(getattr(u, "target_colors", []) or [])

    return {
        "movable_objects": movable,
        "randomized_targets": targets,
        "articulations": articulations,
        "task_config": task_config,
        "scene_state": _scene_state_snapshot(env),
    }


def _adapter_buttonunmaskswap(env, task: str) -> Dict:
    u = env.unwrapped
    movable = []
    targets = []
    articulations = []
    task_config = {}

    bins_list = list(getattr(u, "spawned_bins", []) or [])
    for i, b in enumerate(bins_list):
        entry = {"name": getattr(b, "name", None) or f"bin_{i}", "pose": _actor_pose(b)}
        targets.append(entry)

    for color_name in ["red_cubes", "green_cubes", "blue_cubes"]:
        cubes = list(getattr(u, color_name, []) or [])
        for i, c in enumerate(cubes):
            entry = {
                "name": getattr(c, "name", None) or f"{color_name}_{i}",
                "pose": _actor_pose(c),
            }
            vel = _actor_velocity(c)
            if vel:
                entry["velocity"] = vel
            movable.append(entry)

    button_left = getattr(u, "button_left", None)
    button_right = getattr(u, "button_right", None)
    if button_left is not None:
        targets.append({"name": "button_left", "pose": _actor_pose(button_left)})
    if button_right is not None:
        targets.append({"name": "button_right", "pose": _actor_pose(button_right)})

    task_config["difficulty"] = getattr(u, "difficulty", None)
    task_config["num_bins"] = len(bins_list)
    task_config["swap_times"] = int(getattr(u, "swap_times", 0))
    task_config["pick_times"] = int(getattr(u, "pick_times", 0))
    task_config["target_colors"] = list(getattr(u, "target_colors", []) or [])

    return {
        "movable_objects": movable,
        "randomized_targets": targets,
        "articulations": articulations,
        "task_config": task_config,
        "scene_state": _scene_state_snapshot(env),
    }


def _adapter_pickhighlight(env, task: str) -> Dict:
    u = env.unwrapped
    movable = []
    targets = []
    articulations = []
    task_config = {}

    all_cubes = list(getattr(u, "all_cubes", []) or [])
    for i, cube in enumerate(all_cubes):
        entry = {
            "name": getattr(cube, "name", None) or f"cube_{i}",
            "pose": _actor_pose(cube),
        }
        vel = _actor_velocity(cube)
        if vel:
            entry["velocity"] = vel
        movable.append(entry)

    target_cubes = list(getattr(u, "target_cubes", []) or [])
    for i, tc in enumerate(target_cubes):
        targets.append({"name": f"target_cube_{i}", "pose": _actor_pose(tc)})

    task_config["difficulty"] = getattr(u, "difficulty", None)
    task_config["num_cubes"] = len(all_cubes)
    task_config["num_targets"] = int(getattr(u, "num_target_cubes", len(target_cubes)))

    return {
        "movable_objects": movable,
        "randomized_targets": targets,
        "articulations": articulations,
        "task_config": task_config,
        "scene_state": _scene_state_snapshot(env),
    }


def _adapter_videorepick(env, task: str) -> Dict:
    u = env.unwrapped
    movable = []
    targets = []
    articulations = []
    task_config = {}

    all_cubes = list(getattr(u, "all_cubes", []) or [])
    for i, cube in enumerate(all_cubes):
        entry = {
            "name": getattr(cube, "name", None) or f"cube_{i}",
            "pose": _actor_pose(cube),
        }
        vel = _actor_velocity(cube)
        if vel:
            entry["velocity"] = vel
        movable.append(entry)

    target_cube = getattr(u, "target_cube", None)
    target = getattr(u, "target", None)
    button = getattr(u, "button", None)
    if target_cube is not None:
        targets.append({"name": "target_cube", "pose": _actor_pose(target_cube)})
    if target is not None:
        targets.append({"name": "target", "pose": _actor_pose(target)})
    if button is not None:
        targets.append({"name": "button", "pose": _actor_pose(button)})

    task_config["target_color_name"] = getattr(u, "target_color_name", None)
    task_config["difficulty"] = getattr(u, "difficulty", None)
    task_config["num_cubes"] = len(all_cubes)

    return {
        "movable_objects": movable,
        "randomized_targets": targets,
        "articulations": articulations,
        "task_config": task_config,
        "scene_state": _scene_state_snapshot(env),
    }


def _adapter_videoplacebutton(env, task: str) -> Dict:
    u = env.unwrapped
    movable = []
    targets = []
    articulations = []
    task_config = {}

    all_cubes = list(getattr(u, "all_cubes", []) or [])
    for i, cube in enumerate(all_cubes):
        entry = {
            "name": getattr(cube, "name", None) or f"cube_{i}",
            "pose": _actor_pose(cube),
        }
        vel = _actor_velocity(cube)
        if vel:
            entry["velocity"] = vel
        movable.append(entry)

    target_cube = getattr(u, "target_cube", None)
    target_target = getattr(u, "target_target", None)
    targets_not_true = list(getattr(u, "targets_not_true", []) or [])
    button = getattr(u, "button", None)

    if target_cube is not None:
        targets.append({"name": "target_cube", "pose": _actor_pose(target_cube)})
    if target_target is not None:
        targets.append({"name": "target_target", "pose": _actor_pose(target_target)})
    for i, t in enumerate(targets_not_true):
        targets.append({"name": f"target_false_{i}", "pose": _actor_pose(t)})
    if button is not None:
        targets.append({"name": "button", "pose": _actor_pose(button)})

    task_config["target_color_name"] = getattr(u, "target_color_name", None)
    task_config["difficulty"] = getattr(u, "difficulty", None)
    task_config["num_cubes"] = len(all_cubes)

    return {
        "movable_objects": movable,
        "randomized_targets": targets,
        "articulations": articulations,
        "task_config": task_config,
        "scene_state": _scene_state_snapshot(env),
    }


def _adapter_videoplaceorder(env, task: str) -> Dict:
    u = env.unwrapped
    movable = []
    targets = []
    articulations = []
    task_config = {}

    all_cubes = list(getattr(u, "all_cubes", []) or [])
    for i, cube in enumerate(all_cubes):
        entry = {
            "name": getattr(cube, "name", None) or f"cube_{i}",
            "pose": _actor_pose(cube),
        }
        vel = _actor_velocity(cube)
        if vel:
            entry["velocity"] = vel
        movable.append(entry)

    target_cubes = list(getattr(u, "target_cubes", []) or [])
    for i, tc in enumerate(target_cubes):
        targets.append({"name": f"target_cube_{i}", "pose": _actor_pose(tc)})

    target_sites = list(getattr(u, "target_sites", []) or [])
    for i, ts in enumerate(target_sites):
        targets.append({"name": f"target_site_{i}", "pose": _actor_pose(ts)})

    button = getattr(u, "button", None)
    if button is not None:
        targets.append({"name": "button", "pose": _actor_pose(button)})

    task_config["difficulty"] = getattr(u, "difficulty", None)
    task_config["num_cubes"] = len(all_cubes)
    task_config["order"] = [getattr(c, "name", str(c)) for c in target_cubes]

    return {
        "movable_objects": movable,
        "randomized_targets": targets,
        "articulations": articulations,
        "task_config": task_config,
        "scene_state": _scene_state_snapshot(env),
    }


def _adapter_movecube(env, task: str) -> Dict:
    u = env.unwrapped
    movable = []
    targets = []
    articulations = []
    task_config = {}

    cube = getattr(u, "cube", None)
    if cube is not None:
        entry = {"name": "cube", "pose": _actor_pose(cube)}
        vel = _actor_velocity(cube)
        if vel:
            entry["velocity"] = vel
        movable.append(entry)

    cube_2 = getattr(u, "cube_2", None)
    if cube_2 is not None:
        entry = {"name": "cube_2", "pose": _actor_pose(cube_2)}
        vel = _actor_velocity(cube_2)
        if vel:
            entry["velocity"] = vel
        movable.append(entry)

    goal_site = getattr(u, "goal_site", None)
    goal_site_2 = getattr(u, "goal_site_2", None)
    if goal_site is not None:
        targets.append({"name": "goal_site", "pose": _actor_pose(goal_site)})
    if goal_site_2 is not None:
        targets.append({"name": "goal_site_2", "pose": _actor_pose(goal_site_2)})

    # Pegs are articulation-like objects with DOF
    pegs = list(getattr(u, "pegs", []) or [])
    for i, peg in enumerate(pegs):
        entry = {"name": getattr(peg, "name", None) or f"peg_{i}", "pose": _actor_pose(peg)}
        try:
            entry["qpos"] = _to_serializable(peg.qpos)
        except Exception:
            pass
        articulations.append(entry)

    task_config["difficulty"] = getattr(u, "difficulty", None)
    task_config["way"] = getattr(u, "way", None)
    task_config["direction"] = int(getattr(u, "direction", 1))
    task_config["direction2"] = int(getattr(u, "direction2", 1))
    task_config["obj_flag"] = int(getattr(u, "obj_flag", 1))
    task_config["length"] = float(getattr(u, "length", 0.1))
    task_config["radius"] = float(getattr(u, "radius", 0.01))

    return {
        "movable_objects": movable,
        "randomized_targets": targets,
        "articulations": articulations,
        "task_config": task_config,
        "scene_state": _scene_state_snapshot(env),
    }


def _adapter_insertpeg(env, task: str) -> Dict:
    u = env.unwrapped
    movable = []
    targets = []
    articulations = []
    task_config = {}

    box = getattr(u, "box", None)
    if box is not None:
        targets.append({"name": "box", "pose": _actor_pose(box)})

    pegs = list(getattr(u, "pegs", []) or [])
    for i, peg in enumerate(pegs):
        entry = {"name": getattr(peg, "name", None) or f"peg_{i}", "pose": _actor_pose(peg)}
        try:
            entry["qpos"] = _to_serializable(peg.qpos)
        except Exception:
            pass
        articulations.append(entry)

    grasp_target = getattr(u, "grasp_target", None)
    insert_target = getattr(u, "insert_target", None)
    if grasp_target is not None:
        targets.append({"name": "grasp_target", "pose": _actor_pose(grasp_target)})
    if insert_target is not None:
        targets.append({"name": "insert_target", "pose": _actor_pose(insert_target)})

    task_config["difficulty"] = getattr(u, "difficulty", None)
    task_config["obj_flag"] = int(getattr(u, "obj_flag", 1))
    task_config["direction"] = int(getattr(u, "direction", 1))
    task_config["grasp_target_distance"] = getattr(u, "grasp_target_distance", None)
    task_config["insert_way"] = getattr(u, "insert_way", None)
    task_config["length"] = float(getattr(u, "length", 0.1))
    task_config["radius"] = float(getattr(u, "radius", 0.01))

    return {
        "movable_objects": movable,
        "randomized_targets": targets,
        "articulations": articulations,
        "task_config": task_config,
        "scene_state": _scene_state_snapshot(env),
    }


def _adapter_patternlock(env, task: str) -> Dict:
    u = env.unwrapped
    movable = []
    targets = []
    articulations = []
    task_config = {}

    buttons_grid = list(getattr(u, "buttons_grid", []) or [])
    for i, btn in enumerate(buttons_grid):
        targets.append({"name": f"button_{i}", "pose": _actor_pose(btn)})

    selected = list(getattr(u, "selected_buttons", []) or [])
    selected_names = []
    for i, s in enumerate(selected):
        targets.append({"name": f"selected_{i}", "pose": _actor_pose(s)})
        selected_names.append(getattr(s, "name", str(s)))

    task_config["difficulty"] = getattr(u, "difficulty", None)
    task_config["grid_size"] = int(getattr(u, "grid_size", 3))
    task_config["selected_button_names"] = selected_names
    task_config["num_selected"] = len(selected)

    return {
        "movable_objects": movable,
        "randomized_targets": targets,
        "articulations": articulations,
        "task_config": task_config,
        "scene_state": _scene_state_snapshot(env),
    }


def _adapter_routestick(env, task: str) -> Dict:
    u = env.unwrapped
    movable = []
    targets = []
    articulations = []
    task_config = {}

    buttons_grid = list(getattr(u, "buttons_grid", []) or [])
    for i, btn in enumerate(buttons_grid):
        targets.append({"name": f"button_{i}", "pose": _actor_pose(btn)})

    selected = list(getattr(u, "selected_buttons", []) or [])
    selected_names = []
    for i, s in enumerate(selected):
        targets.append({"name": f"waypoint_{i}", "pose": _actor_pose(s)})
        selected_names.append(getattr(s, "name", str(s)))

    task_config["difficulty"] = getattr(u, "difficulty", None)
    task_config["grid_size"] = int(getattr(u, "grid_size", 3))
    task_config["selected_button_names"] = selected_names
    task_config["num_selected"] = len(selected)
    task_config["direction"] = getattr(u, "direction", None)

    return {
        "movable_objects": movable,
        "randomized_targets": targets,
        "articulations": articulations,
        "task_config": task_config,
        "scene_state": _scene_state_snapshot(env),
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_CONFIG_ADAPTERS: Dict[str, callable] = {
    "PickXtimes": _adapter_pickxtimes,
    "BinFill": _adapter_binfill,
    "SwingXtimes": _adapter_swingxtimes,
    "StopCube": _adapter_stopcube,
    "VideoUnmask": _adapter_videounmask,
    "VideoUnmaskSwap": _adapter_videounmaskswap,
    "ButtonUnmask": _adapter_buttonunmask,
    "ButtonUnmaskSwap": _adapter_buttonunmaskswap,
    "PickHighlight": _adapter_pickhighlight,
    "VideoRepick": _adapter_videorepick,
    "VideoPlaceButton": _adapter_videoplacebutton,
    "VideoPlaceOrder": _adapter_videoplaceorder,
    "MoveCube": _adapter_movecube,
    "InsertPeg": _adapter_insertpeg,
    "PatternLock": _adapter_patternlock,
    "RouteStick": _adapter_routestick,
}


def extract_task_configuration(env, task: str) -> Dict:
    """Extract the full episode-level configuration from a reset RoboMME env."""
    adapter = TASK_CONFIG_ADAPTERS.get(task)
    if adapter is None:
        raise NotImplementedError(f"No task config adapter for task={task}")
    return adapter(env, task)


def apply_task_configuration(env, task: str, config: Dict) -> bool:
    """
    Inject a previously extracted configuration into a reset RoboMME env.

    NOTE: Most RoboMME tasks randomize positions during _load_scene() which is
    called during env.reset(). Directly re-setting actor poses after reset is
    the only practical approach without re-writing the task classes. This
    function sets poses for movable objects and targets that the adapter
    recorded.

    Returns True if at least one pose was successfully applied.
    """
    adapter = TASK_CONFIG_ADAPTERS.get(task)
    if adapter is None:
        raise NotImplementedError(f"No task config adapter for task={task}")

    applied = 0

    # Apply movable object poses
    for obj_info in config.get("movable_objects", []):
        name = obj_info.get("name")
        pose_data = obj_info.get("pose")
        if name is None or pose_data is None:
            continue
        actor = _find_actor_by_name(env, name)
        if actor is None:
            continue
        if _set_actor_pose(actor, pose_data):
            applied += 1

    # Apply randomized target poses
    for tgt_info in config.get("randomized_targets", []):
        name = tgt_info.get("name")
        pose_data = tgt_info.get("pose")
        if name is None or pose_data is None:
            continue
        actor = _find_actor_by_name(env, name)
        if actor is None:
            continue
        if _set_actor_pose(actor, pose_data):
            applied += 1

    return applied > 0


def _find_actor_by_name(env, name: str):
    """Find a Sapien actor by name in the scene."""
    u = env.unwrapped
    scene = getattr(u, "scene", None)
    if scene is None:
        return None
    for actor in getattr(scene, "actors", []) or []:
        if getattr(actor, "name", None) == name:
            return actor
    # Also check task-level attributes
    if hasattr(u, name):
        obj = getattr(u, name)
        if hasattr(obj, "pose"):
            return obj
    return None


def _set_actor_pose(actor, pose_data: Dict) -> bool:
    """Set an actor's pose from serialized position + quaternion."""
    try:
        import sapien
        pos = np.array(pose_data["position"], dtype=np.float32)
        quat = np.array(pose_data["quaternion"], dtype=np.float32)
        actor.set_pose(sapien.Pose(p=pos, q=quat))
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Uniform sampling specification per task
# ---------------------------------------------------------------------------

def get_uniform_spec(env, task: str) -> Optional[Dict]:
    """
    Return the uniform sampling specification for a task.

    Returns a dict with:
        "objects": list of {"name": str, "region_center": [x,y], "region_half_size": float}
        "num_objects": int (actual count from current episode)
        "grid_resolution": {"x": int, "y": int}

    Returns None if the task does not support uniform spatial sampling.
    """
    u = env.unwrapped

    if task == "PickXtimes":
        all_cubes = list(getattr(u, "all_cubes", []) or [])
        if not all_cubes:
            return None
        return {
            "objects": [
                {"name": c.name if hasattr(c, "name") else f"cube_{i}",
                 "region_center": [-0.1, 0.0],
                 "region_half_size": 0.2}
                for i, c in enumerate(all_cubes)
            ],
            "num_objects": len(all_cubes),
            "grid_resolution": {"x": 5, "y": 5},
        }

    if task == "BinFill":
        all_cubes = (
            list(getattr(u, "red_cubes", []) or [])
            + list(getattr(u, "blue_cubes", []) or [])
            + list(getattr(u, "green_cubes", []) or [])
        )
        if not all_cubes:
            return None
        return {
            "objects": [
                {"name": c.name if hasattr(c, "name") else f"cube_{i}",
                 "region_center": [0.0, 0.0],
                 "region_half_size": 0.2}
                for i, c in enumerate(all_cubes)
            ],
            "num_objects": len(all_cubes),
            "grid_resolution": {"x": 4, "y": 4},
        }

    if task == "SwingXtimes":
        all_cubes = list(getattr(u, "all_cubes", []) or [])
        if not all_cubes:
            return None
        return {
            "objects": [
                {"name": c.name if hasattr(c, "name") else f"cube_{i}",
                 "region_center": [-0.1, 0.0],
                 "region_half_size": 0.2}
                for i, c in enumerate(all_cubes)
            ],
            "num_objects": len(all_cubes),
            "grid_resolution": {"x": 5, "y": 5},
        }

    if task == "StopCube":
        return {
            "objects": [
                {"name": "cube", "region_center": [-0.3, -0.3], "region_half_size": 0.1},
            ],
            "num_objects": 1,
            "grid_resolution": {"x": 4, "y": 4},
        }

    if task == "VideoUnmask":
        bins = list(getattr(u, "spawned_bins", []) or [])
        if not bins:
            return None
        return {
            "objects": [
                {"name": b.name if hasattr(b, "name") else f"bin_{i}",
                 "region_center": [0.0, 0.0],
                 "region_half_size": 0.2}
                for i, b in enumerate(bins)
            ],
            "num_objects": len(bins),
            "grid_resolution": {"x": 4, "y": 4},
        }

    if task == "VideoUnmaskSwap":
        bins = list(getattr(u, "spawned_bins", []) or [])
        if not bins:
            return None
        return {
            "objects": [
                {"name": b.name if hasattr(b, "name") else f"bin_{i}",
                 "region_center": [0.0, 0.0],
                 "region_half_size": 0.2}
                for i, b in enumerate(bins)
            ],
            "num_objects": len(bins),
            "grid_resolution": {"x": 4, "y": 4},
        }

    if task == "ButtonUnmask":
        bins = list(getattr(u, "spawned_bins", []) or [])
        if not bins:
            return None
        return {
            "objects": [
                {"name": b.name if hasattr(b, "name") else f"bin_{i}",
                 "region_center": [0.0, 0.0],
                 "region_half_size": 0.2}
                for i, b in enumerate(bins)
            ],
            "num_objects": len(bins),
            "grid_resolution": {"x": 4, "y": 4},
        }

    if task == "ButtonUnmaskSwap":
        bins = list(getattr(u, "spawned_bins", []) or [])
        if not bins:
            return None
        return {
            "objects": [
                {"name": b.name if hasattr(b, "name") else f"bin_{i}",
                 "region_center": [0.0, 0.0],
                 "region_half_size": 0.2}
                for i, b in enumerate(bins)
            ],
            "num_objects": len(bins),
            "grid_resolution": {"x": 4, "y": 4},
        }

    if task == "PickHighlight":
        all_cubes = list(getattr(u, "all_cubes", []) or [])
        if not all_cubes:
            return None
        return {
            "objects": [
                {"name": c.name if hasattr(c, "name") else f"cube_{i}",
                 "region_center": [-0.1, 0.0],
                 "region_half_size": 0.2}
                for i, c in enumerate(all_cubes)
            ],
            "num_objects": len(all_cubes),
            "grid_resolution": {"x": 4, "y": 4},
        }

    if task == "VideoRepick":
        all_cubes = list(getattr(u, "all_cubes", []) or [])
        if not all_cubes:
            return None
        return {
            "objects": [
                {"name": c.name if hasattr(c, "name") else f"cube_{i}",
                 "region_center": [-0.1, 0.0],
                 "region_half_size": 0.2}
                for i, c in enumerate(all_cubes)
            ],
            "num_objects": len(all_cubes),
            "grid_resolution": {"x": 4, "y": 4},
        }

    if task == "VideoPlaceButton":
        all_cubes = list(getattr(u, "all_cubes", []) or [])
        if not all_cubes:
            return None
        return {
            "objects": [
                {"name": c.name if hasattr(c, "name") else f"cube_{i}",
                 "region_center": [-0.1, 0.0],
                 "region_half_size": 0.2}
                for i, c in enumerate(all_cubes)
            ],
            "num_objects": len(all_cubes),
            "grid_resolution": {"x": 4, "y": 4},
        }

    if task == "VideoPlaceOrder":
        all_cubes = list(getattr(u, "all_cubes", []) or [])
        if not all_cubes:
            return None
        return {
            "objects": [
                {"name": c.name if hasattr(c, "name") else f"cube_{i}",
                 "region_center": [-0.1, 0.0],
                 "region_half_size": 0.2}
                for i, c in enumerate(all_cubes)
            ],
            "num_objects": len(all_cubes),
            "grid_resolution": {"x": 4, "y": 4},
        }

    if task == "MoveCube":
        cube = getattr(u, "cube", None)
        if cube is None:
            return None
        return {
            "objects": [
                {"name": "cube", "region_center": [0.0, 0.0], "region_half_size": 0.15},
            ],
            "num_objects": 1,
            "grid_resolution": {"x": 4, "y": 4},
        }

    if task == "InsertPeg":
        # Peg positions are randomized in a region; we can sample peg poses
        pegs = list(getattr(u, "pegs", []) or [])
        if not pegs:
            return None
        return {
            "objects": [
                {"name": p.name if hasattr(p, "name") else f"peg_{i}",
                 "region_center": [0.0, -0.1],
                 "region_half_size": 0.2}
                for i, p in enumerate(pegs)
            ],
            "num_objects": len(pegs),
            "grid_resolution": {"x": 3, "y": 3},
        }

    # PatternLock and RouteStick: button grid positions are fixed per difficulty,
    # no meaningful spatial uniform sampling for these tasks.
    return None


def generate_uniform_configurations(
    uniform_spec: Dict,
    num_episodes: int,
    rng: np.random.RandomState,
    max_perturb_retries: int = 4,
) -> list:
    """
    Generate uniform configurations using stratified Cartesian sampling.

    If the full Cartesian product is manageable, enumerate and shuffle.
    If too large, use a deterministic balanced subset that covers each
    object's grid cells as uniformly as possible.
    """
    objects = uniform_spec["objects"]
    num_objects = uniform_spec["num_objects"]
    res = uniform_spec["grid_resolution"]
    nx, ny = res["x"], res["y"]

    # Build grid points for each object
    grids = []
    for obj in objects:
        cx, cy = obj["region_center"]
        hs = obj["region_half_size"]
        xs = np.linspace(cx - hs, cx + hs, nx)
        ys = np.linspace(cy - hs, cy + hs, ny)
        pts = []
        for x in xs:
            for y in ys:
                pts.append([float(x), float(y)])
        grids.append(pts)

    # If only one object, simple grid
    if len(grids) == 1:
        pts = grids[0]
        rng.shuffle(pts)
        configs = []
        for pt in pts[:num_episodes]:
            configs.append({objects[0]["name"]: pt})
        return configs

    # Multi-object: estimate Cartesian product size
    total_combos = 1
    for g in grids:
        total_combos *= len(g)

    if total_combos <= num_episodes * 10:
        # Full Cartesian product is manageable
        import itertools
        all_configs = list(itertools.product(*grids))
        rng.shuffle(all_configs)
        configs = []
        for combo in all_configs[:num_episodes]:
            cfg = {}
            for i, obj in enumerate(objects):
                cfg[obj["name"]] = list(combo[i])
            configs.append(cfg)
        return configs

    # Stratified sampling: ensure each grid cell gets roughly equal coverage
    configs = []
    max_configs = num_episodes * 2  # safety limit
    attempts = 0
    cell_usage = {}  # (obj_idx, cell_x, cell_y) -> count

    while len(configs) < num_episodes and attempts < max_configs:
        cfg = {}
        valid = True
        for i, obj in enumerate(objects):
            g = grids[i]
            # Choose a cell that has been used least
            min_count = float("inf")
            best_pt = None
            candidates = []
            for pt in g:
                cx_idx = min(nx - 1, max(0, int((pt[0] - (obj["region_center"][0] - obj["region_half_size"])) / (2 * obj["region_half_size"] / (nx - 1))))) if nx > 1 else 0
                cy_idx = min(ny - 1, max(0, int((pt[1] - (obj["region_center"][1] - obj["region_half_size"])) / (2 * obj["region_half_size"] / (ny - 1))))) if ny > 1 else 0
                key = (i, cx_idx, cy_idx)
                count = cell_usage.get(key, 0)
                if count < min_count:
                    min_count = count
                    candidates = [pt]
                elif count == min_count:
                    candidates.append(pt)
            chosen = candidates[rng.randint(0, len(candidates))]
            cfg[obj["name"]] = list(chosen)

        # Check collision / validity constraints (simplified: min distance)
        positions = [np.array(v) for v in cfg.values()]
        min_dist = 0.08  # minimum distance between objects
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                    valid = False
                    break
            if not valid:
                break

        if valid:
            configs.append(cfg)
            for i, obj in enumerate(objects):
                pt = cfg[obj["name"]]
                cx_idx = min(nx - 1, max(0, int((pt[0] - (obj["region_center"][0] - obj["region_half_size"])) / (2 * obj["region_half_size"] / (nx - 1))))) if nx > 1 else 0
                cy_idx = min(ny - 1, max(0, int((pt[1] - (obj["region_center"][1] - obj["region_half_size"])) / (2 * obj["region_half_size"] / (ny - 1))))) if ny > 1 else 0
                key = (i, cx_idx, cy_idx)
                cell_usage[key] = cell_usage.get(key, 0) + 1

        attempts += 1

    return configs