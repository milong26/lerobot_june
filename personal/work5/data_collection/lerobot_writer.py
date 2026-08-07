"""LeRobot dataset writer utilities."""

import json
import numpy as np
from pathlib import Path

import os
os.environ["HF_LEROBOT_HOME"] = str(Path(__file__).parent.parent.parent.parent / "personal" / "work5" / "datasets")

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load task descriptions from LeRobot config
_LEROBOT_CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "src" / "lerobot" / "envs" / "metaworld_config.json"
with open(_LEROBOT_CONFIG_PATH) as _f:
    _METAWORLD_CONFIG = json.load(_f)
_TASK_DESCRIPTIONS = _METAWORLD_CONFIG.get("TASK_DESCRIPTIONS", {})


def get_task_description(task_name):
    """Get task description from LeRobot metaworld config."""
    return _TASK_DESCRIPTIONS.get(task_name, task_name)


def create_dataset(repo_id, output_dir, fps=80, image_size=224):
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


def add_episode_to_dataset(dataset, frames, task="pick-place-v3"):
    task_desc = get_task_description(task)
    for frame in frames:
        frame["task"] = task_desc
        dataset.add_frame(frame)
    dataset.save_episode()


def save_dataset_metadata(output_dir, episode_infos, task_name, strategy_name):
    metadata = {
        "task": task_name,
        "strategy": strategy_name,
        "num_episodes": len(episode_infos),
        "episodes": [],
    }
    for i, info in enumerate(episode_infos):
        metadata["episodes"].append({
            "episode_index": i,
            "obj_init_pos": info.get("obj_init_pos", []).tolist() if hasattr(info.get("obj_init_pos", []), 'tolist') else info.get("obj_init_pos", []),
            "goal_pose": info.get("goal_pose", []).tolist() if hasattr(info.get("goal_pose", []), 'tolist') else info.get("goal_pose", []),
            "success": bool(info.get("success", False)),
            "num_frames": info.get("num_frames", 0),
        })
    metadata_file = Path(output_dir) / "episode_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)


def finalize_dataset(dataset, output_dir, episode_infos, task_name, strategy_name):
    dataset.finalize()
    save_dataset_metadata(output_dir, episode_infos, task_name, strategy_name)
    print(f"Dataset finalized at {output_dir} with {len(episode_infos)} episodes")