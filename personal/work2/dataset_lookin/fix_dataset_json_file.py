"""
补全数据集 episode_initial_states.json 文件

从 LeRobot 数据集中读取每个 episode 的第一帧，提取 observation.environment_state[4:7]
作为 obj_init_pos，补全 JSON 文件中缺失的 episode 数据。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def fix_episode_initial_states(
    dataset_path: str = "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_camcorner",
    json_path: str = "/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_camcorner/episode_initial_states.json",
    output_path: str | None = None,
    dry_run: bool = False,
):
    """
    补全 episode_initial_states.json 文件中缺失的 episode 数据。

    Args:
        dataset_path: LeRobot 数据集根目录路径
        json_path: 现有的 episode_initial_states.json 文件路径
        output_path: 输出文件路径，如果为 None 则覆盖原文件
        dry_run: 如果为 True，只打印信息不写入文件
    """
    dataset_path = Path(dataset_path)
    json_path = Path(json_path)

    if not dataset_path.exists():
        print(f"错误: 找不到数据集目录 {dataset_path}")
        sys.exit(1)

    if not json_path.exists():
        print(f"错误: 找不到 JSON 文件 {json_path}")
        sys.exit(1)

    from lerobot.datasets import LeRobotDataset

    print(f"加载数据集: {dataset_path}")
    dataset = LeRobotDataset(repo_id=dataset_path.name, root=str(dataset_path))
    total_episodes = dataset.num_episodes
    print(f"数据集总 episode 数: {total_episodes}")

    with open(json_path, "r") as f:
        metadata = json.load(f)

    existing_episodes = metadata["episodes"]
    existing_count = len(existing_episodes)
    existing_indices = {ep["episode_index"] for ep in existing_episodes}

    print(f"JSON 中已有 episode 数: {existing_count}")
    print(f"需要补全的 episode 数: {total_episodes - existing_count}")

    if existing_count >= total_episodes:
        print("JSON 文件已完整，无需补全。")
        return

    existing_episode_map = {ep["episode_index"]: ep for ep in existing_episodes}

    episodes_meta = dataset.meta.episodes

    for ep_idx in range(total_episodes):
        if ep_idx in existing_indices:
            continue

        from_idx = int(episodes_meta[ep_idx]["dataset_from_index"])
        to_idx = int(episodes_meta[ep_idx]["dataset_to_index"])
        num_frames = to_idx - from_idx

        frame = dataset[from_idx]

        env_state = frame["observation.environment_state"]
        if hasattr(env_state, "numpy"):
            env_state = env_state.numpy()

        obj_init_pos = env_state[4:7].tolist()

        goal_pose = existing_episodes[0]["goal_pose"] if existing_episodes else [0.1, 0.8, 0.2]

        new_episode = {
            "episode_index": ep_idx,
            "obj_init_pos": obj_init_pos,
            "goal_pose": goal_pose,
            "success": True,
            "num_frames": num_frames,
            "seed_used": "direct_set",
        }

        existing_episode_map[ep_idx] = new_episode
        print(f"  补全 episode {ep_idx}: obj_init_pos={obj_init_pos}")

    metadata["episodes"] = [existing_episode_map[i] for i in range(total_episodes)]
    metadata["num_episodes"] = total_episodes

    if output_path is None:
        output_path = json_path
    else:
        output_path = Path(output_path)

    if dry_run:
        print(f"\n[Dry Run] 将写入 {total_episodes} 个 episode 到 {output_path}")
        return

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n完成! 已写入 {total_episodes} 个 episode 到 {output_path}")


def main():
    parser = argparse.ArgumentParser(description="补全 episode_initial_states.json 文件")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_corner3",
        help="LeRobot 数据集根目录路径",
    )
    parser.add_argument(
        "--json",
        type=str,
        default="/data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_corner3/episode_initial_states.json",
        help="现有的 episode_initial_states.json 文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径 (默认覆盖原文件)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印信息，不写入文件",
    )
    args = parser.parse_args()

    fix_episode_initial_states(
        dataset_path=args.dataset,
        json_path=args.json,
        output_path=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()