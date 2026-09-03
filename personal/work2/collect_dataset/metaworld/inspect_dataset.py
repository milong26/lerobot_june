#!/usr/bin/env python
"""快速查看 LeRobot 数据集的基本信息和统计。"""

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def inspect_dataset(dataset_path: str):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"错误: 数据集目录不存在: {dataset_path}")
        return

    print(f"数据集路径: {dataset_path}")
    print("=" * 60)

    dataset = LeRobotDataset(
        repo_id=dataset_path.name,
        root=str(dataset_path),
    )

    print(f"repo_id: {dataset.repo_id}")
    print(f"total_episodes: {dataset.meta.total_episodes}")
    print(f"total_frames: {dataset.meta.total_frames}")
    print(f"total_tasks: {dataset.meta.total_tasks}")
    print(f"fps: {dataset.meta.fps}")
    print(f"robot_type: {dataset.meta.robot_type}")
    print(f"video_keys: {dataset.meta.video_keys}")
    print(f"image_keys: {dataset.meta.image_keys}")

    print(f"\nFeatures:")
    for name, feat in dataset.meta.features.items():
        print(f"  {name}: dtype={feat['dtype']}, shape={feat.get('shape', 'N/A')}")

    print(f"\nTasks:")
    if dataset.meta.tasks is not None and len(dataset.meta.tasks) > 0:
        for task_name, row in dataset.meta.tasks.iterrows():
            print(f"  [{row['task_index']}] {task_name}")
    else:
        print("  (无 task 信息)")

    print(f"\nEpisodes (前5个):")
    for i, ep in enumerate(dataset.meta.episodes[:5]):
        print(f"  ep {ep['episode_index']}: task_idx={ep.get('task_index', '?')}, "
              f"frames={ep.get('length', '?')}, from={ep.get('dataset_from_index', '?')}")
    if len(dataset.meta.episodes) > 5:
        print(f"  ... 共 {len(dataset.meta.episodes)} 个 episodes")

    print(f"\nVideo keys: {dataset.meta.video_keys}")
    print(f"Image keys: {dataset.meta.image_keys}")

    # 检查 episode_initial_states.json
    metadata_file = dataset_path / "episode_initial_states.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            meta = json.load(f)
        print(f"\nepisode_initial_states.json:")
        print(f"  num_episodes: {meta.get('num_episodes', 'N/A')}")
        print(f"  task: {meta.get('task', 'N/A')}")

        episodes = meta.get("episodes", [])
        if episodes:
            success_count = sum(1 for ep in episodes if ep.get("success"))
            print(f"  episodes in json: {len(episodes)}")
            print(f"  success count: {success_count}")

            has_obj = sum(1 for ep in episodes if ep.get("obj_init_pos") is not None)
            has_goal = sum(1 for ep in episodes if ep.get("goal_pose") is not None)
            has_rand_vec = sum(1 for ep in episodes if ep.get("rand_vec") is not None)
            print(f"  has obj_init_pos: {has_obj}/{len(episodes)}")
            print(f"  has goal_pose: {has_goal}/{len(episodes)}")
            print(f"  has rand_vec: {has_rand_vec}/{len(episodes)}")

            if has_obj > 0:
                first_obj = next(ep["obj_init_pos"] for ep in episodes if ep.get("obj_init_pos"))
                print(f"  示例 obj_init_pos: {first_obj}")
            if has_goal > 0:
                first_goal = next(ep["goal_pose"] for ep in episodes if ep.get("goal_pose"))
                print(f"  示例 goal_pose: {first_goal}")
    else:
        print(f"\nepisode_initial_states.json: 不存在")

    # 验证一致性
    print(f"\n一致性检查:")
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            meta = json.load(f)
        json_count = meta.get("num_episodes", 0)
        ds_count = dataset.meta.total_episodes
        if json_count == ds_count:
            print(f"  [PASS] episode 数量一致: {json_count}")
        else:
            print(f"  [FAIL] episode 数量不一致: json={json_count}, dataset={ds_count}")

    # 随机查看一帧数据
    print(f"\n随机查看一帧数据:")
    frame_idx = np.random.randint(0, dataset.meta.total_frames)
    frame = dataset[frame_idx]
    for key, val in frame.items():
        if hasattr(val, "shape"):
            print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  {key}: {val}")

    print(f"\n{'=' * 60}")
    print("检查完成")


def main():
    parser = argparse.ArgumentParser(description="查看 LeRobot 数据集信息")
    parser.add_argument("--dataset", type=str, required=True, help="数据集路径")
    args = parser.parse_args()
    inspect_dataset(args.dataset)


if __name__ == "__main__":
    main()