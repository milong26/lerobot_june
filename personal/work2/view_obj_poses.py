#!/usr/bin/env python
"""
快速查看LeRobot数据集中每个episode的初始物体位置(obj_pose)。

支持两种查看方式：
1. 从采集时生成的 episode_initial_states.json 文件读取（最快）
2. 直接从LeRobot数据集中读取 observation.environment_state[4:7]

使用示例:
    # 方式1: 从JSON文件查看（推荐，最快）
    python view_obj_poses.py --json ./outputs/metaworld_pick_place/episode_initial_states.json

    # 方式2: 从LeRobot数据集查看
    python view_obj_poses.py --dataset ./outputs/metaworld_pick_place

    # 同时显示两种方式并对比
    python view_obj_poses.py \
        --json ./outputs/metaworld_pick_place/episode_initial_states.json \
        --dataset ./outputs/metaworld_pick_place
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np


def view_from_json(json_path):
    """从JSON文件快速查看obj_pose。"""
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"错误: 找不到文件 {json_file}")
        return None

    with open(json_file, "r") as f:
        metadata = json.load(f)

    print("=" * 80)
    print(f"任务: {metadata['task']}")
    print(f"Episode数量: {metadata['num_episodes']}")
    print("=" * 80)
    print(f"{'Episode':<10} {'Obj X':<12} {'Obj Y':<12} {'Obj Z':<12} {'Goal X':<12} {'Goal Y':<12} {'Goal Z':<12} {'Success':<10}")
    print("-" * 80)

    for ep in metadata["episodes"]:
        obj = ep["obj_init_pos"]
        goal = ep["goal_pose"]
        print(
            f"{ep['episode_index']:<10} "
            f"{obj[0]:<12.4f} {obj[1]:<12.4f} {obj[2]:<12.4f} "
            f"{goal[0]:<12.4f} {goal[1]:<12.4f} {goal[2]:<12.4f} "
            f"{'✓' if ep['success'] else '✗':<10}"
        )

    # 统计信息
    obj_positions = np.array([ep["obj_init_pos"] for ep in metadata["episodes"]])
    print("\n" + "=" * 80)
    print("物体初始位置统计:")
    print(f"  X: min={obj_positions[:, 0].min():.4f}, max={obj_positions[:, 0].max():.4f}, mean={obj_positions[:, 0].mean():.4f}")
    print(f"  Y: min={obj_positions[:, 1].min():.4f}, max={obj_positions[:, 1].max():.4f}, mean={obj_positions[:, 1].mean():.4f}")
    print(f"  Z: min={obj_positions[:, 2].min():.4f}, max={obj_positions[:, 2].max():.4f}, mean={obj_positions[:, 2].mean():.4f}")
    print("=" * 80)

    return metadata


def view_from_dataset(dataset_path):
    """从LeRobot数据集中读取obj_pose。"""
    from lerobot.datasets import LeRobotDataset

    ds_path = Path(dataset_path)
    if not ds_path.exists():
        print(f"错误: 找不到数据集目录 {ds_path}")
        return None

    print(f"\n从LeRobot数据集加载: {ds_path}")
    dataset = LeRobotDataset(root=str(ds_path))

    print(f"总帧数: {dataset.num_frames}")
    print(f"总Episode数: {dataset.num_episodes}")

    # 获取每个episode的第一帧的environment_state
    print(f"\n{'Episode':<10} {'Obj X':<12} {'Obj Y':<12} {'Obj Z':<12}")
    print("-" * 50)

    obj_positions = []
    for ep_idx in range(dataset.num_episodes):
        # 获取该episode的第一帧
        first_frame_idx = dataset.episode_data_index["from"][ep_idx]
        frame = dataset[int(first_frame_idx)]

        env_state = frame["observation.environment_state"]
        if hasattr(env_state, "numpy"):
            env_state = env_state.numpy()

        # 39维observation中 [4:7] 是物体位置
        obj_pos = env_state[4:7]
        obj_positions.append(obj_pos)

        print(f"{ep_idx:<10} {obj_pos[0]:<12.4f} {obj_pos[1]:<12.4f} {obj_pos[2]:<12.4f}")

    obj_positions = np.array(obj_positions)
    print("\n" + "=" * 80)
    print("物体初始位置统计:")
    print(f"  X: min={obj_positions[:, 0].min():.4f}, max={obj_positions[:, 0].max():.4f}, mean={obj_positions[:, 0].mean():.4f}")
    print(f"  Y: min={obj_positions[:, 1].min():.4f}, max={obj_positions[:, 1].max():.4f}, mean={obj_positions[:, 1].mean():.4f}")
    print(f"  Z: min={obj_positions[:, 2].min():.4f}, max={obj_positions[:, 2].max():.4f}, mean={obj_positions[:, 2].mean():.4f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="查看Meta-World数据集的obj_pose初始值")
    parser.add_argument("--json", type=str, default=None, help="episode_initial_states.json 文件路径")
    parser.add_argument("--dataset", type=str, default=None, help="LeRobot数据集目录路径")
    args = parser.parse_args()

    if args.json is None and args.dataset is None:
        print("请指定 --json 或 --dataset 参数")
        parser.print_help()
        sys.exit(1)

    if args.json:
        view_from_json(args.json)

    if args.dataset:
        view_from_dataset(args.dataset)


if __name__ == "__main__":
    main()