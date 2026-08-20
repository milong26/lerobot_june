#!/usr/bin/env python
"""
从指定的episode中读取initial pose并打印出来。

读取episode_initial_states.json文件，打印指定episode的obj_init_pos和goal_pose。

使用示例:
    # 打印单个episode
    python print_initial_poses.py \
        --dataset-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset \
        --episodes 0

    # 打印多个episode
    python print_initial_poses.py \
        --dataset-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset \
        --episodes 0,1,2,3,4

    # 打印episode范围
    python print_initial_poses.py \
        --dataset-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset \
        --episodes 0-9

    # 打印所有成功的episode
    python print_initial_poses.py \
        --dataset-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset \
        --success-only

    # 打印所有episode
    python print_initial_poses.py \
        --dataset-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset \
        --all
"""

import argparse
import json
from pathlib import Path


def load_metadata(dataset_dir):
    """加载episode_initial_states.json文件。"""
    metadata_file = Path(dataset_dir) / "episode_initial_states.json"

    if not metadata_file.exists():
        raise FileNotFoundError(f"找不到元数据文件: {metadata_file}")

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    return metadata


def parse_episode_indices(episodes_str, num_episodes):
    """
    解析episode索引字符串。

    支持的格式:
    - 单个数字: "0"
    - 逗号分隔: "0,1,2,3"
    - 范围: "0-9"
    - 混合: "0,1,5-10"

    Args:
        episodes_str: episode索引字符串
        num_episodes: 总episode数量

    Returns:
        episode索引列表
    """
    indices = set()

    parts = episodes_str.split(",")
    for part in parts:
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start = int(start.strip())
            end = int(end.strip())
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))

    valid_indices = [i for i in indices if 0 <= i < num_episodes]
    valid_indices.sort()

    return valid_indices


def print_episode_pose(episode_data, detailed=False):
    """
    打印单个episode的initial pose信息。

    Args:
        episode_data: episode数据字典
        detailed: 是否打印详细信息
    """
    episode_index = episode_data["episode_index"]
    obj_init_pos = episode_data["obj_init_pos"]
    goal_pose = episode_data["goal_pose"]
    success = episode_data["success"]
    num_frames = episode_data["num_frames"]

    print(f"\nEpisode {episode_index}:")
    print(f"  obj_init_pos: [{obj_init_pos[0]:.6f}, {obj_init_pos[1]:.6f}, {obj_init_pos[2]:.6f}]")
    print(f"  goal_pose:    [{goal_pose[0]:.6f}, {goal_pose[1]:.6f}, {goal_pose[2]:.6f}]")
    print(f"  success: {success}, num_frames: {num_frames}")

    if detailed:
        print(f"  obj_init_pos (X, Y, Z):")
        print(f"    X = {obj_init_pos[0]:.6f}")
        print(f"    Y = {obj_init_pos[1]:.6f}")
        print(f"    Z = {obj_init_pos[2]:.6f}")
        print(f"  goal_pose (X, Y, Z):")
        print(f"    X = {goal_pose[0]:.6f}")
        print(f"    Y = {goal_pose[1]:.6f}")
        print(f"    Z = {goal_pose[2]:.6f}")


def print_statistics(episodes):
    """
    打印episode的统计信息。

    Args:
        episodes: episode数据列表
    """
    if not episodes:
        print("没有符合条件的episode。")
        return

    obj_positions = [ep["obj_init_pos"] for ep in episodes]
    goal_positions = [ep["goal_pose"] for ep in episodes]

    print("\n" + "=" * 80)
    print("统计信息")
    print("=" * 80)
    print(f"Episode数量: {len(episodes)}")
    print(f"成功率: {sum(1 for ep in episodes if ep['success'])}/{len(episodes)} "
          f"({sum(1 for ep in episodes if ep['success'])/len(episodes)*100:.1f}%)")

    print(f"\nObject Initial Position (obj_init_pos):")
    print(f"  X: min={min(p[0] for p in obj_positions):.6f}, "
          f"max={max(p[0] for p in obj_positions):.6f}, "
          f"mean={sum(p[0] for p in obj_positions)/len(obj_positions):.6f}")
    print(f"  Y: min={min(p[1] for p in obj_positions):.6f}, "
          f"max={max(p[1] for p in obj_positions):.6f}, "
          f"mean={sum(p[1] for p in obj_positions)/len(obj_positions):.6f}")
    print(f"  Z: min={min(p[2] for p in obj_positions):.6f}, "
          f"max={max(p[2] for p in obj_positions):.6f}, "
          f"mean={sum(p[2] for p in obj_positions)/len(obj_positions):.6f}")

    print(f"\nGoal Position (goal_pose):")
    print(f"  X: min={min(p[0] for p in goal_positions):.6f}, "
          f"max={max(p[0] for p in goal_positions):.6f}, "
          f"mean={sum(p[0] for p in goal_positions)/len(goal_positions):.6f}")
    print(f"  Y: min={min(p[1] for p in goal_positions):.6f}, "
          f"max={max(p[1] for p in goal_positions):.6f}, "
          f"mean={sum(p[1] for p in goal_positions)/len(goal_positions):.6f}")
    print(f"  Z: min={min(p[2] for p in goal_positions):.6f}, "
          f"max={max(p[2] for p in goal_positions):.6f}, "
          f"mean={sum(p[2] for p in goal_positions)/len(goal_positions):.6f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="从指定的episode中读取initial pose并打印出来",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 打印单个episode
  python print_initial_poses.py \\
      --dataset-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset \\
      --episodes 0

  # 打印多个episode
  python print_initial_poses.py \\
      --dataset-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset \\
      --episodes 0,1,2,3,4

  # 打印episode范围
  python print_initial_poses.py \\
      --dataset-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset \\
      --episodes 0-9

  # 打印所有成功的episode
  python print_initial_poses.py \\
      --dataset-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset \\
      --success-only

  # 打印所有episode
  python print_initial_poses.py \\
      --dataset-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset \\
      --all
        """,
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/data/zhonglinye/jun/lerobot/personal/work2/dataset",
        help="数据集目录路径（包含episode_initial_states.json）",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default=None,
        help="Episode索引，支持单个数字、逗号分隔、范围或混合格式 (如: 0 或 0,1,2 或 0-9 或 0,5-10)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="打印所有episode",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="只打印成功的episode",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="打印详细信息",
    )

    args = parser.parse_args()

    if not args.all and not args.success_only and args.episodes is None:
        parser.error("请指定 --episodes、--all 或 --success-only 中的一个")

    # 加载元数据
    print(f"加载数据集元数据: {args.dataset_dir}")
    metadata = load_metadata(args.dataset_dir)

    task_name = metadata["task"]
    num_episodes = metadata["num_episodes"]
    all_episodes = metadata["episodes"]

    print(f"任务: {task_name}")
    print(f"总Episode数量: {num_episodes}")

    # 确定要打印的episode
    if args.all:
        target_episodes = all_episodes
        print(f"\n打印所有 {len(target_episodes)} 个episode:")
    elif args.success_only:
        target_episodes = [ep for ep in all_episodes if ep["success"]]
        print(f"\n打印 {len(target_episodes)} 个成功的episode:")
    else:
        indices = parse_episode_indices(args.episodes, num_episodes)
        if not indices:
            print("没有有效的episode索引。")
            return
        target_episodes = [all_episodes[i] for i in indices]
        print(f"\n打印 {len(target_episodes)} 个指定的episode (索引: {indices}):")

    # 打印每个episode的pose
    print("-" * 80)
    for episode in target_episodes:
        print_episode_pose(episode, detailed=args.detailed)

    # 打印统计信息
    print_statistics(target_episodes)


if __name__ == "__main__":
    main()