#!/usr/bin/env python
"""
可视化Meta-World数据集的初始位置分布。

读取episode_initial_states.json文件，绘制obj_init_pos和goal_pose的3D分布图。

使用示例:
    python visualize_initial_positions.py \
        --dataset-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset \
        --output-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_metadata(dataset_dir):
    """加载episode_initial_states.json文件。"""
    metadata_file = Path(dataset_dir) / "episode_initial_states.json"

    if not metadata_file.exists():
        raise FileNotFoundError(f"找不到元数据文件: {metadata_file}")

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    return metadata


def plot_position_distribution(metadata, output_dir, task_name):
    """
    绘制初始位置和目标位置的3D分布图。

    Args:
        metadata: 元数据字典
        output_dir: 输出目录
        task_name: 任务名称
    """
    episodes = metadata["episodes"]

    # 提取obj_init_pos和goal_pose
    obj_positions = np.array([ep["obj_init_pos"] for ep in episodes])
    goal_positions = np.array([ep["goal_pose"] for ep in episodes])

    # 创建2x2的子图布局
    fig = plt.figure(figsize=(20, 16))

    # 1. obj_init_pos 3D散点图
    ax1 = fig.add_subplot(221, projection="3d")
    scatter1 = ax1.scatter(
        obj_positions[:, 0],
        obj_positions[:, 1],
        obj_positions[:, 2],
        c=range(len(obj_positions)),
        cmap="viridis",
        alpha=0.7,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax1.set_xlabel("X", fontsize=12)
    ax1.set_ylabel("Y", fontsize=12)
    ax1.set_zlabel("Z", fontsize=12)
    ax1.set_title(f"Object Initial Position Distribution\n({len(episodes)} episodes)", fontsize=14, fontweight="bold")
    plt.colorbar(scatter1, ax=ax1, label="Episode Index")

    # 2. goal_pose 3D散点图
    ax2 = fig.add_subplot(222, projection="3d")
    scatter2 = ax2.scatter(
        goal_positions[:, 0],
        goal_positions[:, 1],
        goal_positions[:, 2],
        c=range(len(goal_positions)),
        cmap="plasma",
        alpha=0.7,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax2.set_xlabel("X", fontsize=12)
    ax2.set_ylabel("Y", fontsize=12)
    ax2.set_zlabel("Z", fontsize=12)
    ax2.set_title(f"Goal Position Distribution\n({len(episodes)} episodes)", fontsize=14, fontweight="bold")
    plt.colorbar(scatter2, ax=ax2, label="Episode Index")

    # 3. obj_init_pos XY平面投影（俯视图）
    ax3 = fig.add_subplot(223)
    scatter3 = ax3.scatter(
        obj_positions[:, 0],
        obj_positions[:, 1],
        c=range(len(obj_positions)),
        cmap="viridis",
        alpha=0.7,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax3.set_xlabel("X", fontsize=12)
    ax3.set_ylabel("Y", fontsize=12)
    ax3.set_title(f"Object Initial Position - XY Projection", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label="Episode Index")

    # 4. goal_pose XY平面投影（俯视图）
    ax4 = fig.add_subplot(224)
    scatter4 = ax4.scatter(
        goal_positions[:, 0],
        goal_positions[:, 1],
        c=range(len(goal_positions)),
        cmap="plasma",
        alpha=0.7,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax4.set_xlabel("X", fontsize=12)
    ax4.set_ylabel("Y", fontsize=12)
    ax4.set_title(f"Goal Position - XY Projection", fontsize=14, fontweight="bold")
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter4, ax=ax4, label="Episode Index")

    plt.suptitle(f"Meta-World Position Distribution: {task_name}", fontsize=18, fontweight="bold", y=1.02)
    plt.tight_layout()

    # 保存图片
    output_path = Path(output_dir) / "position_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"分布图已保存到: {output_path}")

    # 打印统计信息
    print("\n" + "=" * 80)
    print("位置分布统计信息")
    print("=" * 80)
    print(f"\nObject Initial Position (obj_init_pos):")
    print(f"  X: min={obj_positions[:, 0].min():.4f}, max={obj_positions[:, 0].max():.4f}, "
          f"mean={obj_positions[:, 0].mean():.4f}, std={obj_positions[:, 0].std():.4f}")
    print(f"  Y: min={obj_positions[:, 1].min():.4f}, max={obj_positions[:, 1].max():.4f}, "
          f"mean={obj_positions[:, 1].mean():.4f}, std={obj_positions[:, 1].std():.4f}")
    print(f"  Z: min={obj_positions[:, 2].min():.4f}, max={obj_positions[:, 2].max():.4f}, "
          f"mean={obj_positions[:, 2].mean():.4f}, std={obj_positions[:, 2].std():.4f}")

    print(f"\nGoal Position (goal_pose):")
    print(f"  X: min={goal_positions[:, 0].min():.4f}, max={goal_positions[:, 0].max():.4f}, "
          f"mean={goal_positions[:, 0].mean():.4f}, std={goal_positions[:, 0].std():.4f}")
    print(f"  Y: min={goal_positions[:, 1].min():.4f}, max={goal_positions[:, 1].max():.4f}, "
          f"mean={goal_positions[:, 1].mean():.4f}, std={goal_positions[:, 1].std():.4f}")
    print(f"  Z: min={goal_positions[:, 2].min():.4f}, max={goal_positions[:, 2].max():.4f}, "
          f"mean={goal_positions[:, 2].mean():.4f}, std={goal_positions[:, 2].std():.4f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="可视化Meta-World数据集的初始位置分布",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python visualize_initial_positions.py \\
      --dataset-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset \\
      --output-dir /data/zhonglinye/jun/lerobot/personal/work2/dataset
        """,
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="数据集目录路径（包含episode_initial_states.json）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="图片保存目录（默认与dataset-dir相同）",
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = args.dataset_dir

    # 加载元数据
    print(f"加载数据集元数据: {args.dataset_dir}")
    metadata = load_metadata(args.dataset_dir)

    task_name = metadata["task"]
    num_episodes = metadata["num_episodes"]
    print(f"任务: {task_name}")
    print(f"Episode数量: {num_episodes}")

    # 绘制分布图
    plot_position_distribution(metadata, args.output_dir, task_name)


if __name__ == "__main__":
    main()