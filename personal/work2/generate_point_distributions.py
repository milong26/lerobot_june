#!/usr/bin/env python
"""
生成三种不同分布模式的点并可视化。

第一张图：100个均匀分布在整个空间的点
第二张图：100个随机分布的点
第三张图：将空间划分成5x5的格子，共100个点，特定格子点数稍多，每个格子中心至少有一个点

使用示例:
    python generate_point_distributions.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def generate_uniform_points(num_points=100):
    """
    生成均匀分布在整个空间的点。

    使用网格化方式确保点均匀分布。

    Args:
        num_points: 点的数量

    Returns:
        形状为 (num_points, 2) 的numpy数组
    """
    # 计算每个维度需要的点数 (100^(1/2) = 10，取10x10=100)
    n_x = 10
    n_y = 10

    # 创建均匀分布的坐标
    x = np.linspace(0, 1, n_x)
    y = np.linspace(0, 1, n_y)

    # 生成网格点
    xx, yy = np.meshgrid(x, y, indexing='ij')
    points = np.column_stack([xx.ravel(), yy.ravel()])

    # 如果点数超过100，截取前100个
    if len(points) > num_points:
        points = points[:num_points]

    return points


def generate_random_points(num_points=100):
    """
    生成随机分布的点。

    Args:
        num_points: 点的数量

    Returns:
        形状为 (num_points, 2) 的numpy数组
    """
    # 设置随机种子以确保可重复性
    np.random.seed(42)

    # 在[0, 1]范围内随机生成点
    points = np.random.uniform(0, 1, size=(num_points, 2))

    return points


def generate_grid_points(num_points=100):
    """
    生成基于5x5网格分布的点。

    特定格子的点数分布（随机分配）：
    - 第1行：第1、4列点数稍多
    - 第2行：第3列点数稍多
    - 第3行：第3、4列点数稍多
    - 第4行：第2列点数稍多
    - 第5行：第5列点数稍多
    每个格子中心至少有一个点。

    Args:
        num_points: 总点数

    Returns:
        形状为 (num_points, 2) 的numpy数组
    """
    np.random.seed(42)

    # 定义5x5网格中每个格子的点数
    # 行索引从上到下为0-4，列索引从左到右为0-4
    grid_points = np.zeros((5, 5), dtype=int)

    # 每个格子至少1个点（基础点数）
    grid_points[:, :] = 1

    # 已分配25个点（5x5网格每个格子1个）
    remaining_points = num_points - 25

    # 按照要求，特定点数稍多，但随机分配（每个格子增加2-4个点）
    # 第1行（索引0）：第1、4列（索引0、3）
    grid_points[0, 0] += np.random.randint(2, 5)  # 第1行第1列，随机2-4个
    grid_points[0, 3] += np.random.randint(2, 5)  # 第1行第4列，随机2-4个
    
    # 第2行（索引1）：第3列（索引2）
    grid_points[1, 2] += np.random.randint(2, 5)  # 第2行第3列，随机2-4个
    
    # 第3行（索引2）：第3、4列（索引2、3）
    grid_points[2, 2] += np.random.randint(2, 5)  # 第3行第3列，随机2-4个
    grid_points[2, 3] += np.random.randint(2, 5)  # 第3行第4列，随机2-4个
    
    # 第4行（索引3）：第2列（索引1）
    grid_points[3, 1] += np.random.randint(2, 5)  # 第4行第2列，随机2-4个
    
    # 第5行（索引4）：第5列（索引4）
    grid_points[4, 4] += np.random.randint(2, 5)  # 第5行第5列，随机2-4个

    # 计算已分配的额外点数
    allocated = grid_points.sum() - 25
    remaining_points -= allocated

    # 如果还有剩余点数，随机分配到所有格子
    if remaining_points > 0:
        for i in range(remaining_points):
            row = np.random.randint(0, 5)
            col = np.random.randint(0, 5)
            grid_points[row, col] += 1

    # 生成点
    all_points = []

    for row in range(5):
        for col in range(5):
            num_in_cell = grid_points[row, col]

            # 计算格子中心位置
            cell_center_x = (col + 0.5) / 5
            cell_center_y = (row + 0.5) / 5

            # 第一个点放在格子中心
            all_points.append([cell_center_x, cell_center_y])

            # 其余点在格子内随机分布
            for _ in range(num_in_cell - 1):
                x = np.random.uniform(col / 5, (col + 1) / 5)
                y = np.random.uniform(row / 5, (row + 1) / 5)
                all_points.append([x, y])

    points = np.array(all_points)

    return points


def plot_distributions(output_dir):
    """
    绘制三种分布模式的点。

    Args:
        output_dir: 输出目录
    """
    # 生成三种分布的点
    uniform_points = generate_uniform_points(100)
    random_points = generate_random_points(100)
    grid_points = generate_grid_points(100)

    # 1. 均匀分布
    fig1 = plt.figure(figsize=(8, 8))
    ax1 = fig1.add_subplot(111)
    ax1.scatter(
        uniform_points[:, 0],
        uniform_points[:, 1],
        c="blue",
        alpha=0.7,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax1.set_xlabel("X", fontsize=12)
    ax1.set_ylabel("Y", fontsize=12)
    ax1.set_title(f"Uniform Distribution\n(100 points)", fontsize=14, fontweight="bold")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path1 = Path(output_dir) / "uniform_distribution.png"
    plt.savefig(output_path1, dpi=150, bbox_inches="tight")
    print(f"均匀分布图已保存到: {output_path1}")
    plt.close(fig1)

    # 2. 随机分布
    fig2 = plt.figure(figsize=(8, 8))
    ax2 = fig2.add_subplot(111)
    ax2.scatter(
        random_points[:, 0],
        random_points[:, 1],
        c="red",
        alpha=0.7,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax2.set_xlabel("X", fontsize=12)
    ax2.set_ylabel("Y", fontsize=12)
    ax2.set_title(f"Random Distribution\n(100 points)", fontsize=14, fontweight="bold")
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path2 = Path(output_dir) / "random_distribution.png"
    plt.savefig(output_path2, dpi=150, bbox_inches="tight")
    print(f"随机分布图已保存到: {output_path2}")
    plt.close(fig2)

    # 3. 网格分布
    fig3 = plt.figure(figsize=(8, 8))
    ax3 = fig3.add_subplot(111)
    ax3.scatter(
        grid_points[:, 0],
        grid_points[:, 1],
        c="green",
        alpha=0.7,
        s=50,
        edgecolors="black",
        linewidth=0.5,
    )
    ax3.set_xlabel("X", fontsize=12)
    ax3.set_ylabel("Y", fontsize=12)
    ax3.set_title(f"Grid-based Distribution\n(100 points)", fontsize=14, fontweight="bold")
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)

    # 绘制网格线
    for i in range(6):
        ax3.axvline(x=i/5, color='k', alpha=0.3, linewidth=0.5)
        ax3.axhline(y=i/5, color='k', alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    output_path3 = Path(output_dir) / "grid_distribution.png"
    plt.savefig(output_path3, dpi=150, bbox_inches="tight")
    print(f"网格分布图已保存到: {output_path3}")
    plt.close(fig3)

    # 打印统计信息
    print("\n" + "=" * 80)
    print("点分布统计信息")
    print("=" * 80)

    print(f"\n1. Uniform Distribution:")
    print(f"  点数: {len(uniform_points)}")
    print(f"  X: min={uniform_points[:, 0].min():.4f}, max={uniform_points[:, 0].max():.4f}, "
          f"mean={uniform_points[:, 0].mean():.4f}")
    print(f"  Y: min={uniform_points[:, 1].min():.4f}, max={uniform_points[:, 1].max():.4f}, "
          f"mean={uniform_points[:, 1].mean():.4f}")

    print(f"\n2. Random Distribution:")
    print(f"  点数: {len(random_points)}")
    print(f"  X: min={random_points[:, 0].min():.4f}, max={random_points[:, 0].max():.4f}, "
          f"mean={random_points[:, 0].mean():.4f}")
    print(f"  Y: min={random_points[:, 1].min():.4f}, max={random_points[:, 1].max():.4f}, "
          f"mean={random_points[:, 1].mean():.4f}")

    print(f"\n3. Grid-based Distribution:")
    print(f"  点数: {len(grid_points)}")
    print(f"  X: min={grid_points[:, 0].min():.4f}, max={grid_points[:, 0].max():.4f}, "
          f"mean={grid_points[:, 0].mean():.4f}")
    print(f"  Y: min={grid_points[:, 1].min():.4f}, max={grid_points[:, 1].max():.4f}, "
          f"mean={grid_points[:, 1].mean():.4f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="生成三种不同分布模式的点并可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python generate_point_distributions.py
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="100points",
        help="图片保存目录（默认：100points）",
    )

    args = parser.parse_args()

    # 确保输出目录存在
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 绘制分布图
    plot_distributions(args.output_dir)


if __name__ == "__main__":
    main()