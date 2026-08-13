"""
夹爪关闭阈值可视化工具
功能：
- 绘制夹爪位置随时间变化的曲线
- 画出关闭阈值线（默认10%）
- 帮助确定实际环境中多少百分比算"关闭"
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
import os


def plot_gripper_threshold(
    gripper_values: List[float],
    threshold: float = 10.0,
    title: str = "Gripper Position vs Close Threshold",
    save_path: Optional[str] = None,
):
    """
    绘制夹爪位置曲线和关闭阈值线
    
    Args:
        gripper_values: 夹爪位置数据列表（百分比，0-100）
        threshold: 关闭阈值百分比（默认10%）
        title: 图表标题
        save_path: 保存路径，如果为None则显示
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制夹爪位置曲线
    steps = range(len(gripper_values))
    plt.plot(steps, gripper_values, 'b-', linewidth=1.5, label='Gripper Position (%)', alpha=0.7)
    
    # 绘制关闭阈值线
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=2, 
                label=f'Close Threshold ({threshold}%)')
    
    # 填充阈值以下区域（表示"关闭"状态）
    plt.fill_between(steps, 0, threshold, alpha=0.1, color='red', label='Closed Region')
    
    # 标注统计信息
    mean_val = np.mean(gripper_values)
    min_val = np.min(gripper_values)
    max_val = np.max(gripper_values)
    
    stats_text = (
        f"Mean: {mean_val:.2f}%\n"
        f"Min: {min_val:.2f}%\n"
        f"Max: {max_val:.2f}%\n"
        f"Below threshold: {sum(1 for v in gripper_values if v <= threshold)}/{len(gripper_values)}"
    )
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Gripper Position (%)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylim(-5, 105)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] 图表已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_gripper_comparison(
    gripper_values: List[float],
    thresholds: List[float] = [5.0, 10.0, 15.0, 20.0],
    title: str = "Gripper Position - Multiple Thresholds Comparison",
    save_path: Optional[str] = None,
):
    """
    绘制多个阈值的对比图，帮助选择最佳阈值
    
    Args:
        gripper_values: 夹爪位置数据列表
        thresholds: 要对比的阈值列表
        title: 图表标题
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, threshold in enumerate(thresholds):
        ax = axes[idx]
        steps = range(len(gripper_values))
        
        ax.plot(steps, gripper_values, 'b-', linewidth=1.5, alpha=0.7)
        ax.axhline(y=threshold, color='r', linestyle='--', linewidth=2,
                  label=f'Threshold = {threshold}%')
        ax.fill_between(steps, 0, threshold, alpha=0.1, color='red')
        
        below_count = sum(1 for v in gripper_values if v <= threshold)
        below_pct = below_count / len(gripper_values) * 100
        
        ax.set_title(f'Threshold {threshold}%: {below_count}/{len(gripper_values)} ({below_pct:.1f}%)',
                    fontsize=11)
        ax.set_xlabel('Step')
        ax.set_ylabel('Gripper Position (%)')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[INFO] 对比图已保存到: {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_test_data():
    """生成测试数据用于演示"""
    np.random.seed(42)
    
    # 模拟一个完整的抓取动作：张开 -> 闭合 -> 张开
    data = []
    
    # 阶段1: 完全张开 (100%)
    data.extend([100.0] * 20)
    
    # 阶段2: 逐渐闭合 (100% -> 0%)
    data.extend(np.linspace(100, 0, 30).tolist())
    
    # 阶段3: 完全闭合 (0-5%)
    data.extend(np.random.uniform(0, 5, 25).tolist())
    
    # 阶段4: 抓取中 (5-15%，有噪声)
    data.extend(np.random.uniform(5, 15, 20).tolist())
    
    # 阶段5: 释放张开 (15% -> 100%)
    data.extend(np.linspace(15, 100, 25).tolist())
    
    # 阶段6: 完全张开 (100%)
    data.extend([100.0] * 20)
    
    return data


if __name__ == "__main__":
    # 使用测试数据演示
    print("[INFO] 使用测试数据演示...")
    test_data = generate_test_data()
    
    # 1. 绘制单阈值图
    plot_gripper_threshold(
        test_data,
        threshold=10.0,
        title="Gripper Position - Close Threshold Analysis",
        save_path="gripper_threshold_analysis.png"
    )
    
    # 2. 绘制多阈值对比图
    plot_gripper_comparison(
        test_data,
        thresholds=[5.0, 10.0, 15.0, 20.0],
        title="Gripper Position - Multiple Thresholds Comparison",
        save_path="gripper_threshold_comparison.png"
    )
    
    print("\n[INFO] 演示完成！请查看生成的图片文件。")
    print("[INFO] 实际使用时，请替换为真实的夹爪数据。")