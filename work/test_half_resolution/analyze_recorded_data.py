"""
数据分析脚本
功能：
- 从保存的 npz 文件中读取 state 和 action 数据
- 绘制 action（期望值）和 state（实际值）的对比图
- 分析机械臂跟随效果

使用方法：
python analyze_recorded_data.py --data_dir ./debug_recorded_data/20260728_150732
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import re


def load_recorded_data(data_dir: str):
    """
    从 npz 文件中加载数据
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        frames: 帧数据列表
    """
    data_path = Path(data_dir)
    
    # 获取所有 frame_*.npz 文件（不包含 images）
    npz_files = sorted(data_path.glob("frame_*.npz"))
    npz_files = [f for f in npz_files if "images" not in f.name]
    
    if not npz_files:
        raise ValueError(f"在 {data_dir} 中未找到 npz 文件")
    
    print(f"找到 {len(npz_files)} 个数据文件")
    
    frames = []
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=False)
        
        frame_data = {
            'filename': npz_file.name,
            'timestamp': data['timestamp'],
            'state': data['state'],  # 6维关节状态
        }
        
        # 如果有 action 数据
        if 'action' in data:
            frame_data['action'] = data['action']
        
        # 如果有力传感器数据
        if 'force' in data:
            frame_data['force'] = data['force']
        
        frames.append(frame_data)
    
    return frames


def plot_action_vs_state(frames: list, save_path: str = None):
    """
    绘制 action 和 state 的对比图
    
    Args:
        frames: 帧数据列表
        save_path: 图片保存路径（可选）
    """
    # 关节名称
    joint_names = [
        'shoulder_pan',
        'shoulder_lift', 
        'elbow_flex',
        'wrist_flex',
        'wrist_roll',
        'gripper'
    ]
    
    # 检查是否有 action 数据
    has_action = 'action' in frames[0]
    
    if has_action:
        print(f"\n数据包含 action 数据，共 {len(frames)} 帧")
    else:
        print(f"\n数据不包含 action 数据，仅绘制 state 数据，共 {len(frames)} 帧")
    
    # 创建图形：2行3列
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Action vs State Comparison (关节角度对比)', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    # 提取数据
    timestamps = np.array([f['timestamp'] for f in frames])
    states = np.array([f['state'] for f in frames])
    
    if has_action:
        actions = np.array([f['action'][:6] for f in frames])
    
    # 绘制每个关节
    for i in range(6):
        ax = axes[i]
        
        # 绘制 state（实际值）- 红色
        ax.plot(timestamps, states[:, i], 'r-', linewidth=1.5, label='State (实际)', alpha=0.8)
        
        # 如果有 action，绘制 action（期望值）- 蓝色
        if has_action:
            ax.plot(timestamps, actions[:, i], 'b-', linewidth=1.5, label='Action (期望)', alpha=0.8)
        
        ax.set_title(joint_names[i], fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Angle (degrees)', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 打印统计信息
        if has_action:
            error = np.abs(actions[:, i] - states[:, i])
            print(f"\n{joint_names[i]}:")
            print(f"  State 范围: [{states[:, i].min():.2f}, {states[:, i].max():.2f}]")
            print(f"  Action 范围: [{actions[:, i].min():.2f}, {actions[:, i].max():.2f}]")
            print(f"  平均误差: {error.mean():.2f}°, 最大误差: {error.max():.2f}°")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图片已保存到: {save_path}")
    
    plt.show()


def plot_state_only(frames: list, save_path: str = None):
    """
    仅绘制 state 数据（当没有 action 数据时）
    
    Args:
        frames: 帧数据列表
        save_path: 图片保存路径（可选）
    """
    joint_names = [
        'shoulder_pan',
        'shoulder_lift', 
        'elbow_flex',
        'wrist_flex',
        'wrist_roll',
        'gripper'
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('State Trajectory (关节状态轨迹)', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    timestamps = np.array([f['timestamp'] for f in frames])
    states = np.array([f['state'] for f in frames])
    
    for i in range(6):
        ax = axes[i]
        ax.plot(timestamps, states[:, i], 'r-', linewidth=2)
        ax.set_title(joint_names[i], fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Angle (degrees)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        print(f"\n{joint_names[i]}:")
        print(f"  范围: [{states[:, i].min():.2f}, {states[:, i].max():.2f}]")
        print(f"  均值: {states[:, i].mean():.2f}°")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图片已保存到: {save_path}")
    
    plt.show()


def plot_force_data(frames: list, save_path: str = None):
    """
    绘制力传感器数据（如果有）
    
    Args:
        frames: 帧数据列表
        save_path: 图片保存路径（可选）
    """
    if 'force' not in frames[0]:
        print("\n没有力传感器数据")
        return
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle('Force Sensor Data (力传感器数据)', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    timestamps = np.array([f['timestamp'] for f in frames])
    forces = np.array([f['force'] for f in frames])
    
    for i in range(15):
        ax = axes[i]
        ax.plot(timestamps, forces[:, i], 'g-', linewidth=1.5)
        ax.set_title(f'Sensor {i//3} - Axis {i%3}', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Force', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n力传感器图片已保存到: {save_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='分析录制的数据')
    parser.add_argument('--data_dir', type=str, 
                       default='./work/test_half_resolution/debug_recorded_data/20260728_154911',
                       help='数据目录路径（默认：./debug_recorded_data/20260728_154911')
    parser.add_argument('--save_plot', type=str, default=None, help='保存图片路径')
    parser.add_argument('--plot_force', action='store_true', help='是否绘制力传感器数据')
    
    args = parser.parse_args()
    
    print(f"正在加载数据: {args.data_dir}")
    frames = load_recorded_data(args.data_dir)
    
    # 绘制 action vs state
    if 'action' in frames[0]:
        plot_action_vs_state(frames, save_path=args.save_plot)
    else:
        plot_state_only(frames, save_path=args.save_plot)
    
    # 绘制力传感器数据（可选）
    if args.plot_force:
        plot_force_data(frames, save_path=args.save_plot.replace('.png', '_force.png') if args.save_plot else None)


if __name__ == "__main__":
    main()