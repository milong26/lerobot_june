"""
Payload 分析脚本 - 直接显示图像和数据
功能：
- 从录制的数据中重建 payload
- 直接显示图像（不保存）
- 在控制台显示图像的 shape、state 的值等详细信息

使用方法：
python analyze_payload.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor


def load_frames_from_npz(data_dir: str):
    """
    从 npz 文件中加载所有帧数据
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        frames: 帧数据列表
    """
    data_path = Path(data_dir)
    
    # 获取所有 frame_*.npz 文件（不包含 images）
    npz_files = sorted(data_path.glob("frame_*.npz"))
    npz_files = [f for f in npz_files if "images" not in f.name]
    
    # 获取所有 image 文件
    img_files = sorted(data_path.glob("*_images.npz"))
    
    if not npz_files:
        raise ValueError(f"在 {data_dir} 中未找到 npz 文件")
    
    print(f"找到 {len(npz_files)} 个数据文件")
    print(f"找到 {len(img_files)} 个图像文件")
    
    frames = []
    for npz_file in npz_files:
        data = np.load(npz_file, allow_pickle=False)
        
        frame_data = {
            'filename': npz_file.name,
            'timestamp': float(data['timestamp']),
            'state': data['state'],  # 6维关节状态
            'force': data['force'],  # 15维力传感器
        }
        
        # 如果有 action 数据
        if 'action' in data:
            frame_data['action'] = data['action']
        
        frames.append(frame_data)
    
    # 加载图像数据
    for img_file in img_files:
        # 从文件名提取帧索引
        frame_idx = int(img_file.name.split('_')[1])
        
        img_data = np.load(img_file, allow_pickle=False)
        
        # 查找匹配的帧
        for frame in frames:
            if f"frame_{frame_idx:06d}" in frame['filename']:
                frame['images'] = {}
                for key in img_data.files:
                    frame['images'][key] = img_data[key]
                break
    
    return frames


def analyze_payload(frames: list):
    """
    分析并重建 payload，直接显示图像
    
    Args:
        frames: 帧数据列表
    """
    # 初始化 DataProcessor
    processor = DataProcessor( history_size=5)
    
    print("\n" + "="*80)
    print("开始分析 Payload")
    print("="*80)
    
    # 分析前3帧的 payload
    num_frames_to_analyze = min(3, len(frames))
    
    for i in range(num_frames_to_analyze):
        print(f"\n{'='*60}")
        print(f"帧 {i} (索引 {i})")
        print(f"{'='*60}")
        
        # 获取当前帧数据
        current_frame = frames[i]
        state = current_frame['state']
        force = current_frame['force']
        images = current_frame.get('images', {})
        
        print(f"\n时间戳: {current_frame['timestamp']}")
        print(f"\nState (6维):")
        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        for j, name in enumerate(joint_names):
            print(f"  {name}: {state[j]:.2f}")
        
        print(f"\nForce (15维，前6个值): {force[:6]}")
        
        # 构建历史状态
        history_states = []
        for j in range(max(0, i-4), i+1):
            history_states.append((frames[j]['state'], frames[j]['force']))
        
        # 构建历史动作
        history_actions = []
        for j in range(max(0, i-4), i+1):
            if 'action' in frames[j]:
                history_actions.append(frames[j]['action'][:6])
            else:
                history_actions.append(np.zeros(6, dtype=np.float32))
        
        # 构建 payload
        payload = processor.build_payload(
            images=images,
            state=state,
            force=force,
            prompt="Test task",
            history_states=history_states,
            history_actions=history_actions
        )
        
        # 打印 payload 信息
        print(f"\nPayload 结构:")
        print(f"  - image: {len(payload['image'])} 张图像")
        for img_idx, img in enumerate(payload['image']):
            img_array = np.array(img)
            print(f"    图像 {img_idx}: shape={img_array.shape}, dtype={img_array.dtype}")
        
        print(f"\n  - state: {len(payload['state'])} 帧")
        for s_idx, s in enumerate(payload['state']):
            print(f"    State {s_idx}: {s[:6]}")
        
        print(f"\n  - action: {len(payload['action'])} 帧")
        for a_idx, a in enumerate(payload['action']):
            print(f"    Action {a_idx}: {a[:6]}")
        
        print(f"\n  - image_mask: {payload['image_mask']}")
        print(f"  - action_mask: {payload['action_mask'][:6]}")
        
        # 检查图像是否相同
        if len(payload['image']) >= 2:
            img1 = np.array(payload['image'][0])
            img2 = np.array(payload['image'][1])
            if np.array_equal(img1, img2):
                print(f"\n  ⚠️  警告: 图像 0 和图像 1 完全相同！")
            else:
                diff = np.abs(img1.astype(float) - img2.astype(float)).mean()
                print(f"\n  ✓ 图像 0 和图像 1 不同，平均差异: {diff:.2f}")
    
    # 显示图像
    print(f"\n{'='*60}")
    print("显示 Payload 中的图像")
    print(f"{'='*60}")
    
    # 使用最后一帧的 payload 来显示图像
    last_frame = frames[-1]
    images = last_frame.get('images', {})
    
    # 构建历史状态
    history_states = []
    for j in range(max(0, len(frames)-5), len(frames)):
        history_states.append((frames[j]['state'], frames[j]['force']))
    
    # 构建历史动作
    history_actions = []
    for j in range(max(0, len(frames)-5), len(frames)):
        if 'action' in frames[j]:
            history_actions.append(frames[j]['action'][:6])
        else:
            history_actions.append(np.zeros(6, dtype=np.float32))
    
    # 构建 payload
    payload = processor.build_payload(
        images=images,
        state=last_frame['state'],
        force=last_frame['force'],
        prompt="Test task",
        history_states=history_states,
        history_actions=history_actions
    )
    
    # 显示图像
    num_images = len(payload['image'])
    if num_images > 0:
        fig, axes = plt.subplots(1, num_images, figsize=(6*num_images, 4))
        if num_images == 1:
            axes = [axes]
        
        for idx, ax in enumerate(axes):
            img = np.array(payload['image'][idx])
            ax.imshow(img)
            ax.set_title(f"Image {idx}\nShape: {img.shape}")
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # 显示原始图像（如果有）
    if images:
        print(f"\n{'='*60}")
        print("显示原始图像")
        print(f"{'='*60}")
        
        fig, axes = plt.subplots(1, len(images), figsize=(6*len(images), 4))
        if len(images) == 1:
            axes = [axes]
        
        for idx, (key, img) in enumerate(images.items()):
            axes[idx].imshow(img)
            axes[idx].set_title(f"{key}\nShape: {img.shape}")
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # 总结
    print(f"\n{'='*60}")
    print("分析总结")
    print(f"{'='*60}")
    print(f"1. 图像是否正确使用了历史帧？")
    print(f"   - 图像历史缓冲区大小: {len(processor.image_history)}")
    print(f"   - Payload 中图像数量: {len(payload['image'])}")
    print(f"2. State 是否正确使用了历史帧？")
    print(f"   - Payload 中 state 帧数: {len(payload['state'])}")
    print(f"3. Action 是否正确使用了历史帧？")
    print(f"   - Payload 中 action 帧数: {len(payload['action'])}")


def main():
    # 数据目录
    data_dir = "./work/test_half_resolution/debug_recorded_data/20260728_154911"
    
    print(f"正在加载数据: {data_dir}")
    frames = load_frames_from_npz(data_dir)
    
    print(f"\n成功加载 {len(frames)} 帧数据")
    
    # 分析 payload
    analyze_payload(frames)


if __name__ == "__main__":
    main()