"""
对比两个实验的 Payload 差异
1. 数据集实验（test_payload_with_dataset.py）
2. 真实机器人实验（main_controller.py）
"""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor
from lerobot.datasets import LeRobotDataset


DATASET_PATH = os.path.expanduser("/home/qwe/.cache/huggingface/lerobot/ep10/valve")


def load_dataset_frame(dataset, frame_idx=50):
    """从数据集加载一帧"""
    frame = dataset[frame_idx]
    camera_keys = dataset.meta.camera_keys
    has_force = "observation.force" in dataset.meta.features
    
    images = {}
    for key in camera_keys:
        if key in frame:
            img = frame[key]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            images[key.split('.')[-1]] = img
    
    state = frame['observation.state']
    if isinstance(state, torch.Tensor):
        state = state.numpy()
    
    if has_force and 'observation.force' in frame:
        force = frame['observation.force']
        if isinstance(force, torch.Tensor):
            force = force.numpy()
    else:
        force = np.zeros(15, dtype=np.float32)
    
    action = frame['action']
    if isinstance(action, torch.Tensor):
        action = action.numpy()
    
    return {
        'images': images,
        'state': state,
        'force': force,
        'action': action,
        'index': frame_idx
    }


def compare_payloads(payload1, payload2, label1="Dataset", label2="Real Robot"):
    """对比两个 payload 的差异"""
    print("\n" + "="*80)
    print("Payload 对比分析")
    print("="*80)
    
    # 1. 图像对比
    print("\n1. 图像对比:")
    img1_0 = np.array(payload1['image'][0])
    img1_1 = np.array(payload1['image'][1])
    img2_0 = np.array(payload2['image'][0])
    img2_1 = np.array(payload2['image'][1])
    
    print(f"   {label1}:")
    print(f"     Image 0 shape: {img1_0.shape}, dtype: {img1_0.dtype}")
    print(f"     Image 1 shape: {img1_1.shape}, dtype: {img1_1.dtype}")
    print(f"     Image 0 vs 1 相同: {np.array_equal(img1_0, img1_1)}")
    
    print(f"   {label2}:")
    print(f"     Image 0 shape: {img2_0.shape}, dtype: {img2_0.dtype}")
    print(f"     Image 1 shape: {img2_1.shape}, dtype: {img2_1.dtype}")
    print(f"     Image 0 vs 1 相同: {np.array_equal(img2_0, img2_1)}")
    
    img_diff_0 = np.abs(img1_0.astype(float) - img2_0.astype(float)).mean()
    img_diff_1 = np.abs(img1_1.astype(float) - img2_1.astype(float)).mean()
    print(f"   Image 0 平均差异: {img_diff_0:.2f}")
    print(f"   Image 1 平均差异: {img_diff_1:.2f}")
    
    # 2. State 对比
    print("\n2. State 对比:")
    state1 = np.array(payload1['state'])
    state2 = np.array(payload2['state'])
    
    print(f"   {label1}:")
    print(f"     Shape: {state1.shape}")
    print(f"     Frame 0: {state1[0][:6]}")
    print(f"     Frame 1: {state1[1][:6]}")
    
    print(f"   {label2}:")
    print(f"     Shape: {state2.shape}")
    print(f"     Frame 0: {state2[0][:6]}")
    print(f"     Frame 1: {state2[1][:6]}")
    
    state_diff = np.abs(state1 - state2).mean()
    print(f"   State 平均差异: {state_diff:.4f}")
    
    # 3. Action 对比
    print("\n3. Action 对比:")
    action1 = np.array(payload1['action'])
    action2 = np.array(payload2['action'])
    
    print(f"   {label1}:")
    print(f"     Shape: {action1.shape}")
    print(f"     Frame 0: {action1[0][:6]}")
    print(f"     Frame 1: {action1[1][:6]}")
    
    print(f"   {label2}:")
    print(f"     Shape: {action2.shape}")
    print(f"     Frame 0: {action2[0][:6]}")
    print(f"     Frame 1: {action2[1][:6]}")
    
    action_diff = np.abs(action1 - action2).mean()
    print(f"   Action 平均差异: {action_diff:.4f}")
    
    # 4. 其他字段对比
    print("\n4. 其他字段对比:")
    for key in ['prompt', 'steps', 'seed', 'g_scale', 'image_mask', 'action_mask', 'num_loop']:
        val1 = payload1[key]
        val2 = payload2[key]
        match = "✓" if val1 == val2 else "✗"
        print(f"   {key}: {match} ({val1} vs {val2})")


def main():
    print("="*80)
    print("对比数据集实验和真实机器人实验的 Payload")
    print("="*80)
    
    print(f"\n加载数据集: {DATASET_PATH}")
    dataset = LeRobotDataset("ep10/valve", root=DATASET_PATH, episodes=[0])
    
    # 加载连续5帧
    frames = []
    for i in range(50, 55):
        frames.append(load_dataset_frame(dataset, i))
    
    print(f"加载了 5 帧数据 (索引 50-54)")
    print(f"State 示例: {frames[0]['state'][:6]}")
    print(f"Force 示例: {frames[0]['force'][:6]}")
    
    processor = DataProcessor(history_size=5)
    
    # 实验1: 数据集测试（使用 ground truth action）
    print("\n" + "="*80)
    print("实验 1: 数据集测试 (使用 ground truth action)")
    print("="*80)
    
    processor.image_history.clear()
    processor.action_history.clear()
    
    history_states = []
    history_actions = []
    for j in range(5):
        history_states.append((frames[j]['state'], frames[j]['force']))
        history_actions.append(frames[j]['action'])
    
    dataset_payload = processor.build_payload(
        images=frames[4]['images'],
        state=frames[4]['state'],
        force=frames[4]['force'],
        prompt="Grab the cross-shape equipment.",
        history_states=history_states,
        history_actions=history_actions
    )
    
    print(f"历史状态数量: {len(history_states)}")
    print(f"历史动作数量: {len(history_actions)}")
    print(f"历史动作来源: 数据集 ground truth")
    print(f"State[-5]: {history_states[-5][0][:6]}")
    print(f"State[-1]: {history_states[-1][0][:6]}")
    print(f"Action[-5]: {history_actions[-5][:6]}")
    print(f"Action[-1]: {history_actions[-1][:6]}")
    
    # 实验2: 真实机器人（使用零 action）
    print("\n" + "="*80)
    print("实验 2: 真实机器人 (使用零 action)")
    print("="*80)
    
    processor.image_history.clear()
    processor.action_history.clear()
    
    real_history_actions = [np.zeros(24, dtype=np.float32) for _ in range(5)]
    
    real_payload = processor.build_payload(
        images=frames[4]['images'],
        state=frames[4]['state'],
        force=frames[4]['force'],
        prompt="Grab the cross-shape equipment.",
        history_states=history_states,
        history_actions=real_history_actions
    )
    
    print(f"历史状态数量: {len(history_states)}")
    print(f"历史动作数量: {len(real_history_actions)}")
    print(f"历史动作来源: 零值")
    
    # 对比
    compare_payloads(dataset_payload, real_payload, "Dataset (GT action)", "Real Robot (Zero action)")


if __name__ == "__main__":
    main()