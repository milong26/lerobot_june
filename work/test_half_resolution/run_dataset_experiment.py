"""
使用本地 ep10/valve 数据集模拟真实机器人实验
功能：
- 从本地 LeRobot 数据集加载数据（代替真实传感器输入）
- 按照与真实机器人相同的方式打包成 payload
- 发送到服务器获取预测的 action
- 将推理出的 action 发送给真实机器人执行
- 对比服务器预测的 action 和数据集中的 ground truth action
- 绘制对比曲线图

使用方法：
python run_dataset_experiment.py
"""

import asyncio
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import time
import torch
import logging

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor
from ws_client import WSClient
from lerobot.datasets import LeRobotDataset

# LeRobot 导入
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig
from lerobot.robots import (  # noqa: F401
    so_follower,
    koch_follower,
    omx_follower,
    bi_so_follower,
    openarm_follower,
    bi_openarm_follower,
    lekiwi,
    hope_jr,
    reachy2,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# 配置区 - 修改这些参数即可
# ============================================================
DATASET_PATH = os.path.expanduser("/home/qwe/.cache/huggingface/lerobot/ep10/valve")
SERVER_URL = "ws://10.10.16.19:9000"
TASK = "Grab the cross-shape equipment."
MAX_FRAMES = None  # None 表示使用所有帧，或设置具体数字如 100
FPS = 30  # 控制频率
EPISODE_IDX = 0  # 使用哪个 episode

# 机器人配置（需要根据你的实际硬件修改）
ROBOT_CONFIG = SOFollowerRobotConfig(
    port="/dev/ttyACM1",
    cameras={
        "wrist": OpenCVCameraConfig(
            index_or_path=6,
            width=640,
            height=480,
            fps=30,
            fourcc="MJPG",
        ),
        "top": RealSenseCameraConfig(
            serial_number_or_name="806312060427",
            width=640,
            height=480,
            fps=30,
            use_depth=False,
        ),
    },
    id="start_new_heihei_2",
)
# ============================================================


def get_dataset_frames(dataset, episode_idx=0):
    """
    从数据集中获取帧数据
    
    Args:
        dataset: LeRobotDataset 实例
        episode_idx: episode 索引
        
    Returns:
        frames: 帧数据列表
        has_force: 是否有力传感器数据
    """
    # 获取 episode 的帧范围
    from_idx = dataset.meta.episodes["dataset_from_index"][episode_idx]
    to_idx = dataset.meta.episodes["dataset_to_index"][episode_idx]
    
    print(f"Episode {episode_idx}: 帧范围 [{from_idx}, {to_idx})")
    print(f"总帧数: {to_idx - from_idx}")
    
    # 获取相机键
    camera_keys = dataset.meta.camera_keys
    print(f"相机键: {camera_keys}")
    print(f"State 键: observation.state")
    print(f"Action 键: action")
    print(f"Force 键: observation.force (如果存在)")
    
    # 检查是否有 force 数据
    features = dataset.meta.features
    has_force = "observation.force" in features
    print(f"是否有力传感器数据: {has_force}")
    
    frames = []
    for idx in range(from_idx, to_idx):
        frame = dataset[idx]
        
        frame_data = {
            'index': idx,
        }
        
        # 提取图像
        images = {}
        for key in camera_keys:
            if key in frame:
                img = frame[key]
                if isinstance(img, torch.Tensor):
                    img = img.numpy()
                # 转换 (C, H, W) -> (H, W, C)
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                images[key.split('.')[-1]] = img  # 使用 'top', 'wrist' 等键名
        
        frame_data['images'] = images
        
        # 提取 state
        if 'observation.state' in frame:
            state = frame['observation.state']
            if isinstance(state, torch.Tensor):
                state = state.numpy()
            frame_data['state'] = state
        
        # 提取 force（如果有）
        if has_force and 'observation.force' in frame:
            force = frame['observation.force']
            if isinstance(force, torch.Tensor):
                force = force.numpy()
            frame_data['force'] = force
        else:
            frame_data['force'] = np.zeros(15, dtype=np.float32)
        
        # 提取 ground truth action
        if 'action' in frame:
            action = frame['action']
            if isinstance(action, torch.Tensor):
                action = action.numpy()
            frame_data['action'] = action
        
        frames.append(frame_data)
    
    return frames, has_force


async def run_experiment():
    """主实验函数"""
    print("="*80)
    print("使用本地 ep10/valve 数据集模拟真实机器人实验")
    print("="*80)
    
    # 1. 连接真实机器人
    print(f"\n正在连接真实机器人...")
    robot = make_robot_from_config(ROBOT_CONFIG)
    robot.connect()
    print("✅ 真实机器人已连接")
    
    # 2. 加载数据集
    print(f"\n正在加载数据集: {DATASET_PATH}")
    dataset = LeRobotDataset("ep10/valve", root=DATASET_PATH, episodes=[EPISODE_IDX])
    print(f"数据集加载完成")
    print(f"  - 总 episode 数: {dataset.num_episodes}")
    print(f"  - 总帧数: {dataset.num_frames}")
    print(f"  - FPS: {dataset.meta.fps}")
    
    # 3. 获取帧数据
    frames, has_force = get_dataset_frames(dataset, episode_idx=EPISODE_IDX)
    
    # 限制帧数
    if MAX_FRAMES is not None:
        frames = frames[:MAX_FRAMES]
    
    print(f"\n成功加载 {len(frames)} 帧数据")
    
    # 4. 初始化 DataProcessor（与真实机器人相同的方式）
    processor = DataProcessor(history_size=5)
    
    # 5. 初始化 WebSocket 客户端
    ws_client = WSClient(SERVER_URL)
    
    # 6. 连接服务器
    print(f"\n正在连接服务器: {SERVER_URL}")
    await ws_client.connect()
    print("✅ 服务器连接成功")
    
    # 7. 存储结果
    predicted_actions = []
    ground_truth_actions = []
    
    print(f"\n开始实验 {len(frames)} 帧...")
    print("="*80)
    
    # 8. 主循环：数据集输入→处理→发送→接收→执行真实动作
    for i, frame in enumerate(frames):
        loop_start = time.time()
        
        # 构建历史状态（最近5帧，从数据集读取）
        history_states = []
        for j in range(max(0, i-4), i+1):
            history_states.append((frames[j]['state'], frames[j]['force']))
        
        # 构建历史动作（最近5帧）
        history_actions = []
        for j in range(max(0, i-4), i+1):
            if 'action' in frames[j]:
                history_actions.append(frames[j]['action'])
            else:
                history_actions.append(np.zeros(24, dtype=np.float32))
        
        # 构建 payload（与真实机器人相同的方式）
        payload = processor.build_payload(
            images=frame['images'],
            state=frame['state'],
            force=frame['force'],
            prompt=TASK,
            history_states=history_states,
            history_actions=history_actions
        )
        
        # 发送请求并接收响应
        try:
            response = await ws_client.send_and_receive(payload)
            
            # 提取预测的 action
            actions = response['actions']
            if len(actions) > 0:
                # 执行第一个动作
                action_to_execute = actions[0]
                pred_action = np.array(action_to_execute)
                
                # 将 action 转换为机器人格式并发送给真实机器人
                action_dict = {
                    f"{motor}.pos": float(pred_action[j])
                    for j, motor in enumerate(['shoulder_pan', 'shoulder_lift', 'elbow_flex', 
                                                'wrist_flex', 'wrist_roll', 'gripper'])
                }
                
                # 发送 action 到真实机器人执行
                robot.send_action(action_dict)
                print(f"帧 {i}: ✅ 已发送 action 到机器人")
                
                # 记录预测的 action（只取前6维用于对比）
                predicted_actions.append(pred_action[:6])
                
                # 提取 ground truth action
                if 'action' in frame:
                    gt_action = frame['action'][:6]
                    ground_truth_actions.append(gt_action)
                
                # 更新动作历史
                processor.update_action_history(pred_action)
                
                elapsed = time.time() - loop_start
                print(f"帧 {i}: 耗时 {elapsed*1000:.1f}ms | "
                      f"Predicted: {pred_action[:3]}... | "
                      f"GT: {gt_action[:3]}...")
            else:
                print(f"帧 {i}: 服务器返回空 actions")
                
        except Exception as e:
            print(f"帧 {i}: 错误 - {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # 控制频率（30Hz）
        elapsed = time.time() - loop_start
        sleep_time = 1.0/FPS - elapsed
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)
    
    # 9. 断开服务器和机器人
    await ws_client.disconnect()
    robot.disconnect()
    print("✅ 服务器和机器人已断开")
    
    # 10. 绘制对比图
    if predicted_actions and ground_truth_actions:
        print(f"\n{'='*80}")
        print("绘制对比图")
        print(f"{'='*80}")
        plot_comparison(predicted_actions, ground_truth_actions, save_path="./action_comparison_dataset.png")
    else:
        print("\n没有足够的数据绘制对比图")


def plot_comparison(predicted_actions, ground_truth_actions, save_path=None):
    """
    绘制预测 action 和 ground truth action 的对比图
    
    Args:
        predicted_actions: 预测的 action 列表
        ground_truth_actions: 真实的 action 列表
        save_path: 保存路径（可选）
    """
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 
                   'wrist_flex', 'wrist_roll', 'gripper']
    
    num_joints = min(6, len(ground_truth_actions[0]) if ground_truth_actions else 6)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Predicted Action vs Ground Truth Action (Dataset Experiment)', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    steps = np.arange(len(predicted_actions))
    
    for i in range(num_joints):
        ax = axes[i]
        
        pred_vals = [a[i] for a in predicted_actions]
        gt_vals = [a[i] for a in ground_truth_actions]
        
        ax.plot(steps, pred_vals, 'b-', linewidth=1.5, label='Predicted (服务器)', alpha=0.8)
        ax.plot(steps, gt_vals, 'r-', linewidth=1.5, label='Ground Truth (数据集)', alpha=0.8)
        
        ax.set_title(joint_names[i], fontsize=12, fontweight='bold')
        ax.set_xlabel('Step', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 打印统计信息
        error = np.abs(np.array(pred_vals) - np.array(gt_vals))
        print(f"\n{joint_names[i]}:")
        print(f"  Predicted 范围: [{min(pred_vals):.2f}, {max(pred_vals):.2f}]")
        print(f"  Ground Truth 范围: [{min(gt_vals):.2f}, {max(gt_vals):.2f}]")
        print(f"  平均误差: {error.mean():.2f}, 最大误差: {error.max():.2f}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n图片已保存到: {save_path}")
    
    plt.show()


def main():
    asyncio.run(run_experiment())


if __name__ == "__main__":
    main()