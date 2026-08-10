"""
从数据集回放 observation 到服务器

功能：
- 从 LeRobot 数据集读取 observation（图像、state、force）
- 发送给服务器推理
- 执行服务器返回的动作
- 数据集用完后，保持最后一个 action 继续运行，不会突然停止
- 支持键盘控制：右键复位、左键恢复、下键关闭

使用方式（与 main_controller.py 相同的命令行参数）：
python replay_dataset_to_server.py     --robot.type=so100_follower     --robot.port=/dev/ttyACM1     --robot.id=start_new_heihei_2     --task="Grab the rectangular object"     --server_url=ws://10.10.16.19:9001     --fps=30     --action_steps=10     --num_loop=2
"""

import asyncio
import time
import logging
import sys
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import deque

import numpy as np
import draccus
from pynput import keyboard
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor
from ws_client import WSClient

# LeRobot 导入
from lerobot.datasets import LeRobotDataset
from lerobot.robots import make_robot_from_config
from lerobot.robots.config import RobotConfig

# 导入所有机器人模块以触发 @RobotConfig.register_subclass 注册
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

# ==================== 配置 ====================
# 数据集路径
DATASET_ROOT = "/home/qwe/.cache/huggingface/lerobot/ep10/rectangular"
EPISODE_INDEX = 0
# ==============================================


@dataclass
class ReplayConfig:
    """回放配置"""
    robot: RobotConfig
    task: str
    server_url: str = "ws://10.10.16.19:9001"
    fps: int = 30
    action_steps: Optional[int] = None
    num_loop: int = 2


class DatasetReplayer:
    """数据集回放器"""
    
    def __init__(
        self,
        robot_config: RobotConfig,
        dataset_root: str,
        episode_index: int,
        server_url: str,
        task: str,
        fps: int = 30,
        action_steps: Optional[int] = None,
        num_loop: int = 2,
    ):
        self.dataset_root = dataset_root
        self.episode_index = episode_index
        self.server_url = server_url
        self.task = task
        self.fps = fps
        self.dt = 1.0 / fps
        self.action_steps = action_steps
        self.num_loop = num_loop
        
        # 加载数据集
        self.dataset = LeRobotDataset(repo_id="", root=Path(dataset_root))
        print(f"✅ 数据集加载完成，共 {len(self.dataset)} 帧")
        
        # 获取 episode 的起始索引
        self.episode_start = self.dataset.meta.episodes["dataset_from_index"][episode_index]
        self.episode_end = self.dataset.meta.episodes["dataset_to_index"][episode_index]
        print(f"📊 Episode {episode_index}: 帧 {self.episode_start} ~ {self.episode_end}")
        
        self.current_frame_index = self.episode_start
        
        # 获取相机 keys
        self.camera_keys = self.dataset.meta.camera_keys
        if not self.camera_keys:
            raise ValueError("数据集中没有相机数据")
        print(f"📷 相机 keys: {self.camera_keys}")
        
        # 初始化机器人（使用与 main_controller.py 相同的方式）
        print("正在连接机器人...")
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()
        print("✅ 机器人已连接")
        
        # 初始化数据处理器
        self.processor = DataProcessor()
        
        # 初始化 WebSocket 客户端
        self.ws_client = WSClient(server_url)
        
        # 历史帧队列（用于构建 payload 时选取 -133ms 的帧）
        self.frame_history = deque(maxlen=20)  # 保存最近 20 帧
        
        # 控制标志
        self.running = False
        self.last_observation = None
        self.last_action = None
        
        # Debug: 记录帧索引映射
        self.frame_index_map = {}  # timestamp -> dataset_frame_index
        self.debug_mode = True  # 开启debug模式
        
        # Debug: 创建输出目录
        self.debug_output_dir = Path("/home/qwe/jun/lerobot/work/test_half_resolution/debug_output")
        self.debug_output_dir.mkdir(exist_ok=True)
        self.debug_comparison_count = 0  # chunk比较计数器
        self.debug_cumulative_count = 0  # 累积比较计数器
        
        # Debug: 记录实际执行的action
        self.executed_actions = []  # 记录所有实际执行的action (每个action包含6维)
        self.executed_action_frame_indices = []  # 记录每个action对应的数据集帧索引
        
        # 键盘控制状态
        self.initial_state = None  # 复位目标状态
        self.is_resetting = False  # 是否正在执行复位
        self.is_paused = False  # 是否暂停服务器通信（复位后）
        self.keyboard_listener = None
        
        # 线程安全的事件标志
        self._reset_requested = threading.Event()
        self._shutdown_requested = threading.Event()
        self._resume_requested = threading.Event()
    
    def get_observation_from_dataset(self) -> dict:
        """从数据集获取 observation"""
        if self.current_frame_index >= self.episode_end:
            # 数据集已用完，触发安全关闭
            print("⚠️  数据集已用完，触发安全关闭")
            self._shutdown_requested.set()
            return None
        
        # 从数据集读取
        frame_data = self.dataset[self.current_frame_index]
        
        # 提取图像
        images = {}
        for key in self.camera_keys:
            if key in frame_data:
                img = frame_data[key]
                if hasattr(img, "numpy"):
                    img = img.numpy()
                
                # 转换为 uint8
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                
                # CHW -> HWC
                if img.ndim == 3 and img.shape[0] == 3:
                    img = img.transpose(1, 2, 0)
                
                images[key] = img
        
        # 提取 state
        state = frame_data['observation.state']
        if hasattr(state, "numpy"):
            state = state.numpy()
        state = np.array(state, dtype=np.float64)
        
        # 提取 force（如果有）
        if 'observation.force' in frame_data:
            force = frame_data['observation.force']
            if hasattr(force, "numpy"):
                force = force.numpy()
            force = np.array(force, dtype=np.float64)
        else:
            raise ValueError("数据集中没有 force 数据")
        
        observation = {
            'state': state,
            'force': force,
            'images': images,
            'timestamp': time.time(),
            'dataset_frame_index': self.current_frame_index  # Debug: 记录数据集帧索引
        }
        
        # Debug: 打印帧索引信息
        if self.debug_mode:
            print(f"[DEBUG] 读取数据集帧索引: {self.current_frame_index}")
        
        # 保存最后一帧
        self.last_observation = observation
        self.current_frame_index += 1
        
        return observation
    
    def plot_action_comparison(self, server_actions, dataset_actions, observation_frame_idx, title="Action Comparison"):
        """
        绘制服务器推理action与数据集action的对比图（chunk级别）
        
        Args:
            server_actions: numpy array (N, 6) 服务器返回的action（前6维）
            dataset_actions: numpy array (N, 6) 数据集的action
            observation_frame_idx: int 发送的observation帧索引
            title: str 图表标题
        """
        num_steps = len(server_actions)
        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{title}\nObservation Frame: {observation_frame_idx} → Actions [{observation_frame_idx} ~ {observation_frame_idx + num_steps - 1}]', 
                     fontsize=14, fontweight='bold')
        
        axes = axes.flatten()
        
        for joint_idx in range(6):
            ax = axes[joint_idx]
            
            # 服务器action
            server_values = server_actions[:, joint_idx]
            ax.plot(range(num_steps), server_values, 'b-o', label='Server (Inferred)', 
                   linewidth=2, markersize=4, alpha=0.7)
            
            # 数据集action
            dataset_values = dataset_actions[:, joint_idx]
            ax.plot(range(num_steps), dataset_values, 'r-s', label='Dataset (Ground Truth)', 
                   linewidth=2, markersize=4, alpha=0.7)
            
            # 计算误差
            errors = np.abs(server_values - dataset_values)
            max_error = np.max(errors)
            mean_error = np.mean(errors)
            
            ax.set_xlabel('Action Step Index')
            ax.set_ylabel('Value')
            ax.set_title(f'{joint_names[joint_idx]}\nMax Error: {max_error:.2f}, Mean Error: {mean_error:.2f}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        self.debug_comparison_count += 1
        filename = f"chunk_comparison_{self.debug_comparison_count:03d}_obs{observation_frame_idx}.png"
        filepath = self.debug_output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[DEBUG] 💾 已保存Chunk对比图: {filepath}")
    
    def plot_cumulative_comparison(self, episode_start, episode_end):
        """
        绘制累积的action对比图（完整episode）
        
        Args:
            episode_start: int episode起始帧索引
            episode_end: int episode结束帧索引
        """
        if len(self.executed_actions) == 0:
            return
        
        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        
        # 读取数据集完整action
        num_dataset_frames = episode_end - episode_start
        dataset_actions_all = []
        for i in range(num_dataset_frames):
            frame_idx = episode_start + i
            if frame_idx < episode_end:
                frame_data = self.dataset[frame_idx]
                action = frame_data['action']
                if hasattr(action, "numpy"):
                    action = action.numpy()
                dataset_actions_all.append(action)
        
        dataset_actions_array = np.array(dataset_actions_all)  # (N, 6)
        executed_actions_array = np.array(self.executed_actions)  # (M, 6)
        executed_indices = np.array(self.executed_action_frame_indices)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Cumulative Action Comparison - Executed vs Dataset\n'
                     f'Total Executed: {len(self.executed_actions)} actions', 
                     fontsize=14, fontweight='bold')
        
        axes = axes.flatten()
        
        for joint_idx in range(6):
            ax = axes[joint_idx]
            
            # 数据集完整action（作为背景）
            dataset_values = dataset_actions_array[:, joint_idx]
            ax.plot(range(len(dataset_values)), dataset_values, 'r-', 
                   label='Dataset (Complete)', linewidth=2, alpha=0.5)
            
            # 实际执行的action（高亮显示）
            executed_values = executed_actions_array[:, joint_idx]
            ax.plot(executed_indices, executed_values, 'b-o', 
                   label='Executed (Server Inferred)', linewidth=2, markersize=3, alpha=0.8)
            
            # 计算已执行部分的误差
            errors = []
            for i, frame_idx in enumerate(self.executed_action_frame_indices):
                if frame_idx < episode_end:
                    server_val = executed_actions_array[i, joint_idx]
                    dataset_val = dataset_actions_array[frame_idx - episode_start, joint_idx]
                    errors.append(abs(server_val - dataset_val))
            
            if errors:
                max_error = np.max(errors)
                mean_error = np.mean(errors)
                error_text = f'\nMax Error: {max_error:.2f}, Mean Error: {mean_error:.2f}'
            else:
                error_text = ''
            
            ax.set_xlabel('Frame Index')
            ax.set_ylabel('Value')
            ax.set_title(f'{joint_names[joint_idx]}{error_text}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        self.debug_cumulative_count += 1
        filename = f"cumulative_{self.debug_cumulative_count:03d}_executed{len(self.executed_actions)}.png"
        filepath = self.debug_output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[DEBUG] 💾 已保存累积对比图: {filepath} (已执行{len(self.executed_actions)}个action)")
    
    def send_action(self, action):
        """发送动作到机器人"""
        if isinstance(action, np.ndarray):
            action_dict = {
                'shoulder_pan.pos': action[0],
                'shoulder_lift.pos': action[1],
                'elbow_flex.pos': action[2],
                'wrist_flex.pos': action[3],
                'wrist_roll.pos': action[4],
                'gripper.pos': action[5],
            }
        else:
            action_dict = action
        
        self.robot.send_action(action_dict)
    
    def _setup_keyboard_listener(self):
        """设置键盘监听器"""
        def on_press(key):
            try:
                if key == keyboard.Key.right:
                    print("[INFO] 检测到右键 -> 请求复位")
                    self._reset_requested.set()
                elif key == keyboard.Key.down:
                    print("[INFO] 检测到下键 -> 请求安全关闭")
                    self._shutdown_requested.set()
                elif key == keyboard.Key.left:
                    print("[INFO] 检测到左键 -> 请求恢复")
                    self._resume_requested.set()
            except AttributeError:
                pass
        
        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()
        print("[INFO] 键盘监听器已启动 (右键=复位, 左键=恢复, 下键=关闭)")
    
    async def _execute_reset(self, current_state):
        """执行复位操作"""
        if self.initial_state is None:
            logger.warning("[WARN] 未设置初始状态，无法复位")
            return
        
        self.is_resetting = True
        # 计算插值步数（平滑过渡，避免跳变）
        diff = self.initial_state - current_state
        max_diff = np.max(np.abs(diff))
        
        # 根据最大差值计算步数，每步最多移动 1.5 弧度
        steps = max(10, int(max_diff / 1.5))
        
        print(f"复位需要 {steps} 步，最大差值: {max_diff:.3f}")
        
        for step in range(1, steps + 1):
            if not self.running:
                break
            
            # 线性插值
            alpha = step / steps
            target_state = current_state + alpha * diff
            
            # 发送插值后的状态
            self.send_action(target_state.tolist())
            
            # 控制频率
            await asyncio.sleep(self.dt)
        
        print("[INFO] 复位完成")
        self.is_resetting = False
        self.is_paused = True  # 复位后暂停服务器通信
        print("[INFO] 已暂停服务器通信，按左键恢复")
    
    async def _safe_shutdown(self):
        """安全关闭程序"""
        self.running = False
        
        # 停止键盘监听
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        
        # 断开 WebSocket
        await self.ws_client.disconnect()
        print("WebSocket 已断开")
        
        # 断开机器人（检查是否已连接）
        try:
            self.robot.disconnect()
            print("机器人已断开")
        except Exception as e:
            print(f"⚠️  机器人断开时出错（可能未连接）: {e}")
        
        print("[INFO] 安全关闭完成")
    
    def build_payload(self, frame_current: dict, frame_minus_133ms: dict) -> dict:
        """构建 payload"""
        # 处理图像
        processed_image_minus_133ms = self.processor.process_images(frame_minus_133ms['images'])
        processed_image_current = self.processor.process_images(frame_current['images'])
        
        # 构建历史状态
        history_states = [
            {'state': frame_minus_133ms['state'], 'force': frame_minus_133ms['force']},
            {'state': frame_current['state'], 'force': frame_current['force']}
        ]
        
        payload = self.processor.build_payload_with_two_frames(
            image_minus_133ms=processed_image_minus_133ms,
            image_current=processed_image_current,
            history_states=history_states,
            prompt=self.task,
            steps=0,
            num_loop=self.num_loop,
        )
        
        return payload
    
    async def run(self):
        """运行主循环"""
        self.running = True
        step_count = 0
        last_loop_time = None
        
        try:
            # 启动键盘监听
            self._setup_keyboard_listener()
            
            # 等待一段时间让机器人稳定
            time.sleep(1.0)
            
            # 记录初始状态（从数据集第一帧获取）
            initial_obs = self.get_observation_from_dataset()
            if initial_obs is None:
                print("[ERROR] 数据集为空，无法获取初始状态")
                await self._safe_shutdown()
                return
            self.initial_state = np.array(initial_obs['state'], dtype=np.float64)
            print(f"[INFO] 初始状态已记录: {self.initial_state}")
            
            # 连接服务器
            await self.ws_client.connect()
            
            print(f"开始回放数据集到服务器")
            print(f"服务器: {self.server_url}")
            print(f"任务: {self.task}")
            print(f"控制频率: {self.fps} Hz")
            
            # 第一轮：发送初始状态占位动作
            print("[INFO] 第一轮：发送初始状态占位动作")
            initial_action = {
                'shoulder_pan.pos': self.initial_state[0],
                'shoulder_lift.pos': self.initial_state[1],
                'elbow_flex.pos': self.initial_state[2],
                'wrist_flex.pos': self.initial_state[3],
                'wrist_roll.pos': self.initial_state[4],
                'gripper.pos': self.initial_state[5],
            }
            self.send_action(initial_action)
            await asyncio.sleep(self.dt)
            step_count = 1
            
            # 采集第一帧（执行占位动作后）
            observation = self.get_observation_from_dataset()
            if observation is None:
                print("[WARN] 数据集为空，无法开始")
                await self._safe_shutdown()
                return
            first_frame_time = time.time()
            self.frame_history.append({
                'state': observation['state'],
                'force': observation['force'],
                'images': observation['images'],
                'timestamp': first_frame_time,
                'dataset_frame_index': observation['dataset_frame_index']  # Debug: 记录帧索引
            })
            
            # 构建第一帧 payload 并发送，获取动作序列
            frame_current = self.frame_history[-1]
            frame_minus_133ms = self.frame_history[-1]  # 第一帧没有历史，使用同一帧
            
            payload = self.build_payload(frame_current, frame_minus_133ms)
            
            print("[INFO] 发送初始 payload，等待服务器返回动作序列...")
            response = await self.ws_client.send_and_receive(payload)
            actions = response['actions']
            
            if self.action_steps is not None and len(actions) > self.action_steps:
                actions = actions[:self.action_steps]
                print(f"[WARN] 动作数量限制: {len(response['actions'])} -> {self.action_steps}")
            
            # Debug: 比较服务器返回的action与数据集中的action
            if self.debug_mode:
                current_frame_idx = self.frame_history[-1]['dataset_frame_index']
                full_actions = response['actions']  # 完整的50个action (50, 24)
                print(f"\n{'='*80}")
                print(f"[DEBUG] ===== 初始Payload比较 =====")
                print(f"[DEBUG] 📤 发送Observation: 帧索引 {current_frame_idx}")
                print(f"[DEBUG] 📥 服务器返回完整action形状: {full_actions.shape}")
                print(f"[DEBUG] ⚠️  限制后执行action数量: {len(actions)}")
                print(f"[DEBUG] 📊 将比较完整的 {len(full_actions)} 个action (帧 {current_frame_idx} ~ {current_frame_idx + len(full_actions) - 1})")
                
                # 提取服务器action的前6维
                server_actions_6d = full_actions[:, :6]  # (50, 6)
                
                # 读取数据集对应帧的action
                num_compare = min(len(full_actions), self.episode_end - current_frame_idx)
                dataset_actions_list = []
                
                print(f"\n[DEBUG] 逐帧对比:")
                for i in range(num_compare):
                    dataset_action_frame = current_frame_idx + i
                    dataset_frame = self.dataset[dataset_action_frame]
                    dataset_action = dataset_frame['action']
                    if hasattr(dataset_action, "numpy"):
                        dataset_action = dataset_action.numpy()
                    dataset_action = np.array(dataset_action, dtype=np.float64)
                    dataset_actions_list.append(dataset_action)
                
                dataset_actions_array = np.array(dataset_actions_list)  # (N, 6)
                
                # 绘制对比图
                self.plot_action_comparison(
                    server_actions_6d[:num_compare],
                    dataset_actions_array,
                    current_frame_idx,
                    title="Initial Payload - Action Comparison"
                )
                
                print(f"{'='*80}\n")
            
            print(f"[INFO] 收到 {len(actions)} 个动作，开始主循环")
            
            # 主循环：执行动作 → 采集 → 构建 payload → 发送 → 获取下一轮动作
            while self.running:
                loop_start = time.time()
                
                # 检查关闭请求（最高优先级）
                if self._shutdown_requested.is_set():
                    print("检测到关闭请求")
                    await self._safe_shutdown()
                    return
                
                # 检查复位请求（高优先级）
                if self._reset_requested.is_set():
                    print("检测到复位请求")
                    self._reset_requested.clear()
                    
                    # 清空 actions 队列
                    actions = []
                    print("[INFO] 已清空 actions 队列")
                    
                    # 获取当前状态（从数据集最后一帧）
                    if self.last_observation is not None:
                        current_state = np.array(self.last_observation['state'], dtype=np.float64)
                    else:
                        print("[WARN] 没有历史观测，无法复位")
                        continue
                    
                    # 缓慢返回初始状态（平滑插值）
                    print("[INFO] 开始缓慢返回初始状态...")
                    await self._execute_reset(current_state)
                    continue
                
                # 检查恢复请求
                if self._resume_requested.is_set() and self.is_paused:
                    print("检测到恢复请求")
                    self._resume_requested.clear()
                    self.is_paused = False
                    
                    # 采集当前帧（从数据集）
                    observation = self.get_observation_from_dataset()
                    if observation is None:
                        print("[WARN] 数据集已用完，无法恢复")
                        await self._safe_shutdown()
                        return
                    resume_time = time.time()
                    self.frame_history.append({
                        'state': observation['state'],
                        'force': observation['force'],
                        'images': observation['images'],
                        'timestamp': resume_time,
                        'dataset_frame_index': observation['dataset_frame_index']  # Debug: 记录帧索引
                    })
                    print(f"[INFO] 恢复时采集帧: {resume_time:.3f}, 数据集帧索引: {observation['dataset_frame_index']}")
                    
                    # 使用同一帧作为当前帧和 -133ms 帧（类似程序刚开始）
                    frame_current = self.frame_history[-1]
                    frame_minus_133ms = self.frame_history[-1]
                    
                    # 构建 payload
                    payload = self.build_payload(frame_current, frame_minus_133ms)
                    
                    print("[INFO] 发送恢复 payload（两帧相同），等待服务器返回动作序列...")
                    response = await self.ws_client.send_and_receive(payload)
                    actions = response['actions']
                    
                    if self.action_steps is not None and len(actions) > self.action_steps:
                        actions = actions[:self.action_steps]
                        print(f"[WARN] 动作数量限制: {len(response['actions'])} -> {self.action_steps}")
                    
                    print(f"[INFO] 收到 {len(actions)} 个动作，恢复正常执行")
                    continue
                
                # 如果处于暂停状态，发送初始状态保持位置
                if self.is_paused:
                    if self.debug_mode:
                        print(f"[DEBUG] ⏸️  暂停中 - 发送Hold Action (对应初始state): {initial_action}")
                    self.send_action(initial_action)
                    await asyncio.sleep(self.dt)
                    continue
                
                # 计算与上一轮循环的间隔时间
                if last_loop_time is not None:
                    loop_interval = loop_start - last_loop_time
                else:
                    loop_interval = 0.0
                last_loop_time = loop_start
                
                # ========== 阶段 1: 执行动作序列（上一轮服务器返回的动作） ==========
                t1 = time.time()
                last_action_complete_time = None
                
                if len(actions) > 0:
                    for action_idx, action_to_execute in enumerate(actions):
                        action_start = time.time()
                        
                        # 检查复位请求（最高优先级，中断当前动作执行）
                        if self._reset_requested.is_set():
                            print("[WARN] 复位请求中断当前动作执行")
                            self._reset_requested.clear()
                            if self.last_observation is not None:
                                current_state = np.array(self.last_observation['state'], dtype=np.float64)
                                await self._execute_reset(current_state)
                            break
                        
                        # 检查关闭请求
                        if self._shutdown_requested.is_set():
                            await self._safe_shutdown()
                            return
                        
                        # 将 numpy array 转换为字典格式
                        if isinstance(action_to_execute, np.ndarray):
                            action_dict = {
                                'shoulder_pan.pos': action_to_execute[0],
                                'shoulder_lift.pos': action_to_execute[1],
                                'elbow_flex.pos': action_to_execute[2],
                                'wrist_flex.pos': action_to_execute[3],
                                'wrist_roll.pos': action_to_execute[4],
                                'gripper.pos': action_to_execute[5],
                            }
                        else:
                            action_dict = action_to_execute
                        
                        # Debug: 打印发送的action信息
                        if self.debug_mode and action_idx < 3:  # 只打印前3个，避免太多输出
                            if isinstance(action_to_execute, np.ndarray):
                                action_info = action_to_execute[:6]
                            else:
                                action_info = np.array([
                                    action_dict['shoulder_pan.pos'],
                                    action_dict['shoulder_lift.pos'],
                                    action_dict['elbow_flex.pos'],
                                    action_dict['wrist_flex.pos'],
                                    action_dict['wrist_roll.pos'],
                                    action_dict['gripper.pos'],
                                ])
                            print(f"[DEBUG] 🚀 发送Action[{action_idx + 1}] (来自服务器返回的第{action_idx + 1}个action): {action_info}")
                        
                        # 发送动作到机器人
                        if action_idx < len(actions) - 1:
                            # 非最后一帧：发送实际动作值
                            #  也不用发送，就保持原来的状态好了
                            self.send_action(actions[action_idx])
                            # self.send_action(self.initial_state)
                        else:
                            # self.send_action(self.initial_state)
                            # 最后一帧：发送当前采集到的 state，保持机械臂位置与 payload 的 0 state 一致
                            hold_action = observation['state']
                            if isinstance(hold_action, dict):
                                hold_action_dict = hold_action
                            else:
                                hold_action_dict = {
                                    'shoulder_pan.pos': hold_action[0],
                                    'shoulder_lift.pos': hold_action[1],
                                    'elbow_flex.pos': hold_action[2],
                                    'wrist_flex.pos': hold_action[3],
                                    'wrist_roll.pos': hold_action[4],
                                    'gripper.pos': hold_action[5],
                                }
                            self.send_action(hold_action_dict)
                            print(f"[DEBUG] 🛑 最后一帧发送Hold Action (数据集observation索引: {observation['dataset_frame_index']})")
                        
                        # 记录实际执行的action（用于累积对比图）
                        if self.debug_mode:
                            if isinstance(action_to_execute, np.ndarray):
                                action_6d = action_to_execute[:6].copy()
                            else:
                                action_6d = np.array([
                                    action_dict['shoulder_pan.pos'],
                                    action_dict['shoulder_lift.pos'],
                                    action_dict['elbow_flex.pos'],
                                    action_dict['wrist_flex.pos'],
                                    action_dict['wrist_roll.pos'],
                                    action_dict['gripper.pos'],
                                ])
                            self.executed_actions.append(action_6d)
                            # 记录这个action对应的数据集帧索引（当前采集的帧）
                            # 注意：action_idx是0-based，但我们要记录的是执行这个action后采集的帧
                            self.executed_action_frame_indices.append(observation['dataset_frame_index'] if 'observation' in dir() else self.current_frame_index - 1)
                        
                        # 保存最后一个 action
                        self.last_action = action_to_execute
                        
                        # 控制每个动作的执行频率（严格 30Hz）
                        action_elapsed = time.time() - action_start
                        action_sleep = self.dt - action_elapsed
                        if action_sleep > 0:
                            await asyncio.sleep(action_sleep)
                        
                        # 每个 action 执行完后都采集一帧（从数据集）
                        observation = self.get_observation_from_dataset()
                        if observation is None:
                            # 数据集已用完，触发关闭
                            print("[INFO] 数据集已用完，完成当前 action 后关闭")
                            # 先保存当前帧到历史记录（使用最后一帧）
                            if self.last_observation is not None:
                                self.frame_history.append({
                                    'state': self.last_observation['state'],
                                    'force': self.last_observation['force'],
                                    'images': self.last_observation['images'],
                                    'timestamp': time.time(),
                                    'dataset_frame_index': self.episode_end - 1
                                })
                            # 跳出 action 循环
                            break
                        self.frame_history.append({
                            'state': observation['state'],
                            'force': observation['force'],
                            'images': observation['images'],
                            'timestamp': time.time(),
                            'dataset_frame_index': observation['dataset_frame_index']  # Debug: 记录帧索引
                        })
                        
                        # 更新最后执行的action对应的的帧索引
                        if self.debug_mode and len(self.executed_action_frame_indices) > action_idx:
                            self.executed_action_frame_indices[-1] = observation['dataset_frame_index']
                        
                        # Debug: 打印采集的帧索引
                        if self.debug_mode and action_idx < 3:
                            print(f"[DEBUG] 📸 采集帧索引: {observation['dataset_frame_index']} (执行完Action[{action_idx + 1}]后)")
                        
                        step_count += 1
                    
                    # 记录最后一个 action 完成的时间
                    last_action_complete_time = time.time()
                    print(f"[INFO] 最后一个 action 完成时间: {last_action_complete_time:.3f}")
                
                # 检查是否因为数据集用完而触发了关闭
                if self._shutdown_requested.is_set():
                    print("[INFO] 数据集已用完，准备关闭")
                    await self._safe_shutdown()
                    return
                
                t_send = time.time() - t1
                
                # ========== 阶段 2: 选取 2 帧构建 payload ==========
                current_time = last_action_complete_time if last_action_complete_time else time.time()
                target_past_time = current_time - 0.166
                
                # 从后向前遍历队列
                frame_current = None
                frame_minus_133ms = None
                min_current_diff = float('inf')
                min_past_diff = float('inf')
                
                queue_len = len(self.frame_history)
                check_count = min(10, queue_len)
                
                for i in range(queue_len - 1, queue_len - check_count - 1, -1):
                    if i < 0:
                        break
                    
                    frame = self.frame_history[i]
                    ts = frame['timestamp']
                    
                    current_diff = abs(ts - current_time)
                    if current_diff < min_current_diff:
                        min_current_diff = current_diff
                        frame_current = frame
                    
                    past_diff = abs(ts - target_past_time)
                    if past_diff < min_past_diff:
                        min_past_diff = past_diff
                        frame_minus_133ms = frame
                    
                    if min_current_diff < 0.005 and min_past_diff < 0.005:
                        break
                
                # 验证选取结果
                if frame_current is None or frame_minus_133ms is None:
                    if queue_len >= 1:
                        print(f"⚠️  队列只有 {queue_len} 帧，使用同一帧")
                        frame_current = self.frame_history[-1]
                        frame_minus_133ms = self.frame_history[-1]
                    else:
                        print(f"⚠️  队列为空，跳过本轮")
                        await asyncio.sleep(self.dt)
                        continue
                
                actual_interval_ms = (frame_current['timestamp'] - frame_minus_133ms['timestamp']) * 1000
                dataset_status = "✅ 数据集" if self.current_frame_index < self.episode_end else "⚠️  已用完"
                
                # Debug: 打印选取的帧索引信息
                if self.debug_mode:
                    current_frame_idx = frame_current.get('dataset_frame_index', 'N/A')
                    past_frame_idx = frame_minus_133ms.get('dataset_frame_index', 'N/A')
                    print(f"[DEBUG] 选取帧: 当前帧={current_frame_idx}, -133ms帧={past_frame_idx}")
                
                print(
                    f"{dataset_status} | 队列 {queue_len} 帧 | "
                    f"检查 {check_count} 帧 | "
                    f"当前帧: {current_time:.3f} | "
                    f"当前帧误差: {min_current_diff*1000:.1f}ms | "
                    f"-133ms帧误差: {min_past_diff*1000:.1f}ms | "
                    f"实际间隔: {actual_interval_ms:.1f}ms"
                )
                
                # ========== 阶段 3: 构建 payload ==========
                t2 = time.time()
                payload = self.build_payload(frame_current, frame_minus_133ms)
                t_build = time.time() - t2
                
                # ========== 阶段 4: 发送请求并接收响应 ==========
                t3 = time.time()
                response = await self.ws_client.send_and_receive(payload)
                t_ws = time.time() - t3
                
                # 获取下一轮要执行的动作序列
                actions = response['actions']
                
                if self.action_steps is not None and len(actions) > self.action_steps:
                    actions = actions[:self.action_steps]
                    print(f"[WARN] 动作数量限制: {len(response['actions'])} -> {self.action_steps}")
                
                # Debug: 比较服务器返回的action与数据集中的action
                if self.debug_mode:
                    current_frame_idx = frame_current.get('dataset_frame_index', None)
                    if current_frame_idx is not None:
                        full_actions = response['actions']  # 完整的50个action (50, 24)
                        print(f"\n{'='*80}")
                        print(f"[DEBUG] ===== Step {step_count} Payload比较 =====")
                        print(f"[DEBUG] 📤 发送Observation: 帧索引 {current_frame_idx} (当前帧), {frame_minus_133ms.get('dataset_frame_index', 'N/A')} (-133ms帧)")
                        print(f"[DEBUG] 📥 服务器返回完整action形状: {full_actions.shape}")
                        print(f"[DEBUG] ⚠️  限制后执行action数量: {len(actions)}")
                        print(f"[DEBUG] 📊 将比较完整的 {len(full_actions)} 个action (帧 {current_frame_idx} ~ {current_frame_idx + len(full_actions) - 1})")
                        
                        # 提取服务器action的前6维
                        server_actions_6d = full_actions[:, :6]  # (50, 6)
                        
                        # 读取数据集对应帧的action
                        num_compare = min(len(full_actions), self.episode_end - current_frame_idx)
                        dataset_actions_list = []
                        
                        for i in range(num_compare):
                            dataset_action_frame = current_frame_idx + i
                            dataset_frame = self.dataset[dataset_action_frame]
                            dataset_action = dataset_frame['action']
                            if hasattr(dataset_action, "numpy"):
                                dataset_action = dataset_action.numpy()
                            dataset_action = np.array(dataset_action, dtype=np.float64)
                            dataset_actions_list.append(dataset_action)
                        
                        dataset_actions_array = np.array(dataset_actions_list)  # (N, 6)
                        
                        # 绘制对比图
                        self.plot_action_comparison(
                            server_actions_6d[:num_compare],
                            dataset_actions_array,
                            current_frame_idx,
                            title=f"Step {step_count} - Action Comparison"
                        )
                        
                        print(f"{'='*80}\n")
                
                # 打印本轮统计
                elapsed = time.time() - loop_start
                print(
                    f"Step {step_count} | "
                    f"循环间隔: {loop_interval*1000:.1f}ms | "
                    f"构建: {t_build*1000:.1f}ms | "
                    f"WS: {t_ws*1000:.1f}ms | "
                    f"执行: {t_send*1000:.1f}ms | "
                    f"总耗时: {elapsed*1000:.1f}ms"
                )
                
                # 每5次循环保存一次累积对比图
                if self.debug_mode and step_count % 5 == 0:
                    self.plot_cumulative_comparison(self.episode_start, self.episode_end)
                
        except KeyboardInterrupt:
            print("⚠️  用户中断执行")
        except Exception as e:
            logger.error(f"❌ 执行出错: {e}", exc_info=True)
        finally:
            await self.stop()
    
    async def stop(self):
        """停止"""
        print("🛑 开始安全关闭...")
        self.running = False
        
        # 保存最终的累积对比图
        if self.debug_mode and len(self.executed_actions) > 0:
            print("[DEBUG] 保存最终的累积对比图...")
            self.plot_cumulative_comparison(self.episode_start, self.episode_end)
        
        # 停止键盘监听
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        
        # 断开 WebSocket
        await self.ws_client.disconnect()
        print("WebSocket 已断开")
        
        # 断开机器人
        self.robot.disconnect()
        print("机器人已断开")
        
        print("✅ 安全关闭完成")


@draccus.wrap()
def main(cfg: ReplayConfig):
    """主函数"""
    replayer = DatasetReplayer(
        robot_config=cfg.robot,
        dataset_root=DATASET_ROOT,
        episode_index=EPISODE_INDEX,
        server_url=cfg.server_url,
        task=cfg.task,
        fps=cfg.fps,
        action_steps=cfg.action_steps,
        num_loop=cfg.num_loop,
    )
    
    asyncio.run(replayer.run())


if __name__ == "__main__":
    main()