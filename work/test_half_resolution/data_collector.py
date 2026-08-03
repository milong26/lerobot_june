"""
数据采集模块
功能：
- 连接机器人硬件
- 实时读取摄像头图像、关节状态、力传感器数据
- 维护历史数据缓冲区
- 异步保存历史数据到本地（跳帧策略，降低CPU压力）
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Any
import torch
import threading
import queue
import time
import os
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class DataSaver:
    """异步数据保存器（后台线程，不阻塞主循环）"""
    
    def __init__(
        self,
        save_dir: str = "./recorded_data",
        save_images: bool = True
    ):
        """
        Args:
            save_dir: 保存目录
            save_images: 是否保存图像（占用空间大，可选择关闭）
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_images = save_images
        
        # 线程安全队列
        self.data_queue = queue.Queue(maxsize=100)  # 限制队列大小防止内存溢出
        
        # 后台保存线程
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.running = False
        
        # 帧计数器
        self.frame_count = 0
        self.saved_count = 0  # 实际保存成功的帧数
        
        # 创建时间戳子目录
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.save_dir / self.timestamp
        self.session_dir.mkdir(exist_ok=True)
        
    def start(self):
        """启动后台保存线程"""
        self.running = True
        self.save_thread.start()
        print(f"📁 数据保存目录: {self.session_dir}")
        
    def stop(self):
        """停止后台保存线程"""
        self.running = False
        self.save_thread.join(timeout=5)
        print(f"✅ 数据保存完成，共保存 {self.saved_count} 帧（处理 {self.frame_count} 帧）")
        
    def queue_data(self, observation: Dict[str, Any], action: Optional[np.ndarray] = None):
        """
        将数据加入保存队列（非阻塞）
        
        Args:
            observation: 观测数据
            action: 执行的动作（可选）
        """
        try:

            # 准备保存的数据（轻量级）
            save_data = {
                'timestamp': np.array(observation.get('timestamp', time.time()), dtype=np.float64),  # 转为 numpy 数组
                'state': observation['state'].copy(),  # 6维
                'force': observation['force'].copy(),  # 15维
                'frame_idx': self.frame_count
            }
            
            # 可选保存图像（保存原始尺寸，与 payload 一致）
            if self.save_images and 'images' in observation:
                save_data['images'] = {}
                for key, img in observation['images'].items():
                    if isinstance(img, np.ndarray):
                        save_data['images'][key] = img.copy()  # 保存原始图像
                    else:
                        save_data['images'][key] = img
            
            # 保存动作
            if action is not None:
                save_data['action'] = action.copy()
            
            # 非阻塞入队
            self.data_queue.put_nowait(save_data)
            
        except queue.Full:
            # 队列满了，丢弃当前帧（不影响主循环）
            pass
        
        self.frame_count += 1
    
    def _save_worker(self):
        """后台保存工作线程"""
        while self.running:
            try:
                # 从队列获取数据（超时1秒）
                save_data = self.data_queue.get(timeout=1)
                
                # 保存为 npz 格式（轻量、快速）
                frame_idx = save_data.pop('frame_idx')
                
                # 分离图像和其他数据
                images = save_data.pop('images', None)
                
                # 保存 state/force/action 到 npz
                npz_path = self.session_dir / f"frame_{frame_idx:06d}.npz"
                np.savez_compressed(npz_path, **save_data)
                
                # 保存图像为 npz（如果启用）
                if images:
                    img_path = self.session_dir / f"frame_{frame_idx:06d}_images.npz"
                    np.savez_compressed(img_path, **images)  # 将字典展开为多个数组保存
                
                self.saved_count += 1  # 保存成功计数
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ 保存数据失败: {e}")


class DataCollector:
    """机器人数据采集器"""
    
    def __init__(self, robot, history_size: int = 5, enable_recording: bool = False, recording_config: Dict[str, Any] = None, wowskin_sensor=None, wowskin_baseline=None):
        """
        Args:
            robot: LeRobot 机器人实例（已连接）
            history_size: 历史数据缓冲区大小（默认5帧）
            enable_recording: 是否启用数据录制
            recording_config: 录制配置字典
            wowskin_sensor: WowSkin 力传感器实例
            wowskin_baseline: WowSkin baseline 数据
        """
        self.robot = robot
        self.history_size = history_size
        
        # WowSkin 力传感器
        self.wowskin_sensor = wowskin_sensor
        self.wowskin_baseline = wowskin_baseline
        
        # 历史数据缓冲区
        self.image_history = deque(maxlen=history_size)
        self.state_history = deque(maxlen=history_size)
        self.force_history = deque(maxlen=history_size)
        
        # 数据保存器
        self.data_saver = None
        if enable_recording:
            config = recording_config or {}
            self.data_saver = DataSaver(
                save_dir=config.get('save_dir', './recorded_data'),
                save_images=config.get('save_images', True)
            )
        
    def start_recording(self):
        """启动数据录制"""
        if self.data_saver:
            self.data_saver.start()
            
    def stop_recording(self):
        """停止数据录制"""
        if self.data_saver:
            self.data_saver.stop()
            
    def get_observation(self) -> Dict[str, Any]:
        """
        获取当前观测数据
        
        Returns:
            {
                'images': {camera_name: np.ndarray},  # 图像字典
                'state': np.ndarray,  # 6维关节状态
                'force': np.ndarray,  # 15维力传感器数据
                'timestamp': float
            }
        """
        # 从机器人获取原始观测
        raw_obs = self.robot.get_observation()
        
        # 提取图像数据（直接使用 'wrist' 和 'top' 键）
        images = {}
        for key in ['wrist', 'top']:
            if key in raw_obs:
                if isinstance(raw_obs[key], torch.Tensor):
                    images[key] = raw_obs[key].numpy()
                else:
                    images[key] = raw_obs[key]
        
        # 提取关节状态（6个电机的位置，保持原始顺序）
        # SO100 返回的键格式: "{motor}.pos"
        motor_keys = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 
                      'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
        state = np.array([raw_obs[k] for k in motor_keys], dtype=np.float32)
        
        # 力传感器数据（15维）
        # 如果有 wowskin 传感器，从 wowskin 中获取
        force = np.zeros(15, dtype=np.float32)
        if self.wowskin_sensor is not None:
            try:
                # 从 wowskin 获取最新数据
                wowskin_sample = self.wowskin_sensor.get_data(num_samples=1)[0][1:]
                if self.wowskin_baseline is not None:
                    wowskin_sample = wowskin_sample - self.wowskin_baseline
                # 取前15维（5个磁传感器 × 3轴 = 15维）
                force = np.array(wowskin_sample[:15], dtype=np.float32)
            except Exception as e:
                logger.warning(f"获取 WowSkin 数据失败: {e}")
        
        logger.debug(f"获取观测: state shape={state.shape}, force shape={force.shape}")
        
        observation = {
            'images': images,
            'state': state,
            'force': force,
            'timestamp': raw_obs.get('timestamp', 0.0)
        }
        
        # 更新历史缓冲区
        self._update_history(observation)
        
        return observation
    
    def _update_history(self, observation: Dict[str, Any]):
        """更新历史数据缓冲区"""
        self.image_history.append(observation['images'])
        self.state_history.append(observation['state'])
        self.force_history.append(observation['force'])
    
    def get_history(self) -> Dict[str, list]:
        """
        获取历史数据
        
        Returns:
            {
                'images': list of image dicts,
                'states': list of state arrays,
                'forces': list of force arrays
            }
        """
        return {
            'images': list(self.image_history),
            'states': list(self.state_history),
            'forces': list(self.force_history)
        }
    
    def send_action(self, action: np.ndarray) -> Dict[str, Any]:
        """
        发送动作到机器人执行
        
        Args:
            action: 动作数组（21维：6维state + 15维force）
            
        Returns:
            执行结果
        """
        # 将动作转换为字典格式
        action_dict = {}
        action_features = self.robot.action_features
        
        for i, key in enumerate(action_features):
            if i < len(action):
                action_dict[key] = float(action[i])
        
        # 发送动作
        result = self.robot.send_action(action_dict)
        
        # 录制数据（异步，不阻塞）
        if self.data_saver:
            # 获取最新观测
            latest_obs = {
                'images': self.image_history[-1] if self.image_history else {},
                'state': self.state_history[-1] if self.state_history else np.zeros(6, dtype=np.float32),
                'force': self.force_history[-1] if self.force_history else np.zeros(15, dtype=np.float32),
                'timestamp': time.time()
            }
            self.data_saver.queue_data(latest_obs, action)
        
        return result