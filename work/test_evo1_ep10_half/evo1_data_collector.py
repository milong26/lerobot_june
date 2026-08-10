"""
Evo-1 数据采集模块
功能：
- 连接机器人硬件
- 实时读取摄像头图像、关节状态
- 保存最新观测数据
- 异步保存历史数据到本地（跳帧策略，降低CPU压力）
"""

import numpy as np
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


class Evo1DataSaver:
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
        
        self.data_queue = queue.Queue(maxsize=100)
        
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.running = False
        
        self.frame_count = 0
        self.saved_count = 0
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.save_dir / self.timestamp
        self.session_dir.mkdir(exist_ok=True)
        
    def start(self):
        """启动后台保存线程"""
        self.running = True
        self.save_thread.start()
        print(f"[INFO] 数据保存目录: {self.session_dir}")
        
    def stop(self):
        """停止后台保存线程"""
        self.running = False
        self.save_thread.join(timeout=5)
        print(f"[INFO] 数据保存完成，共保存 {self.saved_count} 帧（处理 {self.frame_count} 帧）")
        
    def queue_data(self, observation: Dict[str, Any], action: Optional[np.ndarray] = None):
        """
        将数据加入保存队列（非阻塞）
        
        Args:
            observation: 观测数据
            action: 执行的动作（可选）
        """
        try:
            save_data = {
                'timestamp': np.array(observation.get('timestamp', time.time()), dtype=np.float64),
                'state': observation['state'].copy(),
                'frame_idx': self.frame_count
            }
            
            if self.save_images and 'images' in observation:
                save_data['images'] = {}
                for key, img in observation['images'].items():
                    if isinstance(img, np.ndarray):
                        save_data['images'][key] = img.copy()
                    else:
                        save_data['images'][key] = img
            
            if action is not None:
                save_data['action'] = action.copy()
            
            self.data_queue.put_nowait(save_data)
            
        except queue.Full:
            pass
        
        self.frame_count += 1
    
    def _save_worker(self):
        """后台保存工作线程"""
        while self.running:
            try:
                save_data = self.data_queue.get(timeout=1)
                
                frame_idx = save_data.pop('frame_idx')
                
                images = save_data.pop('images', None)
                
                npz_path = self.session_dir / f"frame_{frame_idx:06d}.npz"
                np.savez_compressed(npz_path, **save_data)
                
                if images:
                    img_path = self.session_dir / f"frame_{frame_idx:06d}_images.npz"
                    np.savez_compressed(img_path, **images)
                
                self.saved_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[WARN] 保存数据失败: {e}")


class Evo1DataCollector:
    """Evo-1 机器人数据采集器"""
    
    def __init__(self, robot, enable_recording: bool = False, recording_config: Dict[str, Any] = None, wowskin_sensor=None, wowskin_baseline=None):
        """
        Args:
            robot: LeRobot 机器人实例（已连接）
            enable_recording: 是否启用数据录制
            recording_config: 录制配置字典
            wowskin_sensor: WowSkin 力传感器实例
            wowskin_baseline: WowSkin baseline 数据
        """
        self.robot = robot
        
        # WowSkin 力传感器
        self.wowskin_sensor = wowskin_sensor
        self.wowskin_baseline = wowskin_baseline
        
        self.latest_observation = None
        
        self.data_saver = None
        if enable_recording:
            config = recording_config or {}
            self.data_saver = Evo1DataSaver(
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
                'images': {camera_name: np.ndarray},
                'state': np.ndarray,
                'force': np.ndarray,
                'timestamp': float
            }
        """
        raw_obs = self.robot.get_observation()
        
        images = {}
        for key in ['wrist', 'top']:
            if key in raw_obs:
                if isinstance(raw_obs[key], torch.Tensor):
                    images[key] = raw_obs[key].numpy()
                else:
                    images[key] = raw_obs[key]
        
        motor_keys = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 
                      'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
        state = np.array([raw_obs[k] for k in motor_keys], dtype=np.float32)
        
        # 力传感器数据（15维）
        force = np.zeros(15, dtype=np.float32)
        if self.wowskin_sensor is not None:
            try:
                wowskin_sample = self.wowskin_sensor.get_data(num_samples=1)[0][1:]
                if self.wowskin_baseline is not None:
                    wowskin_sample = wowskin_sample - self.wowskin_baseline
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
        
        self.latest_observation = observation
        
        return observation
    
    def get_latest_observation(self) -> Optional[Dict[str, Any]]:
        """
        获取最新观测数据
        
        Returns:
            最新观测数据字典，如果没有则返回 None
        """
        return self.latest_observation
    
    def get_only_state(self) -> Dict[str, float]:
        """
        仅获取当前关节状态（不获取图像和力传感器，速度更快）
        
        Returns:
            关节状态字典，格式: {'shoulder_pan.pos': float, ...}
        """
        return self.robot.get_only_state()
    
    def send_action(self, action) -> Dict[str, Any]:
        """
        发送动作到机器人执行
        
        Args:
            action: 动作数组（np.ndarray）或动作字典（Dict[str, float]）
            
        Returns:
            执行结果
        """
        # 判断输入类型
        if isinstance(action, dict):
            # 已经是字典格式，直接使用
            action_dict = action
        else:
            # 将 numpy array 转换为字典格式
            action_dict = {}
            action_features = self.robot.action_features
            
            for i, key in enumerate(action_features):
                if i < len(action):
                    action_dict[key] = float(action[i])
        
        result = self.robot.send_action(action_dict)
        
        if self.data_saver:
            if self.latest_observation is not None:
                latest_obs = {
                    'images': self.latest_observation['images'],
                    'state': self.latest_observation['state'],
                    'timestamp': time.time()
                }
                self.data_saver.queue_data(latest_obs, action)
        
        return result