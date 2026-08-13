"""
数据采集模块
功能：
- 连接机器人硬件
- 实时读取摄像头图像、关节状态、力传感器数据
- 保存最新观测数据
"""

import numpy as np
from typing import Dict, Optional, Any
import torch
import threading
import time
import logging

logger = logging.getLogger(__name__)


class DataCollector:
    """机器人数据采集器"""
    
    def __init__(self, robot, wowskin_sensor=None, wowskin_baseline=None):
        self.robot = robot
        self._robot_lock = threading.Lock()
        self.wowskin_sensor = wowskin_sensor
        self.wowskin_baseline = wowskin_baseline
        self.latest_observation = None
        
    def get_only_state(self) -> Dict[str, float]:
        with self._robot_lock:
            return self.robot.get_only_state()
    
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
        with self._robot_lock:
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
    
    def send_action(self, action) -> Dict[str, Any]:
        """
        发送动作到机器人执行
        
        Args:
            action: 动作数组（np.ndarray）或动作字典（Dict[str, float]）
            
        Returns:
            执行结果
        """
        if isinstance(action, dict):
            action_dict = action
        else:
            action_dict = {}
            action_features = self.robot.action_features
            
            for i, key in enumerate(action_features):
                if i < len(action):
                    action_dict[key] = float(action[i])
        
        with self._robot_lock:
            result = self.robot.send_action(action_dict)
        
        return result