"""
数据处理模块
功能：
- 加载 norm_stats.json 归一化统计信息
- 实现反归一化：将机器人的原始数据转换为模型需要的格式
- 维度填充：state(6维)→24维，force(15维)→24维
- 图像编码：转换为 uint8 列表格式（用于 JSON 传输）
- 图像拼接：top视角（左）+ wrist视角（右），分辨率减半（320x240）
- 构建 payload：按照服务器要求的格式组装数据
"""

import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
from collections import deque


# ============================================================
# 以下归一化/反归一化代码已注释掉
# 原因：实时控制时从机器人获取的是原始物理值，服务器期望的也是原始值
# 不需要进行归一化/反归一化操作
# 保留代码仅供参考，后续如需使用可取消注释
# ============================================================

# class Normalizer:
#     """归一化/反归一化工具类"""
#     
#     def __init__(self, stats_or_path):
#         """
#         Args:
#             stats_or_path: norm_stats.json 路径或统计信息字典
#         """
#         if isinstance(stats_or_path, str):
#             with open(stats_or_path, "r") as f:
#                 stats = json.load(f)
#         else:
#             stats = stats_or_path
#         
#         def pad_to_24(x):
#             """填充到24维"""
#             x = torch.tensor(x, dtype=torch.float32)
#             if x.shape[0] < 24:
#                 pad = torch.zeros(24 - x.shape[0], dtype=torch.float32)
#                 x = torch.cat([x, pad], dim=0)
#             elif x.shape[0] > 24:
#                 raise ValueError(f"Input length {x.shape[0]} exceeds expected 24")
#             return x
#         
#         if len(stats) != 1:
#             raise ValueError(f"norm_stats.json should contain only one robot key, but: {list(stats.keys())}")
#         
#         robot_key = list(stats.keys())[0]
#         robot_stats = stats[robot_key]
#         
#         self.state_min = pad_to_24(robot_stats["observation.state"]["min"])
#         self.state_max = pad_to_24(robot_stats["observation.state"]["max"])
#         self.force_min = pad_to_24(robot_stats["observation.force"]["min"])
#         self.force_max = pad_to_24(robot_stats["observation.force"]["max"])
#         self.action_min = pad_to_24(robot_stats["action"]["min"])
#         self.action_max = pad_to_24(robot_stats["action"]["max"])
#     
#     def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
#         """归一化 state"""
#         state_min = self.state_min.to(state.device, dtype=state.dtype)
#         state_max = self.state_max.to(state.device, dtype=state.dtype)
#         return torch.clamp(2 * (state - state_min) / (state_max - state_min + 1e-8) - 1, -1.0, 1.0)
#     
#     def normalize_force(self, force: torch.Tensor) -> torch.Tensor:
#         """归一化 force"""
#         force_min = self.force_min.to(force.device, dtype=force.dtype)
#         force_max = self.force_max.to(force.device, dtype=force.dtype)
#         return torch.clamp(2 * (force - force_min) / (force_max - force_min + 1e-8) - 1, -1.0, 1.0)
#     
#     def denormalize_state(self, state: torch.Tensor) -> torch.Tensor:
#         """反归一化 state"""
#         state_min = self.state_min.to(state.device, dtype=state.dtype)
#         state_max = self.state_max.to(state.device, dtype=state.dtype)
#         if state.ndim == 1:
#             state = state.view(1, -1)
#         return (state + 1.0) / 2.0 * (state_max - state_min + 1e-8) + state_min
#     
#     def denormalize_force(self, force: torch.Tensor) -> torch.Tensor:
#         """反归一化 force"""
#         force_min = self.force_min.to(force.device, dtype=force.dtype)
#         force_max = self.force_max.to(force.device, dtype=force.dtype)
#         if force.ndim == 1:
#             force = force.view(1, -1)
#         return (force + 1.0) / 2.0 * (force_max - force_min + 1e-8) + force_min
#     
#     def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
#         """反归一化 action"""
#         action_min = self.action_min.to(action.device, dtype=action.dtype)
#         action_max = self.action_max.to(action.device, dtype=action.dtype)
#         if action.ndim == 1:
#             action = action.view(1, -1)
#         return (action + 1.0) / 2.0 * (action_max - action_min + 1e-8) + action_min


class DataProcessor:
    """数据处理和 payload 构建"""
    
    def __init__(self):
        """
        Args:
            norm_stats_path: norm_stats.json 文件路径（当前未使用，保留接口兼容）
        """
        # self.normalizer = Normalizer(norm_stats_path)  # 已注释，实时控制不需要归一化
        
    def process_images(self, images: Dict[str, np.ndarray]) -> List[List]:
        """
        处理图像数据：
        - 分辨率减半（640x480 → 320x240）
        - 拼接：top视角（左）+ wrist视角（右）
        - 转换为 uint8 列表格式
        
        Args:
            images: {camera_name: np.ndarray} 图像字典
            
        Returns:
            拼接后的图像列表（uint8）
        """
        processed_images = []
        
        # 获取 top 和 wrist 图像
        top_img = None
        wrist_img = None
        
        for key, img in images.items():
            if 'top' in key.lower():
                top_img = img
            elif 'wrist' in key.lower():
                wrist_img = img
        
        # 如果找不到对应图像，使用占位符
        if top_img is None:
            raise ValueError("top图不在图像中，无法处理")
        if wrist_img is None:
            raise ValueError("wrist图不在图像中，无法处理")
        
        # 分辨率减半
        top_img = self._resize_image(top_img, (320, 240))
        wrist_img = self._resize_image(wrist_img, (320, 240))
        
        # 水平拼接：top（左）+ wrist（右）
        combined_img = np.hstack([top_img, wrist_img])  # 640x240
        
        # 转换为 uint8 列表
        # TODO: 这个转换有必要吗？
        processed_images.append(combined_img.astype(np.uint8).tolist())
        
        return processed_images
    
    def _resize_image(self, img: np.ndarray, target_size: tuple) -> np.ndarray:
        """
        调整图像分辨率（使用简单的切片降采样）
        
        Args:
            img: 原始图像 (H, W, C)
            target_size: (width, height)
            
        Returns:
            降采样后的图像
        """
        h, w = img.shape[:2]
        target_w, target_h = target_size
        
        # 简单的间隔采样
        h_step = max(1, h // target_h)
        w_step = max(1, w // target_w)
        
        return img[::h_step, ::w_step, :]
    
    def process_state(self, state: np.ndarray, force: np.ndarray) -> np.ndarray:
        """
        处理状态数据：
        - state(6维) + force(15维) = 21维
        - 直接返回原始值（不归一化/反归一化）
        
        Args:
            state: 6维关节状态
            force: 15维力传感器数据
            
        Returns:
            21维状态数组（原始物理值）
        """
        # 拼接 state 和 force
        combined = np.concatenate([state, force])  # 21维
        
        # 调试
        # print(f"process_state: state shape={state.shape}, force shape={force.shape}, combined shape={combined.shape}")
        
        return combined
    
    def build_payload_with_history(
        self,
        image_history: List,
        history_states: List[Dict],
        prompt: str,
        history_actions: Optional[List[np.ndarray]] = None,
        steps: int = 11, 
        seed: int = 22,
        g_scale: float = 1.0,
        num_loop: int = 2
    ) -> Dict[str, Any]:
        """
        使用已处理的历史数据构建 payload（供后台采集模式使用）
        
        Args:
            image_history: 已处理的图像历史列表（索引 0 = -4 step，索引 4 = 0 step）
            history_states: 历史状态列表，每个元素包含 {'state', 'force', 'timestamp'}
            prompt: 任务描述
            history_actions: 历史动作列表
            steps: 推理步数
            seed: 随机种子
            g_scale: 引导缩放系数
            num_loop: 循环次数
            
        Returns:
            完整的 payload 字典
        """
        # 构建历史图像窗口：取 -4 step 和 0 step（共2帧）
        # image_history 索引 0 = -4 step，索引 4 = 0 step
        if len(image_history) >= 5:
            image_minus_4 = [image_history[0]]
            image_current = [image_history[4]]
        else:
            print(f"  [WARN] 图像历史长度不足5帧，使用最后一帧填充")
            # 历史不足，用最后一帧填充
            fallback_img = image_history[-1] if image_history else None
            if fallback_img is None:
                raise ValueError("图像历史为空，无法构建 payload")
            image_minus_4 = [fallback_img]
            image_current = [fallback_img]
        
        processed_images = image_minus_4 + image_current
        
        # 构建历史状态窗口：取 -4 step 和 0 step
        # history_states 索引 0 = -4 step，索引 4 = 0 step
        if history_states and len(history_states) >= 5:
            state_minus_4 = self.process_state(history_states[0]['state'], history_states[0]['force'])
            state_current = self.process_state(history_states[4]['state'], history_states[4]['force'])
        else:
            # 历史不足，用最后一帧填充
            fallback = history_states[-1] if history_states else None
            if fallback is None:
                raise ValueError("状态历史为空，无法构建 payload")
            state_minus_4 = self.process_state(fallback['state'], fallback['force'])
            state_current = state_minus_4
        
        # 默认动作（零动作，24维）
        action_minus_4 = np.zeros(24, dtype=np.float32)
        action_current = np.zeros(24, dtype=np.float32)
        
        # 构建 payload
        payload = {
            "image": processed_images,
            "state": [state_minus_4.tolist(), state_current.tolist()],
            "action": [action_minus_4.tolist(), action_current.tolist()],
            "prompt": prompt,
            "steps": steps,
            "seed": seed,
            "g_scale": g_scale,
            "video_name": prompt,
            "image_mask": [1, 1, 0],
            "action_mask": [1, 1, 1, 1, 1, 1] + [0] * 18,
            "num_loop": num_loop
        }
        
        return payload
    
    def build_payload_with_two_frames(
        self,
        image_minus_133ms: List,
        image_current: List,
        history_states: List[Dict],
        prompt: str,
        steps: int = 11, 
        seed: int = 22,
        g_scale: float = 1.0,
        num_loop: int = 2
    ) -> Dict[str, Any]:
        """
        使用 2 帧数据构建 payload（当前帧和 -133ms 帧）
        
        所有数据均来自后台采集缓冲区，不会调用 collector.get_observation()
        
        Args:
            image_minus_133ms: 已处理的 -133ms 图像
            image_current: 已处理的当前图像
            history_states: 历史状态列表 [frame_minus_133ms, frame_current]
            prompt: 任务描述
            steps: 推理步数
            seed: 随机种子
            g_scale: 引导缩放系数
            num_loop: 循环次数
            
        Returns:
            完整的 payload 字典
        """
        # 合并 2 张图像
        processed_images = image_minus_133ms + image_current
        
        # 构建历史状态：取 -133ms 和当前帧
        if history_states and len(history_states) >= 2:
            state_minus_133ms = self.process_state(history_states[0]['state'], history_states[0]['force'])
            state_current = self.process_state(history_states[1]['state'], history_states[1]['force'])
        else:
            raise ValueError("history_states 必须包含 2 帧数据")
        
        # 默认动作（零动作，24维）
        action_minus_4 = np.zeros(24, dtype=np.float32)
        action_current = np.zeros(24, dtype=np.float32)
        
        # 构建 payload
        payload = {
            "image": processed_images,
            "state": [state_minus_133ms.tolist(), state_current.tolist()],
            "action": [action_minus_4.tolist(), action_current.tolist()],
            "prompt": prompt,
            "steps": steps,
            "seed": seed,
            "g_scale": g_scale,
            "video_name": prompt,
            "image_mask": [1, 1, 0],
            "action_mask": [1, 1, 1, 1, 1, 1] + [0] * 18,
            "num_loop": num_loop
        }
        
        return payload