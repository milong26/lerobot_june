"""
数据处理模块
功能：
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


class DataProcessor:
    """数据处理和 payload 构建"""
    
    def __init__(self):
        pass
        
    def process_images(self, images: Dict[str, np.ndarray]) -> List[List]:
        """
        处理图像数据：
        - 分辨率减半（640x480 → 320x240）
        - 拼接：top视角（左）+ wrist视角（右）
        - 转换为 uint8 列表格式
        """
        processed_images = []
        
        top_img = None
        wrist_img = None
        
        for key, img in images.items():
            if 'top' in key.lower():
                top_img = img
            elif 'wrist' in key.lower():
                wrist_img = img
        
        if top_img is None:
            raise ValueError("top图不在图像中，无法处理")
        if wrist_img is None:
            raise ValueError("wrist图不在图像中，无法处理")
        
        top_img = self._resize_image(top_img, (320, 240))
        wrist_img = self._resize_image(wrist_img, (320, 240))
        
        combined_img = np.hstack([top_img, wrist_img])
        
        processed_images.append(combined_img.astype(np.uint8).tolist())
        
        return processed_images
    
    def _resize_image(self, img: np.ndarray, target_size: tuple) -> np.ndarray:
        h, w = img.shape[:2]
        target_w, target_h = target_size
        
        h_step = max(1, h // target_h)
        w_step = max(1, w // target_w)
        
        return img[::h_step, ::w_step, :]
    
    def process_state(self, state: np.ndarray, force: np.ndarray) -> np.ndarray:
        """
        处理状态数据：
        - state(6维) + force(15维) = 21维
        """
        combined = np.concatenate([state, force])
        return combined
    
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
        使用 2 帧数据构建 payload
        """
        processed_images = image_minus_133ms + image_current
        
        if history_states and len(history_states) >= 2:
            state_minus_133ms = self.process_state(history_states[0]['state'], history_states[0]['force'])
            state_current = self.process_state(history_states[1]['state'], history_states[1]['force'])
        else:
            raise ValueError("history_states 必须包含 2 帧数据")
        
        action_minus_4 = np.zeros(24, dtype=np.float32)
        action_current = np.zeros(24, dtype=np.float32)
        
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