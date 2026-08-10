"""
Evo-1 数据处理模块
功能：
- 图像处理：将两张图像分别调整为 320x240（不拼接）
- 状态处理：将 state 填充到 24 维
- 构建 Evo-1 服务器要求的 payload 格式
"""

import numpy as np
from typing import Dict, List, Any, Optional
import cv2


class Evo1DataProcessor:
    """Evo-1 数据处理和 payload 构建"""
    
    def __init__(self):
        pass
    
    def process_images_separate(
        self, 
        images: Dict[str, np.ndarray]
    ) -> List[List]:
        """
        处理图像数据（Evo-1 格式）：
        - 分别调整 top 和 wrist 图像到 320x240
        - 不拼接，保持两张独立的图像
        - 转换为 uint8 列表格式
        
        Args:
            images: {camera_name: np.ndarray} 图像字典
            
        Returns:
            包含两张图像的列表，每张图像是 uint8 列表格式
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
            raise ValueError("top 图像不在图像字典中，无法处理")
        if wrist_img is None:
            raise ValueError("wrist 图像不在图像字典中，无法处理")
        
        top_img = self._resize_image(top_img, (320, 240))
        wrist_img = self._resize_image(wrist_img, (320, 240))
        
        processed_images.append(top_img.astype(np.uint8).tolist())
        processed_images.append(wrist_img.astype(np.uint8).tolist())
        
        return processed_images
    
    def _resize_image(self, img: np.ndarray, target_size: tuple) -> np.ndarray:
        """
        调整图像分辨率（使用 OpenCV resize）
        
        Args:
            img: 原始图像 (H, W, C)
            target_size: (width, height)
            
        Returns:
            降采样后的图像
        """
        return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    def process_state(self, state: np.ndarray) -> List[float]:
        """
        处理状态数据：
        - 直接返回原始的 6 维 state
        - 服务器端会自动填充到 24 维
        
        Args:
            state: 原始关节状态（6 维）
            
        Returns:
            6 维状态列表
        """
        return state.tolist()
    
    def build_evo1_payload(
        self,
        images: List[List],
        state: List[float],
        prompt: str,
        image_mask: List[int] = None,
        action_mask: List[int] = None
    ) -> Dict[str, Any]:
        """
        构建 Evo-1 服务器要求的 payload 格式
        
        Args:
            images: 两张处理后的图像列表 [img1, img2]
            state: 24 维状态列表
            prompt: 任务描述
            image_mask: 图像掩码，默认 [1, 1, 0]
            action_mask: 动作掩码，默认前 6 维为 1，其余为 0
            
        Returns:
            完整的 Evo-1 payload 字典
        """
        if image_mask is None:
            image_mask = [1, 1, 0]
        
        if action_mask is None:
            action_mask = [1, 1, 1, 1, 1, 1] + [0] * 18
        
        payload = {
            "image": images,
            "state": state,
            "prompt": prompt,
            "image_mask": image_mask,
            "action_mask": action_mask
        }
        
        return payload
    
    def build_evo1_payload_with_two_frames(
        self,
        images_current: Dict[str, np.ndarray],
        state_current: np.ndarray,
        prompt: str,
        image_mask: List[int] = None,
        action_mask: List[int] = None
    ) -> Dict[str, Any]:
        """
        使用两帧数据构建 Evo-1 payload
        
        
        Args:
            images_minus_133ms: -133ms 时刻的图像字典
            images_current: 当前时刻的图像字典
            state_minus_133ms: -133ms 时刻的状态
            state_current: 当前时刻的状态
            prompt: 任务描述
            image_mask: 图像掩码
            action_mask: 动作掩码
            
        Returns:
            完整的 Evo-1 payload 字典
        """
        processed_images_current = self.process_images_separate(images_current)
        # processed_images_current 是一个列表：[top_img, wrist_img]
        # process_images_separate 保证顺序：索引 0 = top, 索引 1 = wrist
        
        # 服务器要求 3 张图像，使用：
        images = [
            processed_images_current[0],  # top 图像（有效，mask=1）
            processed_images_current[1],  # wrist 图像（有效，mask=1）
            processed_images_current[0],  # 占位图（无效，mask=0，复用 top 图）
        ]
        
        state = self.process_state(state_current)
        
        return self.build_evo1_payload(
            images=images,
            state=state,
            prompt=prompt,
            image_mask=image_mask,
            action_mask=action_mask
        )