"""
Evo-1 WebSocket 通信客户端
功能：
- 建立与 Evo-1 推理服务器的 WebSocket 连接
- 发送 JSON payload（两张独立的 320x240 图像）
- 接收服务器返回的动作序列
- 错误处理和自动重连
"""

import asyncio
import json
import numpy as np
import websockets
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class Evo1WSClient:
    """Evo-1 WebSocket 推理客户端"""
    
    def __init__(
        self,
        server_url: str = "ws://localhost:8001",
        max_size: int = 100_000_000,
        ping_interval: int = 30,
        ping_timeout: int = 120
    ):
        """
        Args:
            server_url: Evo-1 推理服务器地址
            max_size: 最大消息大小（字节）
            ping_interval: ping 间隔（秒）
            ping_timeout: ping 超时（秒）
        """
        self.server_url = server_url
        self.max_size = max_size
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        
        self.ws = None
        self.connected = False
    
    async def connect(self):
        """建立 WebSocket 连接"""
        try:
            self.ws = await websockets.connect(
                self.server_url,
                max_size=self.max_size,
                ping_interval=self.ping_interval,
                ping_timeout=self.ping_timeout
            )
            self.connected = True
            logger.info(f"✅ 已连接到 Evo-1 推理服务器: {self.server_url}")
        except Exception as e:
            logger.error(f"❌ 连接 Evo-1 服务器失败: {e}")
            raise
    
    async def disconnect(self):
        """断开 WebSocket 连接"""
        if self.ws and self.connected:
            await self.ws.close()
            self.connected = False
            logger.info("已断开与 Evo-1 服务器的连接")
    
    async def send_and_receive(self, payload: Dict[str, Any]) -> List:
        """
        发送请求并接收 Evo-1 服务器返回的动作序列
        
        Args:
            payload: 请求数据字典（包含 image, state, prompt, image_mask, action_mask）
            
        Returns:
            动作序列列表（List[List[float]]），shape: [num_steps, 24]
        """
        if not self.connected or self.ws is None:
            raise RuntimeError("未连接到 Evo-1 服务器，请先调用 connect()")
        
        try:
            payload_str = json.dumps(payload)
            
            await self.ws.send(payload_str)
            logger.debug("已发送 Evo-1 推理请求")
            
            response_str = await self.ws.recv()
            actions = json.loads(response_str)
            
            
            return actions
            
        except Exception as e:
            logger.error(f"Evo-1 发送/接收失败: {e}")
            self.connected = False
            raise
    
    async def reconnect(self):
        """重新连接"""
        logger.info("尝试重新连接到 Evo-1 服务器...")
        await self.disconnect()
        await self.connect()