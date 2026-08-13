"""
WebSocket 通信客户端
功能：
- 建立与推理服务器的 WebSocket 连接
- 发送 JSON payload
- 接收服务器返回的推理结果
- 解析响应：提取 act（动作）和 sta（状态）
- 错误处理和自动重连
"""

import asyncio
import json
import numpy as np
import websockets
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class WSClient:
    """WebSocket 推理客户端"""
    
    def __init__(
        self,
        server_url: str = "ws://10.10.16.19:9000",
        max_size: int = 100_000_000,
        ping_interval: int = 30,
        ping_timeout: int = 120
    ):
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
            logger.info(f"✅ 已连接到推理服务器: {self.server_url}")
        except Exception as e:
            logger.error(f"❌ 连接服务器失败: {e}")
            raise
    
    async def disconnect(self):
        """断开 WebSocket 连接"""
        if self.ws and self.connected:
            await self.ws.close()
            self.connected = False
            logger.info("已断开与服务器的连接")
    
    async def send_and_receive(self, payload: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        发送请求并接收响应
        
        Args:
            payload: 请求数据字典
            
        Returns:
            {
                'actions': np.ndarray,  # 预测动作
                'states': np.ndarray    # 预测状态
            }
        """
        if not self.connected or self.ws is None:
            raise RuntimeError("未连接到服务器，请先调用 connect()")
        
        try:
            # 发送 JSON payload
            await self.ws.send(json.dumps(payload))
            logger.debug("已发送推理请求")
            
            # 接收响应
            response_str = await self.ws.recv()
            response = json.loads(response_str)
            
            # 接收服务器返回的 action 和 state
            actions = np.asarray(response['act'], dtype=np.float32)
            states = np.asarray(response['sta'], dtype=np.float32)
            
            logger.debug(f"收到推理结果 - actions shape: {actions.shape}, states shape: {states.shape}")
            
            return {
                'actions': actions,
                'states': states
            }
            
        except Exception as e:
            logger.error(f"发送/接收失败: {e}")
            raise