"""
主控制循环
功能：
- 整合 data_collector、data_processor、ws_client 三个模块
- 实现主循环：采集→处理→发送→接收→执行
- 控制执行频率（30Hz）
- 将服务器返回的动作发送给机器人执行
- 异常处理和优雅退出

执行示例：
python main_controller.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.cameras.wrist.type=opencv \
    --robot.cameras.wrist.index_or_path=6 \
    --robot.cameras.wrist.width=640 \
    --robot.cameras.wrist.height=480 \
    --robot.cameras.wrist.fps=30 \
    --robot.cameras.wrist.fourcc=MJPG \
    --robot.cameras.top.type=intelrealsense \
    --robot.cameras.top.serial_number_or_name=806312060427 \
    --robot.cameras.top.width=640 \
    --robot.cameras.top.height=480 \
    --robot.cameras.top.fps=30 \
    --robot.cameras.top.use_depth=False \
    --robot.id=start_new_heihei_2 \
    --task="Grab the cross-shape equipment." \
    --server_url=ws://10.10.16.19:9000 \
    --norm_stats_path=./norm_stats.json \
    --fps=30
"""

import asyncio
import time
import logging
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from collections import deque
import numpy as np
import draccus

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_collector import DataCollector
from data_processor import DataProcessor
from ws_client import WSClient

# LeRobot 导入
from lerobot.robots import make_robot_from_config
from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401

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

# WowSkin 导入
try:
    from anyskin import AnySkinProcess
    HAS_WOWSKIN = True
except ImportError:
    HAS_WOWSKIN = False
    logger.warning("anyskin 未安装，WowSkin 力传感器将不可用")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WowSkinConfig:
    """WowSkin 力传感器配置"""
    enabled: bool = False
    port: str = ""
    num_mags: int = 5
    temp_filtered: bool = True


@dataclass
class RecordingConfig:
    """录制配置"""
    enable_recording: bool = False
    record_dir: str = "./recorded_data"
    skip_frames: int = 3
    save_images: bool = False


@dataclass
class ControllerConfig:
    """主控制器配置"""
    robot: RobotConfig
    task: str
    server_url: str = "ws://10.10.16.19:9000"
    norm_stats_path: str = "./norm_stats.json"
    fps: int = 30
    max_steps: Optional[int] = None
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    wowskin: WowSkinConfig = field(default_factory=WowSkinConfig)


class MainController:
    """主控制器"""
    
    def __init__(
        self,
        robot_config: RobotConfig,
        task: str,
        server_url: str,
        norm_stats_path: str,
        fps: int = 30,
        history_size: int = 5,
        enable_recording: bool = False,
        recording_config: Dict[str, Any] = None,
        wowskin_config: Optional[WowSkinConfig] = None
    ):
        """
        Args:
            robot_config: 机器人配置对象
            task: 任务描述
            server_url: 推理服务器地址
            norm_stats_path: 归一化统计文件路径
            fps: 控制频率
            history_size: 历史数据窗口大小
            enable_recording: 是否启用数据录制
            recording_config: 录制配置字典
            wowskin_config: WowSkin 力传感器配置
        """
        self.task = task
        self.fps = fps
        self.dt = 1.0 / fps
        
        # 初始化 WowSkin 力传感器
        self.wowskin_sensor = None
        self.wowskin_baseline = None
        if wowskin_config and wowskin_config.enabled and HAS_WOWSKIN:
            logger.info(f"正在初始化 WowSkin 力传感器 (port={wowskin_config.port})...")
            self.wowskin_sensor = AnySkinProcess(
                num_mags=wowskin_config.num_mags,
                port=wowskin_config.port,
            )
            self.wowskin_sensor.start()
            time.sleep(1.0)  # 等待传感器初始化
            
            # 采集 baseline
            logger.info("正在采集 WowSkin baseline...")
            baseline_data = self.wowskin_sensor.get_data(num_samples=20)
            baseline_data = np.array(baseline_data)[:, 1:]  # 跳过时间戳
            self.wowskin_baseline = np.mean(baseline_data, axis=0)
            logger.info(f"✅ WowSkin baseline: {self.wowskin_baseline}")
        
        # 初始化机器人
        logger.info("正在连接机器人...")
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()
        logger.info("✅ 机器人已连接")
        
        # 初始化数据采集器（带录制功能）
        self.collector = DataCollector(
            self.robot,
            history_size=history_size,
            enable_recording=enable_recording,
            recording_config=recording_config,
            wowskin_sensor=self.wowskin_sensor,
            wowskin_baseline=self.wowskin_baseline
        )
        
        # 初始化数据处理器
        self.processor = DataProcessor(norm_stats_path, history_size=history_size)
        
        # 初始化 WebSocket 客户端
        self.ws_client = WSClient(server_url)
        
        # 历史状态缓冲区（使用 deque 自动管理大小，O(1) 操作）
        self.state_history = deque(maxlen=5)
        
        # 控制标志
        self.running = False
    
    async def run(self, max_steps: Optional[int] = None):
        """
        运行主控制循环
        
        Args:
            max_steps: 最大执行步数（None 表示无限循环）
        """
        self.running = True
        step_count = 0
        
        try:
            # 启动数据录制（如果启用）
            self.collector.start_recording()
            
            # 连接服务器
            await self.ws_client.connect()
            
            logger.info(f"🚀 开始执行任务: {self.task}")
            logger.info(f"控制频率: {self.fps} Hz")
            
            while self.running:
                loop_start = time.time()
                
                # 1. 采集数据
                observation = self.collector.get_observation()
                
                # 2. 更新历史状态（deque 自动限制大小为 5，无需手动截断）
                self.state_history.append({
                    'state': observation['state'],
                    'force': observation['force']
                })
                
                # 3. 构建 payload
                payload = self.processor.build_payload(
                    images=observation['images'],
                    state=observation['state'],
                    force=observation['force'],
                    prompt=self.task,
                    history_states=[
                        (h['state'], h['force']) for h in self.state_history
                    ],
                    history_actions=self.processor.get_action_history()
                )
                
                # 4. 发送请求并接收响应
                response = await self.ws_client.send_and_receive(payload)
                print(response)
                
                # TODO: 本地接收到 action 后的处理
                actions = response['actions']
                states = response['states']
                
                # 解析状态（前6维是state，6-21维是force）
                pred_state = states[:, :6]
                pred_force = states[:, 6:21]
                
                # 5. 执行动作
                if len(actions) > 0:
                    # 执行第一个动作
                    action_to_execute = actions[0]
                    
                    # TODO: 发送动作到机器人
                    self.collector.send_action(action_to_execute)
                    
                    # 更新动作历史
                    self.processor.update_action_history(action_to_execute)
                
                # 6. 控制频率
                elapsed = time.time() - loop_start
                sleep_time = self.dt - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
                step_count += 1
                logger.info(f"Step {step_count} | 耗时: {elapsed*1000:.1f}ms")
                
                # 检查是否达到最大步数
                if max_steps and step_count >= max_steps:
                    logger.info(f"已达到最大步数 {max_steps}，停止执行")
                    break
                
        except KeyboardInterrupt:
            logger.info("⚠️  用户中断执行")
        except Exception as e:
            logger.error(f"❌ 执行出错: {e}", exc_info=True)
        finally:
            await self.stop()
    
    async def stop(self):
        """停止控制器"""
        self.running = False
        
        # 停止数据录制
        self.collector.stop_recording()
        
        # 停止 WowSkin 传感器
        if self.wowskin_sensor is not None:
            logger.info("正在停止 WowSkin 传感器...")
            self.wowskin_sensor.pause_streaming()
            self.wowskin_sensor.join()
        
        await self.ws_client.disconnect()
        self.robot.disconnect()
        logger.info("🛑 控制器已停止")


@draccus.wrap()
def main(cfg: ControllerConfig):
    """主函数"""
    # 构建录制配置
    recording_config = {
        'save_dir': cfg.recording.record_dir,
        'skip_frames': cfg.recording.skip_frames,
        'save_images': cfg.recording.save_images
    }
    
    # 创建控制器
    controller = MainController(
        robot_config=cfg.robot,
        task=cfg.task,
        server_url=cfg.server_url,
        norm_stats_path=cfg.norm_stats_path,
        fps=cfg.fps,
        enable_recording=cfg.recording.enable_recording,
        recording_config=recording_config,
        wowskin_config=cfg.wowskin
    )
    
    # 运行
    asyncio.run(controller.run(max_steps=cfg.max_steps))


if __name__ == "__main__":
    main()