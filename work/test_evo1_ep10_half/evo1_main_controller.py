"""
Evo-1 主控制循环
- 适配 Evo-1 服务器（两张独立的 320x240 图像）
- 整合 evo1_data_collector、evo1_data_processor、evo1_ws_client 三个模块
- 实现主循环：采集→处理→发送→接收→执行
- 控制执行频率（30Hz）
- 将服务器返回的动作发送给机器人执行
- 异常处理和退出

执行示例：
python evo1_main_controller.py \
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
    --server_url=ws://10.10.16.19:8001 \
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
import threading
from pynput import keyboard

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evo1_data_collector import Evo1DataCollector
from evo1_data_processor import Evo1DataProcessor
from evo1_ws_client import Evo1WSClient

from lerobot.robots import make_robot_from_config
from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.cameras.realsense import RealSenseCameraConfig

from lerobot.robots import (
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
    save_images: bool = False


@dataclass
class Evo1ControllerConfig:
    """Evo-1 主控制器配置"""
    robot: RobotConfig
    task: str
    server_url: str = "ws://localhost:8001"
    fps: int = 30
    max_steps: Optional[int] = None
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    wowskin: WowSkinConfig = field(default_factory=WowSkinConfig)
    action_steps: Optional[int] = None


class Evo1MainController:
    """Evo-1 主控制器"""
    
    def __init__(
        self,
        robot_config: RobotConfig,
        task: str,
        server_url: str,
        fps: int = 30,
        enable_recording: bool = False,
        recording_config: Dict[str, Any] = None,
        wowskin_config: Optional[WowSkinConfig] = None,
        action_steps: Optional[int] = None,
    ):
        """
        Args:
            robot_config: 机器人配置对象
            task: 任务描述
            server_url: Evo-1 推理服务器地址
            fps: 控制频率
            enable_recording: 是否启用数据录制
            recording_config: 录制配置字典
            wowskin_config: WowSkin 力传感器配置
            action_steps: 动作步数限制
        """
        self.task = task
        self.server_url = server_url
        self.fps = fps
        self.dt = 1.0 / fps
        self.action_steps = action_steps
        
        # 初始化 WowSkin 力传感器
        self.wowskin_sensor = None
        self.wowskin_baseline = None
        if wowskin_config and wowskin_config.enabled and HAS_WOWSKIN:
            print(f"正在初始化 WowSkin 力传感器 (port={wowskin_config.port})...")
            self.wowskin_sensor = AnySkinProcess(
                num_mags=wowskin_config.num_mags,
                port=wowskin_config.port,
            )
            self.wowskin_sensor.start()
            time.sleep(2.0)  # 等待传感器初始化
            
            # 采集 baseline
            print("正在采集 WowSkin baseline...")
            baseline_data = self.wowskin_sensor.get_data(num_samples=20)
            baseline_data = np.array(baseline_data)[:, 1:]  # 跳过时间戳
            self.wowskin_baseline = np.mean(baseline_data, axis=0)
            print(f"[INFO] WowSkin baseline: {self.wowskin_baseline}")
        
        print("[INFO] 正在连接机器人...")
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()
        print("[INFO] 机器人已连接")
        
        self.collector = Evo1DataCollector(
            self.robot,
            enable_recording=enable_recording,
            recording_config=recording_config,
            wowskin_sensor=self.wowskin_sensor,
            wowskin_baseline=self.wowskin_baseline,
        )
        
        self.processor = Evo1DataProcessor()
        
        self.ws_client = Evo1WSClient(server_url)
        
        self.frame_history = deque(maxlen=20)
        
        self._last_action_completion_time = None
        
        self.running = False
        self._cleanup_done = False
        
        self.initial_state = None
        self.is_resetting = False
        self.is_paused = False
        self.keyboard_listener = None
        
        self._reset_requested = threading.Event()
        self._shutdown_requested = threading.Event()
        self._resume_requested = threading.Event()
        
        self._last_obs_refresh_time = 0.0  # 用于暂停期间定期刷新相机观测
    
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
        if self.initial_state is None:
            logger.warning("[WARN] 未设置初始状态，无法复位")
            return
        
        self.is_resetting = True
        diff = self.initial_state - current_state
        max_diff = np.max(np.abs(diff))
        
        steps = max(10, int(max_diff / 1.5))
        
        print(f"复位需要 {steps} 步，最大差值: {max_diff:.3f}")
        
        for step in range(1, steps + 1):
            if not self.running:
                break
            
            alpha = step / steps
            target_state = current_state + alpha * diff
            
            self.collector.send_action(target_state.tolist())
            
            await asyncio.sleep(self.dt)
        
        print("[INFO] 复位完成")
        self.is_resetting = False
        self.is_paused = True
        print("[INFO] 已暂停服务器通信，按左键恢复")
    
    async def _safe_shutdown(self):
        """安全关闭程序"""
        self.running = False
        
        if self.keyboard_listener:
            self.keyboard_listener.stop()
    
        self.collector.stop_recording()
        print("数据录制已停止")
        
        # 停止 WowSkin 传感器
        if self.wowskin_sensor is not None:
            print("正在停止 WowSkin 传感器...")
            self.wowskin_sensor.pause_streaming()
            self.wowskin_sensor.join()
            print("WowSkin 传感器已停止")
        
        await self.ws_client.disconnect()
        print("WebSocket 已断开")
        
        self.robot.disconnect()
        print("机器人已断开")
        
        print("[INFO] 安全关闭完成")
    
    async def run(self, max_steps: Optional[int] = None):
        """
        运行 Evo-1 主控制循环
        
        Args:
            max_steps: 最大执行步数（None 表示无限循环）
        """
        self.running = True
        step_count = 0
        last_loop_time = None
        
        try:
            self._setup_keyboard_listener()
            
            time.sleep(1.0)
            
            initial_obs = self.collector.get_observation()
            self.initial_state = np.array(initial_obs['state'], dtype=np.float64)
            print(f"[INFO] 初始状态已记录: {self.initial_state}")
            
            self.collector.start_recording()
            
            await self.ws_client.connect()
            
            print(f"开始执行 Evo-1 任务: {self.task}")
            print(f"控制频率: {self.fps} Hz")
            print(f"Evo-1 服务器: {self.server_url}")
            
            print("[INFO] 第一轮：发送初始状态占位动作")
            self.collector.send_action(self.initial_state.tolist())
            await asyncio.sleep(self.dt)
            step_count = 1
            
            observation = self.collector.get_observation()
            first_frame_time = time.time()
            self.frame_history.append({
                'state': observation['state'],
                'images': observation['images'],
                'timestamp': first_frame_time
            })
            print(f"[INFO] 第一帧采集时间: {first_frame_time:.3f}")
            
            frame_current = self.frame_history[-1]
            
            payload = self.processor.build_evo1_payload_with_two_frames(
                images_current=frame_current['images'],
                state_current=frame_current['state'],
                prompt=self.task,
            )
            
            print("[INFO] 发送初始 payload，等待 Evo-1 服务器返回动作序列...")
            actions = await self.ws_client.send_and_receive(payload)
            
            if self.action_steps is not None and len(actions) > self.action_steps:
                actions = actions[:self.action_steps]
                print(f"[WARN] 动作数量限制: {len(actions)} -> {self.action_steps}")
            
            print(f"[INFO] 收到 {len(actions)} 个动作，开始主循环")
            
            while self.running:
                loop_start = time.time()
                
                if self._shutdown_requested.is_set():
                    print("检测到关闭请求")
                    await self._safe_shutdown()
                    return
                
                if self._reset_requested.is_set():
                    print("检测到复位请求")
                    self._reset_requested.clear()
                    
                    actions = []
                    print("[INFO] 已清空 actions 队列")
                    
                    current_obs = self.collector.get_observation()
                    current_state = np.array(current_obs['state'], dtype=np.float64)
                    
                    print("[INFO] 开始缓慢返回初始状态...")
                    diff = self.initial_state - current_state
                    max_diff = np.max(np.abs(diff))
                    
                    steps = max(10, int(max_diff / 1.5))
                    print(f"复位需要 {steps} 步，最大差值: {max_diff:.3f}")
                    
                    for step in range(1, steps + 1):
                        if not self.running:
                            break
                        
                        alpha = step / steps
                        target_state = current_state + alpha * diff
                        
                        self.collector.send_action(target_state.tolist())
                        
                        await asyncio.sleep(self.dt)
                    
                    print("[INFO] 已返回初始状态")
                    
                    self.is_paused = True
                    print("[INFO] 已进入等待状态，按左键恢复")
                    continue
                
                if self._resume_requested.is_set() and self.is_paused:
                    print("检测到恢复请求")
                    self._resume_requested.clear()
                    self.is_paused = False
                    
                    observation = self.collector.get_observation()
                    resume_time = time.time()
                    self.frame_history.append({
                        'state': observation['state'],
                        'images': observation['images'],
                        'timestamp': resume_time
                    })
                    print(f"[INFO] 恢复时采集帧: {resume_time:.3f}")
                    
                    frame_current = self.frame_history[-1]
                    
                    payload = self.processor.build_evo1_payload_with_two_frames(
                        images_current=frame_current['images'],
                        state_current=frame_current['state'],
                        prompt=self.task,
                    )
                    
                    print("[INFO] 发送恢复 payload，等待 Evo-1 服务器返回动作序列...")
                    actions = await self.ws_client.send_and_receive(payload)
                    
                    if self.action_steps is not None and len(actions) > self.action_steps:
                        actions = actions[:self.action_steps]
                        print(f"[WARN] 动作数量限制: {len(actions)} -> {self.action_steps}")
                    
                    print(f"[INFO] 收到 {len(actions)} 个动作，恢复正常执行")
                    continue
                
                if self.is_paused:
                    self.collector.send_action(self.initial_state.tolist())
                    
                    # 每 3 秒调用一次 get_observation 刷新相机，防止相机等待时间过长
                    current_time = time.time()
                    if current_time - self._last_obs_refresh_time >= 3.0:
                        self.collector.get_observation()
                        self._last_obs_refresh_time = current_time
                    
                    await asyncio.sleep(self.dt)
                    continue
                
                if last_loop_time is not None:
                    loop_interval = loop_start - last_loop_time
                else:
                    loop_interval = 0.0
                last_loop_time = loop_start
                
                t1 = time.time()
                
                if len(actions) > 0:
                    for action_idx, action_to_execute in enumerate(actions):
                        action_start = time.time()
                        
                        if self._reset_requested.is_set():
                            print("[WARN] 复位请求中断当前动作执行")
                            self._reset_requested.clear()
                            current_obs = self.collector.get_observation()
                            current_state = np.array(current_obs['state'], dtype=np.float64)
                            await self._execute_reset(current_state)
                            break
                        
                        if self._shutdown_requested.is_set():
                            print("检测到关闭请求")
                            await self._safe_shutdown()
                            return
                        
                        action_dict = action_to_execute
                        
                        # 发送动作到机器人
                        if action_idx < len(actions) - 1:
                            # 非最后一帧：发送实际动作值
                            self.collector.send_action(action_dict)
                        else:
                            # 最后一帧：立即读取当前 state 并发送，让机械臂在移动中立即停止
                            # 原理：目标位置 = 当前位置，电机停止移动
                            current_obs = self.collector.get_only_state()
                            self.collector.send_action(current_obs)
                            print(f"最后一帧：立即停止机械臂 | state: {current_obs}")
                        
                        # self.collector.send_action(action_dict)  # 旧代码：直接发送所有动作
                        
                        action_elapsed = time.time() - action_start
                        action_sleep = self.dt - action_elapsed
                        if action_sleep > 0:
                            await asyncio.sleep(action_sleep)
                        
                        observation = self.collector.get_observation()
                        self.frame_history.append({
                            'state': observation['state'],
                            'images': observation['images'],
                            'timestamp': time.time()
                        })
                        
                        step_count += 1
                    
                    last_action_complete_time = time.time()
                
                t_send = time.time() - t1
                
                current_time = last_action_complete_time if last_action_complete_time else time.time()
                
                frame_current = None
                min_current_diff = float('inf')
                min_past_diff = float('inf')
                
                queue_len = len(self.frame_history)
                check_count = min(15, queue_len)
                
                for i in range(queue_len - 1, queue_len - check_count - 1, -1):
                    if i < 0:
                        break
                    
                    frame = self.frame_history[i]
                    ts = frame['timestamp']
                    
                    current_diff = abs(ts - current_time)
                    if current_diff < min_current_diff:
                        min_current_diff = current_diff
                        frame_current = frame
                    
                    
                    if min_current_diff < 0.005 and min_past_diff < 0.005:
                        break
                
                
                
                t2 = time.time()
                
                payload = self.processor.build_evo1_payload_with_two_frames(
                    images_current=frame_current['images'],
                    state_current=frame_current['state'],
                    prompt=self.task,
                )
                t_build = time.time() - t2
                
                t3 = time.time()
                actions = await self.ws_client.send_and_receive(payload)
                t_ws = time.time() - t3
                
                if self.action_steps is not None and len(actions) > self.action_steps:
                    actions = actions[:self.action_steps]
                    print(f"[WARN] 动作数量限制: {len(actions)} -> {self.action_steps}")
                
                elapsed = time.time() - loop_start
                print(
                    f"Step {step_count} | "
                    f"循环间隔: {loop_interval*1000:.1f}ms | "
                    f"构建: {t_build*1000:.1f}ms | "
                    f"WS: {t_ws*1000:.1f}ms | "
                    f"执行: {t_send*1000:.1f}ms | "
                    f"总耗时: {elapsed*1000:.1f}ms"
                )
                
        except KeyboardInterrupt:
            print("[WARN] 用户中断执行")
        except Exception as e:
            logger.error(f"[ERROR] 执行出错: {e}", exc_info=True)
        finally:
            await self.stop()
    
    async def stop(self):
        """停止控制器"""
        if self._cleanup_done:
            return
        
        self._cleanup_done = True
        self.running = False
        
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            print("键盘监听器已停止")
        
        self.collector.stop_recording()
        
        # 停止 WowSkin 传感器
        if self.wowskin_sensor is not None:
            print("正在停止 WowSkin 传感器...")
            self.wowskin_sensor.pause_streaming()
            self.wowskin_sensor.join()
        
        await self.ws_client.disconnect()
        
        try:
            if hasattr(self.robot, 'is_connected') and self.robot.is_connected:
                self.robot.disconnect()
                print("机器人已断开")
            else:
                self.robot.disconnect()
                print("机器人已断开")
        except Exception as e:
            logger.warning(f"断开机器人连接时出现警告: {e}")
        
        print("Evo-1 控制器已停止")


@draccus.wrap()
def main(cfg: Evo1ControllerConfig):
    """主函数"""
    recording_config = {
        'save_dir': cfg.recording.record_dir,
        'save_images': cfg.recording.save_images
    }
    
    controller = Evo1MainController(
        robot_config=cfg.robot,
        task=cfg.task,
        server_url=cfg.server_url,
        fps=cfg.fps,
        enable_recording=cfg.recording.enable_recording,
        recording_config=recording_config,
        wowskin_config=cfg.wowskin,
        action_steps=cfg.action_steps,
    )
    
    asyncio.run(controller.run())


if __name__ == "__main__":
    main()