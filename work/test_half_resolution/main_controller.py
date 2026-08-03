"""
主控制循环
- 整合 data_collector、data_processor、ws_client 三个模块
- 实现主循环：采集→处理→发送→接收→执行
- 控制执行频率（30Hz）
- 将服务器返回的动作发送给机器人执行
- 异常处理和退出

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
    save_images: bool = False


@dataclass
class ControllerConfig:
    """主控制器配置"""
    robot: RobotConfig
    task: str
    server_url: str = "ws://10.10.16.19:9000"
    fps: int = 30
    max_steps: Optional[int] = None
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    wowskin: WowSkinConfig = field(default_factory=WowSkinConfig)
    num_loop: int = 5
    action_steps: Optional[int] = None


class MainController:
    """主控制器"""
    
    def __init__(
        self,
        robot_config: RobotConfig,
        task: str,
        server_url: str,
        fps: int = 30,
        history_size: int = 5,
        enable_recording: bool = False,
        recording_config: Dict[str, Any] = None,
        wowskin_config: Optional[WowSkinConfig] = None,
        num_loop: int = 5,
        action_steps: Optional[int] = None,
    ):
        """
        Args:
            robot_config: 机器人配置对象
            task: 任务描述
            server_url: 推理服务器地址
            fps: 控制频率
            history_size: 历史数据窗口大小
            enable_recording: 是否启用数据录制
            recording_config: 录制配置字典
            wowskin_config: WowSkin 力传感器配置
        """
        self.task = task
        self.fps = fps
        self.dt = 1.0 / fps
        self.num_loop = num_loop
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
            print(f"✅ WowSkin baseline: {self.wowskin_baseline}")
        
        # 初始化机器人
        print("正在连接机器人...")
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()
        print("机器人已连接")
        
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
        self.processor = DataProcessor(history_size=history_size)
        
        # 初始化 WebSocket 客户端
        self.ws_client = WSClient(server_url)        
        # 后台采集线程
        self._collection_thread = None
        
        # 非阻塞采集队列（带时间戳）
        # 存储格式：{'state', 'force', 'images', 'timestamp'}
        self.collection_queue = deque(maxlen=15)  # 保存最近 15 帧（约 0.5 秒）
        
        # 采集线程控制标志
        self._collection_running = False
        
        # 控制标志
        self.running = False
        self._cleanup_done = False  # 防止重复清理
        
        # 键盘控制状态
        self.initial_state = None  # 复位目标状态
        self.is_resetting = False  # 是否正在执行复位
        self.is_paused = False  # 是否暂停服务器通信（复位后）
        self.keyboard_listener = None
        
        # 线程安全的事件标志
        self._reset_requested = threading.Event()
        self._shutdown_requested = threading.Event()
        self._resume_requested = threading.Event()
    
    def _setup_keyboard_listener(self):
        """设置键盘监听器"""
        def on_press(key):
            try:
                if key == keyboard.Key.right:
                    print("⌨️  检测到右键 → 请求复位")
                    self._reset_requested.set()
                elif key == keyboard.Key.down:
                    print("⌨️  检测到下键 → 请求安全关闭")
                    self._shutdown_requested.set()
                elif key == keyboard.Key.left:
                    print("⌨️  检测到左键 → 请求恢复")
                    self._resume_requested.set()
            except AttributeError:
                pass
        
        self.keyboard_listener = keyboard.Listener(on_press=on_press)
        self.keyboard_listener.start()
        print("✅ 键盘监听器已启动 (右键=复位, 左键=恢复, 下键=关闭)")
    
    async def _execute_reset(self, current_state):
        """
        执行复位操作：从当前状态平滑移动到初始状态
        
        Args:
            current_state: 当前关节角度
        """
        if self.initial_state is None:
            logger.warning("⚠️  未设置初始状态，无法复位")
            return
        
        self.is_resetting = True
        print("🔄 开始复位操作...")
        
        # 计算插值步数（平滑过渡，避免跳变）
        diff = self.initial_state - current_state
        max_diff = np.max(np.abs(diff))
        
        # 根据最大差值计算步数，每步最多移动 0.15 弧度（可根据需要调整）
        steps = max(10, int(max_diff / 1.5))
        
        print(f"复位需要 {steps} 步，最大差值: {max_diff:.3f}")
        
        for step in range(1, steps + 1):
            if not self.running:
                break
            
            # 线性插值
            alpha = step / steps
            target_state = current_state + alpha * diff
            
            # 发送插值后的状态
            self.collector.send_action(target_state.tolist())
            
            # 控制频率
            await asyncio.sleep(self.dt)
            
            # if step % 10 == 0 or step == steps:
            #     print(f"复位进度: {step}/{steps} ({alpha*100:.1f}%)")
        
        print("✅ 复位完成")
        self.is_resetting = False
        self.is_paused = True  # 复位后暂停服务器通信
        print("⏸️  已暂停服务器通信，按左键恢复")
    
    async def _safe_shutdown(self):
        """安全关闭程序"""
        print("🛑 开始安全关闭...")
        self.running = False
        
        # 停止键盘监听
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            print("键盘监听器已停止")
        
        # 停止数据录制
        self.collector.stop_recording()
        print("数据录制已停止")
        
        # 停止 WowSkin 传感器
        if self.wowskin_sensor is not None:
            print("正在停止 WowSkin 传感器...")
            self.wowskin_sensor.pause_streaming()
            self.wowskin_sensor.join()
            print("WowSkin 传感器已停止")
        
        # 断开 WebSocket
        await self.ws_client.disconnect()
        print("WebSocket 已断开")
        
        # 断开机器人
        self.robot.disconnect()
        print("机器人已断开")
        
        print("✅ 安全关闭完成")
    
    def _continuous_collection_thread(self):
        """
        后台持续采集线程
        以 30Hz 频率持续采集，将数据存入 collection_queue
        """
        print("🎥 后台采集线程已启动 (30Hz)")
        while self._collection_running and self.running:
            frame_start = time.time()
            
            try:
                observation = self.collector.get_observation()
                timestamp = time.time()
                
                # 存入队列（带时间戳）
                frame_data = {
                    'state': observation['state'],
                    'force': observation['force'],
                    'images': observation['images'],
                    'timestamp': timestamp
                }
                self.collection_queue.append(frame_data)
                
            except Exception as e:
                logger.warning(f"后台采集出错: {e}")
            
            elapsed = time.time() - frame_start
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        print("🎥 后台采集线程已停止")
    
    async def run(self, max_steps: Optional[int] = None):
        """
        运行主控制循环
        
        设计思路：
        1. 启动后台采集线程（持续 30Hz 采集，完全不阻塞主循环）
        2. 后台线程负责：采集 → 记录时间戳 → 放入 collection_queue
        3. 主循环：从队列选取当前帧和 -133ms 帧 → 构建 payload → 发送 → 执行
        4. 所有 payload 数据均来自缓冲区，不会再次调用 collector.get_observation()
        
        Args:
            max_steps: 最大执行步数（None 表示无限循环）
        """
        self.running = True
        step_count = 0
        last_loop_time = None
        
        try:
            # 启动键盘监听
            self._setup_keyboard_listener()
            
            # 等待一段时间让机器人稳定
            time.sleep(1.0)
            
            # 记录初始状态（复位目标）
            initial_obs = self.collector.get_observation()
            self.initial_state = np.array(initial_obs['state'], dtype=np.float64)
            print(f"✅ 初始状态已记录: {self.initial_state}")
            
            # 启动数据录制（如果启用）
            self.collector.start_recording()
            
            # 连接服务器
            await self.ws_client.connect()
            
            # 启动后台持续采集线程
            self._collection_running = True
            self._collection_thread = threading.Thread(
                target=self._continuous_collection_thread,
                daemon=True
            )
            self._collection_thread.start()
            
            print(f"开始执行任务: {self.task}")
            print(f"控制频率: {self.fps} Hz")
            print(f"采集策略: 后台持续采集 (30Hz)，主循环从队列选取当前帧和 -133ms 帧")
            
            while self.running:
                # 检查关闭请求（最高优先级）
                if self._shutdown_requested.is_set():
                    print("检测到关闭请求")
                    await self._safe_shutdown()
                    return
                
                # 检查复位请求（高优先级）
                if self._reset_requested.is_set():
                    print("检测到复位请求")
                    self._reset_requested.clear()
                    
                    # 获取当前状态
                    current_obs = self.collector.get_observation()
                    current_state = np.array(current_obs['state'], dtype=np.float64)
                    
                    # 执行复位
                    await self._execute_reset(current_state)
                    
                    # 复位完成后继续循环（会进入暂停状态）
                    continue
                
                # 检查恢复请求
                if self._resume_requested.is_set() and self.is_paused:
                    print("检测到恢复请求")
                    self._resume_requested.clear()
                    self.is_paused = False
                    print("▶️  已恢复服务器通信")
                
                # 如果处于暂停状态，发送初始状态保持位置
                if self.is_paused:
                    loop_start = time.time()
                    self.collector.send_action(self.initial_state.tolist())
                    await asyncio.sleep(self.dt)
                    elapsed = time.time() - loop_start
                    print(f"⏸️  暂停中 | 保持初始状态 | 耗时: {elapsed*1000:.1f}ms")
                    continue
                
                loop_start = time.time()
                
                # 计算与上一轮循环的间隔时间
                if last_loop_time is not None:
                    loop_interval = loop_start - last_loop_time
                else:
                    loop_interval = 0.0
                last_loop_time = loop_start
                
                # ========== 阶段 1: 从后台采集队列选取 2 帧（从后向前，高效） ==========
                t1 = time.time()
                
                # 当前时间戳
                current_time = time.time()
                target_past_time = current_time - 0.133  # 133ms 前的时间点
                
                # 从后向前遍历队列（最新帧在末尾）
                # 只需检查最后 10 帧（30Hz 下约 333ms，足够覆盖 133ms）
                frame_current = None
                frame_minus_133ms = None
                min_current_diff = float('inf')
                min_past_diff = float('inf')
                
                # 从队列末尾向前遍历（最多检查 10 帧）
                queue_len = len(self.collection_queue)
                check_count = min(10, queue_len)
                
                for i in range(queue_len - 1, queue_len - check_count - 1, -1):
                    if i < 0:
                        break
                    
                    frame = self.collection_queue[i]
                    ts = frame['timestamp']
                    
                    # 查找最接近当前的帧
                    current_diff = abs(ts - current_time)
                    if current_diff < min_current_diff:
                        min_current_diff = current_diff
                        frame_current = frame
                    
                    # 查找最接近 133ms 前的帧
                    past_diff = abs(ts - target_past_time)
                    if past_diff < min_past_diff:
                        min_past_diff = past_diff
                        frame_minus_133ms = frame
                    
                    # 优化：如果已经找到足够接近的帧，提前退出
                    if min_current_diff < 0.005 and min_past_diff < 0.005:  # 5ms 以内
                        break
                
                # 验证选取结果
                if frame_current is None or frame_minus_133ms is None:
                    queue_len = len(self.collection_queue)
                    print(f"⚠️  队列数据不足: {queue_len} 帧，跳过本轮")
                    continue
                
                actual_interval_ms = (frame_current['timestamp'] - frame_minus_133ms['timestamp']) * 1000
                print(
                    f"⏱️  非阻塞采集: 队列 {queue_len} 帧 | "
                    f"检查 {check_count} 帧 | "
                    f"当前帧误差: {min_current_diff*1000:.1f}ms | "
                    f"-133ms帧误差: {min_past_diff*1000:.1f}ms | "
                    f"实际间隔: {actual_interval_ms:.1f}ms"
                )
                
                t_collect = time.time() - t1
                
                # ========== 阶段 2: 构建 payload（仅使用缓冲区数据） ==========
                t2 = time.time()
                
                # 处理 2 帧图像（-133ms 和当前）
                processed_image_minus_133ms = self.processor.process_images(frame_minus_133ms['images'])
                processed_image_current = self.processor.process_images(frame_current['images'])
                
                # 构建历史状态（仅 2 帧）
                history_states = [
                    {'state': frame_minus_133ms['state'], 'force': frame_minus_133ms['force']},
                    {'state': frame_current['state'], 'force': frame_current['force']}
                ]
                
                payload = self.processor.build_payload_with_two_frames(
                    image_minus_133ms=processed_image_minus_133ms,
                    image_current=processed_image_current,
                    history_states=history_states,
                    prompt=self.task,
                    history_actions=self.processor.get_action_history(),
                    steps=step_count,
                    num_loop=self.num_loop,
                )
                t_build = time.time() - t2
                
                # ========== 阶段 3: 发送请求并接收响应 ==========
                t3 = time.time()
                response = await self.ws_client.send_and_receive(payload)
                t_ws = time.time() - t3
                
                # 获取动作序列
                actions = response['actions']
                
                # 限制动作执行数量
                if self.action_steps is not None and len(actions) > self.action_steps:
                    actions = actions[:self.action_steps]
                    print(f"⚠️  动作数量限制: {len(response['actions'])} -> {self.action_steps}")
                
                # ========== 阶段 4: 执行动作（严格 30Hz 频率控制） ==========
                t4 = time.time()
                if len(actions) > 0:
                    for action_idx, action_to_execute in enumerate(actions):
                        action_start = time.time()
                        
                        # 检查复位请求（最高优先级，中断当前动作执行）
                        if self._reset_requested.is_set():
                            print("⚠️  复位请求中断当前动作执行")
                            self._reset_requested.clear()
                            current_obs = self.collector.get_observation()
                            current_state = np.array(current_obs['state'], dtype=np.float64)
                            await self._execute_reset(current_state)
                            break
                        
                        # 检查关闭请求
                        if self._shutdown_requested.is_set():
                            print("检测到关闭请求")
                            await self._safe_shutdown()
                            return
                        
                        # 发送动作到机器人
                        self.collector.send_action(action_to_execute)
                        
                        # 更新动作历史
                        self.processor.update_action_history(action_to_execute)
                        
                        # 控制每个动作的执行频率（严格 30Hz）
                        action_elapsed = time.time() - action_start
                        action_sleep = self.dt - action_elapsed
                        if action_sleep > 0:
                            await asyncio.sleep(action_sleep)
                        
                        # 打印每个动作的执行信息
                        actual_dt = time.time() - action_start
                        print(
                            f"Step {step_count}.{action_idx+1}/{len(actions)} | "
                            f"目标: {self.dt*1000:.1f}ms | "
                            f"实际: {actual_dt*1000:.1f}ms"
                        )
                        
                        step_count += 1
                t_send = time.time() - t4
                
                # 打印本轮统计
                elapsed = time.time() - loop_start
                print(
                    f"Step {step_count} | "
                    f"循环间隔: {loop_interval*1000:.1f}ms | "
                    f"采集: {t_collect*1000:.1f}ms | "
                    f"构建: {t_build*1000:.1f}ms | "
                    f"WS: {t_ws*1000:.1f}ms | "
                    f"执行: {t_send*1000:.1f}ms | "
                    f"总耗时: {elapsed*1000:.1f}ms"
                )
                
        except KeyboardInterrupt:
            print("⚠️  用户中断执行")
        except Exception as e:
            logger.error(f"❌ 执行出错: {e}", exc_info=True)
        finally:
            await self.stop()
    
    async def stop(self):
        """停止控制器"""
        # 防止重复清理
        if self._cleanup_done:
            return
        
        self._cleanup_done = True
        self.running = False
        self._collection_running = False  # 停止后台采集线程
        
        # 停止后台采集线程
        if self._collection_thread is not None and self._collection_thread.is_alive():
            self._collection_thread.join(timeout=2.0)
            print("后台采集线程已停止")
        
        # 停止键盘监听
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            print("键盘监听器已停止")
        
        # 停止数据录制
        self.collector.stop_recording()
        
        # 停止 WowSkin 传感器
        if self.wowskin_sensor is not None:
            print("正在停止 WowSkin 传感器...")
            self.wowskin_sensor.pause_streaming()
            self.wowskin_sensor.join()
        
        # 断开 WebSocket
        await self.ws_client.disconnect()
        
        # 断开机器人（检查是否已连接）
        try:
            if hasattr(self.robot, 'is_connected') and self.robot.is_connected:
                self.robot.disconnect()
                print("机器人已断开")
            else:
                self.robot.disconnect()
                print("机器人已断开")
        except Exception as e:
            logger.warning(f"断开机器人连接时出现警告: {e}")
        
        print("控制器已停止")


@draccus.wrap()
def main(cfg: ControllerConfig):
    """主函数"""
    # 构建录制配置
    recording_config = {
        'save_dir': cfg.recording.record_dir,
        'save_images': cfg.recording.save_images
    }
    
    # 创建控制器
    controller = MainController(
        robot_config=cfg.robot,
        task=cfg.task,
        server_url=cfg.server_url,
        fps=cfg.fps,
        enable_recording=cfg.recording.enable_recording,
        recording_config=recording_config,
        wowskin_config=cfg.wowskin,
        num_loop=cfg.num_loop,
        action_steps=cfg.action_steps,
    )
    
    # 运行
    asyncio.run(controller.run())


if __name__ == "__main__":
    main()