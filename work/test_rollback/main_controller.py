"""
主控制循环（带回滚重试机制）
- 整合 data_collector、data_processor、ws_client、rollback_manager 四个模块
- 实现主循环：采集→处理→发送→接收→执行
- 控制执行频率（30Hz）
- 将服务器返回的动作发送给机器人执行
- 异常处理和退出
- 新增：基于力传感器差异的回滚重试机制

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
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_collector import DataCollector
from data_processor import DataProcessor
from ws_client import WSClient
from rollback_manager import RollbackManager, RollbackConfig

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
    print("anyskin 未安装，WowSkin 力传感器将不可用")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ForceDataRecorder:
    """力传感器对比数据记录（新版状态机模式）"""
    
    def __init__(self, save_dir: str = "./force_comparison", rollback_config: Optional[RollbackConfig] = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.rollback_config = rollback_config
        
        self.step_indices = [] #TODO: 应该可以不需要保存这个？
        self.gripper_states = []  # 状态机状态字符串列表
        self.gripper_close_threshold = None  # 动态计算的闭合阈值
        self.initial_gripper_value = None  # 初始 gripper 值
    
    def record(self, step_idx: int, actual_force: np.ndarray, predicted_force: np.ndarray,
               gripper_state: str = "unknown"):
        """记录一帧的力传感器数据（保存状态机状态）"""
        self.step_indices.append(step_idx)
        self.gripper_states.append(gripper_state)
    
    def save_to_file(self, state_trajectory: Optional[list] = None):
        """保存数据到 npz 文件（包含完整配置和状态机轨迹）"""
        if not self.step_indices:
            print("[WARN] 没有数据，跳过保存")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        npz_path = self.save_dir / f"force_data_{timestamp}.npz"
        
        # 新版状态机配置参数
        config_params = {}
        if self.rollback_config is not None:
            config_params = {
                'enabled': self.rollback_config.enabled,
                'max_consecutive_failures': self.rollback_config.max_consecutive_failures,
                'max_rollback_count': self.rollback_config.max_rollback_count,
                'reset_wait_time': self.rollback_config.reset_wait_time,
                'use_force_check': self.rollback_config.use_force_check,
                'use_state_check': self.rollback_config.use_state_check,
                'min_start_steps': self.rollback_config.min_start_steps,
                'force_filter_cutoff_freq': self.rollback_config.force_filter_cutoff_freq,
                'force_sampling_rate': self.rollback_config.force_sampling_rate,
                'force_delay_steps': self.rollback_config.force_delay_steps,
                'gripper_velocity_threshold': self.rollback_config.gripper_velocity_threshold,
                'stable_window': self.rollback_config.stable_window,
                'settle_steps': self.rollback_config.settle_steps,
                'sustain_steps': self.rollback_config.sustain_steps,
                'max_closing_duration': self.rollback_config.max_closing_duration,
                'force_ratio_threshold': self.rollback_config.force_ratio_threshold,
                'filter_order': self.rollback_config.filter_order,
            }
        
        # 添加动态计算的 gripper_close_threshold 和 initial_gripper_value
        if self.gripper_close_threshold is not None:
            config_params['gripper_close_threshold'] = self.gripper_close_threshold
        if self.initial_gripper_value is not None:
            config_params['initial_gripper_value'] = self.initial_gripper_value
        
        # 从状态机轨迹提取滤波后的力范数
        if state_trajectory is not None and len(state_trajectory) > 0:
            trajectory_steps = np.array([t[0] for t in state_trajectory])
            trajectory_states = np.array([t[1] for t in state_trajectory])
            trajectory_actual = np.array([t[2] for t in state_trajectory])
            trajectory_predicted = np.array([t[3] for t in state_trajectory])
        else:
            trajectory_steps = np.array(self.step_indices)
            trajectory_states = np.array(self.gripper_states)
            trajectory_actual = np.array([])
            trajectory_predicted = np.array([])
        
        # 保存数据
        save_dict = {
            'step_indices': np.array(self.step_indices),
            'gripper_states': np.array(self.gripper_states),
            'config_params': np.array([str(config_params)]),
            'trajectory_steps': trajectory_steps,
            'trajectory_states': trajectory_states,
            'trajectory_actual_norms': trajectory_actual,
            'trajectory_predicted_norms': trajectory_predicted,
        }
        
        np.savez_compressed(npz_path, **save_dict)
        
        print(f"[INFO]数据已保存: {npz_path}")
        if config_params:
            print(f"[INFO] 配置参数: {config_params}")
        if state_trajectory:
            print(f"[INFO] 状态机轨迹: {len(state_trajectory)} 条记录")


class GripperDataRecorder:
    """Gripper 数据记录"""
    def __init__(self, save_dir: str = "./state_comparison"):
        self.save_dir = Path(save_dir)  # 保存目录路径
        self.save_dir.mkdir(parents=True, exist_ok=True)  # 创建目录
        self.step_indices = []  # 步数索引列表
        self.actual_gripper = []  # 实际 gripper 值（每步）
        self.predicted_gripper = []  # 预测 gripper 值（每步）
    
    def record(self, step_idx: int, actual_gripper_pos: float, predicted_gripper_pos: float):
        """记录一帧的 gripper 数据"""
        self.step_indices.append(step_idx)
        self.actual_gripper.append(actual_gripper_pos)
        self.predicted_gripper.append(predicted_gripper_pos)
    
    def save_to_file(self):
        """保存数据到 npz 文件"""
        if not self.step_indices:
            print("[WARN] 没有 gripper 数据，跳过保存")
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        npz_path = self.save_dir / f"gripper_data_{timestamp}.npz"
        
        np.savez_compressed(
            npz_path,
            step_indices=np.array(self.step_indices),
            actual_gripper=np.array(self.actual_gripper),
            predicted_gripper=np.array(self.predicted_gripper),
        )
        print(f"[INFO] Gripper 数据已保存: {npz_path}")


@dataclass
class WowSkinConfig:
    """WowSkin 力传感器配置"""
    enabled: bool = False
    port: str = ""
    num_mags: int = 5
    temp_filtered: bool = True


@dataclass
class ControllerConfig:
    """主控制器配置（新版状态机）"""
    robot: RobotConfig  # 机器人配置
    task: str  # 任务描述
    server_url: str = "ws://10.10.16.19:9001"  # 服务器地址
    fps: int = 30  # 控制频率
    max_steps: Optional[int] = None  # 最大步数
    wowskin: WowSkinConfig = field(default_factory=WowSkinConfig)  # WowSkin 力传感器配置
    num_loop: int = 5  # 循环次数
    action_steps: Optional[int] = None  # 动作步数
    # 回滚配置
    rollback_enabled: bool = True  # 是否启用回滚
    max_consecutive_failures: int = 3  # 连续失败多少次触发回滚
    max_rollback_count: int = 10  # 最大回滚次数
    reset_wait_time: float = 2.0  # 回滚后等待时间
    use_force_check: bool = True  # 是否使用力传感器检测
    use_state_check: bool = True  # 是否使用 state 检测
    # 数据记录配置
    record_force: bool = True  # 是否记录力数据
    record_force_save_dir: str = "./force_comparison"  # 力数据保存目录
    record_gripper: bool = True  # 是否记录 gripper 数据
    record_gripper_save_dir: str = "./gripper_comparison"  # gripper 数据保存目录
    # 基础配置
    min_start_steps: int = 100  # 排除前面多少步
    force_filter_cutoff_freq: float = 2.0  # Butterworth 滤波截止频率
    force_sampling_rate: float = 30.0  # 采样率
    force_delay_steps: int = 50  # 力传感器延迟补偿步数
    # 状态机配置
    gripper_velocity_threshold: float = 0.5  # gripper速度噪声阈值
    stable_window: int = 10  # 连续多少步速度低于阈值认为进入稳定候选
    settle_steps: int = 30  # 进入STABLE_CANDIDATE后等待力信号建立的步数
    sustain_steps: int = 10  # 力信号需要连续达标多少步才确认接触/滑脱
    max_closing_duration: int = 100  # 最大闭合步数（从STABLE_CANDIDATE开始计时），超时未稳定判定"一点没抓住"
    grasp_wait_steps: int = 10  # 进入STABLE_CANDIDATE后等待夹爪闭合的步数（此期间不检测力失败）
    # 力阈值配置
    force_ratio_threshold: float = 0.5  # 实际力/预测力的比值阈值
    filter_order: int = 4  # Butterworth滤波器阶数


class MainController:
    """主控制器（带回滚重试机制）"""

    # 初始化并连接设备
    def __init__(
        self,
        robot_config: RobotConfig,
        task: str,
        server_url: str,
        fps: int = 30,
        wowskin_config: Optional[WowSkinConfig] = None,
        num_loop: int = 5,
        action_steps: Optional[int] = None,
        rollback_config: Optional[RollbackConfig] = None,
        force_ratio_threshold: float = 10.0,
        max_consecutive_failures: int = 3,
        max_rollback_count: int = 10,
        reset_wait_time: float = 2.0,
        use_force_check: bool = False,
        use_state_check: bool = True,
        record_force: bool = True,
        record_force_save_dir: str = "./force_comparison",
        record_gripper: bool = True,
        record_gripper_save_dir: str = "./gripper_comparison",
    ):
        self.task = task
        self.fps = fps
        self.dt = 1.0 / fps
        self.num_loop = num_loop
        self.action_steps = action_steps
        
        # 初始化力传感器记录器（只保存力相关数据）
        self.force_recorder = ForceDataRecorder(save_dir=record_force_save_dir, rollback_config=rollback_config) if record_force else None
        
        # 初始化 gripper 记录器
        self.gripper_recorder = GripperDataRecorder(save_dir=record_gripper_save_dir) if record_gripper else None
        
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
            time.sleep(2.0)
            
            print("正在采集 WowSkin baseline...")
            baseline_data = self.wowskin_sensor.get_data(num_samples=20)
            baseline_data = np.array(baseline_data)[:, 1:]
            self.wowskin_baseline = np.mean(baseline_data, axis=0)
            print(f"[INFO] WowSkin baseline: {self.wowskin_baseline}")
        
        # 初始化机器人
        print("[INFO] 正在连接机器人...")
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()
        print("[INFO] 机器人已连接")
        
        # 初始化数据采集器
        self.collector = DataCollector(
            self.robot,
            wowskin_sensor=self.wowskin_sensor,
            wowskin_baseline=self.wowskin_baseline
        )
        
        # 初始化数据处理器
        self.processor = DataProcessor()
        
        # 初始化 WebSocket 客户端
        self.ws_client = WSClient(server_url)
        
        # 初始化回滚管理器
        self.rollback_mgr = RollbackManager(rollback_config)
        
        # 历史帧队列
        self.frame_history = deque(maxlen=20)
        
        # 控制标志
        self.running = False
        self._cleanup_done = False
        
        # 键盘控制状态
        self.initial_state = None
        self.is_resetting = False
        self.is_paused = False
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

    # 复位robot到current_state，is_rollback是后来加入的
    async def _execute_reset(self, current_state, is_rollback: bool = False):
        if self.initial_state is None:
            logger.warning("[WARN] 未设置初始状态，无法复位")
            return
        
        self.is_resetting = True
        prefix = "[ROLLBACK RESET]" if is_rollback else "[INFO]"
        
        if is_rollback:
            print(f"{prefix} 开始回滚复位...")
        
        # 步骤 1: 完全张开夹爪（gripper 是 state 的第 6 维，索引 5，完全张开值 = 100.0）
        gripper_open_value = 100.0  # 完全张开的夹爪值
        open_steps = max(5, int(abs(gripper_open_value - current_state[5]) / 1.5))
        for step in range(1, open_steps + 1):
            if not self.running:
                break
            alpha = step / open_steps
            target_state = current_state.copy()
            target_state[5] = current_state[5] + alpha * (gripper_open_value - current_state[5])
            self.collector.send_action(target_state.tolist())
            await asyncio.sleep(self.dt)
        
        # 等待夹爪完全张开
        open_wait = 1.0
        await asyncio.sleep(open_wait)
        
        # 步骤 2: 线性插值平滑回到初始状态
        current_obs_dict = self.collector.get_only_state()
        motor_keys = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 
                      'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
        current_state = np.array([current_obs_dict[k] for k in motor_keys], dtype=np.float64)
        
        diff = self.initial_state - current_state
        max_diff = np.max(np.abs(diff))
        steps = max(10, int(max_diff / 1.5))

        for step in range(1, steps + 1):
            if not self.running:
                break
            
            alpha = step / steps
            target_state = current_state + alpha * diff
            self.collector.send_action(target_state.tolist())
            await asyncio.sleep(self.dt)
        
        # 清空动作队列和历史数据
        self.frame_history.clear()
        
        if is_rollback:
            # 回滚复位：等待后自动继续
            print(f"{prefix} 复位完成，等待 {self.rollback_mgr.config.reset_wait_time} 秒...")
            await asyncio.sleep(self.rollback_mgr.config.reset_wait_time)
            self.rollback_mgr.reset_after_rollback()  # rollback_happened += 1, g_seed += 1
            print(f"{prefix} 回滚复位完成，准备重新发送请求")
        else:
            # 手动复位：暂停等待用户恢复
            # 对应参考代码 L578-L584：清空 actions，等待恢复
            self.is_paused = True
            print(f"{prefix} 复位完成，已暂停服务器通信，按左键恢复")
        
        self.is_resetting = False
    
    # 安全关闭程序and机器人
    async def _safe_shutdown(self):
        """安全关闭程序"""
        self.running = False
        
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        
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
    
    async def _select_frames_for_payload(self, current_time: float):
        """从历史队列中选取 2 帧用于构建 payload"""
        target_past_time = current_time - 0.166
        
        frame_current = None
        frame_minus_133ms = None
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
            
            past_diff = abs(ts - target_past_time)
            if past_diff < min_past_diff:
                min_past_diff = past_diff
                frame_minus_133ms = frame
            
            if min_current_diff < 0.005 and min_past_diff < 0.005:
                break
        
        if frame_current is None or frame_minus_133ms is None:
            if queue_len >= 1:
                print(f"⚠️  队列只有 {queue_len} 帧，使用同一帧")
                frame_current = self.frame_history[-1]
                frame_minus_133ms = self.frame_history[-1]
            else:
                return None, None
        
        # actual_interval_ms = (frame_current['timestamp'] - frame_minus_133ms['timestamp']) * 1000
        # print(
        #     f"[INFO] 队列 {queue_len} 帧 | "
        #     f"检查 {check_count} 帧 | "
        #     f"当前帧误差: {min_current_diff*1000:.1f}ms | "
        #     f"-133ms帧误差: {min_past_diff*1000:.1f}ms | "
        #     f"实际间隔: {actual_interval_ms:.1f}ms"
        # )
        
        return frame_current, frame_minus_133ms
    
    async def _build_and_send_payload(
        self, 
        frame_current, 
        frame_minus_133ms, 
        step_count: int,
        seed: int = 22,
        num_loop: Optional[int] = None
    ):
        """构建 payload 并发送到服务器，返回 actions 和 predicted_states"""
        processed_image_minus_133ms = self.processor.process_images(frame_minus_133ms['images'])
        processed_image_current = self.processor.process_images(frame_current['images'])
        
        history_states = [
            {'state': frame_minus_133ms['state'], 'force': frame_minus_133ms['force']},
            {'state': frame_current['state'], 'force': frame_current['force']}
        ]
        
        # 使用传入的 num_loop，如果没有则使用默认的 self.num_loop
        effective_num_loop = num_loop if num_loop is not None else self.num_loop
        
        payload = self.processor.build_payload_with_two_frames(
            image_minus_133ms=processed_image_minus_133ms,
            image_current=processed_image_current,
            history_states=history_states,
            prompt=self.task,
            steps=step_count,
            seed=seed,
            num_loop=effective_num_loop,
        )
        
        response = await self.ws_client.send_and_receive(payload)
        actions = response['actions']
        predicted_states = response.get('states', None)
        
        if self.action_steps is not None and len(actions) > self.action_steps:
            actions = actions[:self.action_steps]
            if predicted_states is not None:
                predicted_states = predicted_states[:self.action_steps]
        
        return actions, predicted_states
    
    async def _handle_reset_request(self):
        """处理手动复位请求"""
        print("检测到复位请求")
        self._reset_requested.clear()
        current_obs = self.collector.get_observation()
        current_state = np.array(current_obs['state'], dtype=np.float64)
        await self._execute_reset(current_state, is_rollback=False)
    
    async def _handle_resume_request(self, step_count: int):
        """
        处理恢复请求
        
        Returns:
            (actions, predicted_states) 或 None（如果恢复失败）
        """
        print("检测到恢复请求")
        self._resume_requested.clear()
        self.is_paused = False
        self.rollback_mgr.reset_all()
        
        observation = self.collector.get_observation()
        self.frame_history.append({
            'state': observation['state'],
            'force': observation['force'],
            'images': observation['images'],
            'timestamp': time.time()
        })
        
        frame_current = self.frame_history[-1]
        frame_minus_133ms = self.frame_history[-1]
        
        actions, predicted_states = await self._build_and_send_payload(
            frame_current, frame_minus_133ms, step_count
        )
        
        print(f"[INFO] 收到 {len(actions)} 个动作，恢复正常执行")
        return actions, predicted_states
    
    async def _handle_rollback(self, step_count: int):
        """
        处理回滚后的重新请求
        
        Returns:
            (actions, predicted_states)
        """
        print("[ROLLBACK] 回滚完成，重新采集帧并发送请求...")
        
        observation = self.collector.get_observation()
        self.frame_history.append({
            'state': observation['state'],
            'force': observation['force'],
            'images': observation['images'],
            'timestamp': time.time()
        })
        
        frame_current = self.frame_history[-1]
        frame_minus_133ms = self.frame_history[-1]
        
        # 如果回滚次数大于2次，num_loop从1开始递增
        # 第1次回滚(g_seed=1): num_loop = 2 (默认)
        # 第2次回滚(g_seed=2): num_loop = 2 (默认)
        # 第3次回滚(g_seed=3): num_loop = 1
        # 第4次回滚(g_seed=4): num_loop = 2
        # 第5次回滚(g_seed=5): num_loop = 3
        current_g_seed = self.rollback_mgr.g_seed
        if current_g_seed > 2:
            adjusted_num_loop = current_g_seed - 2
            print(f"[ROLLBACK] 回滚次数={current_g_seed} > 2，调整 num_loop 从 {self.num_loop} 到 {adjusted_num_loop}")
        else:
            adjusted_num_loop = self.num_loop
        
        actions, predicted_states = await self._build_and_send_payload(
            frame_current, frame_minus_133ms, step_count, 
            seed=current_g_seed,
            num_loop=adjusted_num_loop
        )
        
        return actions, predicted_states
    
    async def run(self, max_steps: Optional[int] = None):
        """运行主控制循环（带回滚重试机制）"""
        self.running = True
        step_count = 0
        last_loop_time = None
        
        try:
            # 启动键盘监听
            self._setup_keyboard_listener()
            
            # 等待机器人稳定
            time.sleep(1.0)
            
            # 记录初始状态
            initial_obs = self.collector.get_observation()
            self.initial_state = np.array(initial_obs['state'], dtype=np.float64)
            print(f"[INFO] 初始状态已记录: {self.initial_state}")
            
            # 连接服务器
            await self.ws_client.connect()
            
            print(f"开始执行任务: {self.task}")
            print(f"控制频率: {self.fps} Hz")
            print(f"回滚检测: 启用 (状态机模式)")
            print(f"  - Gripper: velocity_threshold={self.rollback_mgr.config.gripper_velocity_threshold}, "
                  f"stable_window={self.rollback_mgr.config.stable_window}, "
                  f"max_closing_duration={self.rollback_mgr.config.max_closing_duration}")
            print(f"  - Force: force_ratio_threshold={self.rollback_mgr.config.force_ratio_threshold}")
            print(f"  - 去抖: sustain_steps={self.rollback_mgr.config.sustain_steps}, "
                  f"max_consecutive={self.rollback_mgr.config.max_consecutive_failures}, "
                  f"max_rollback={self.rollback_mgr.config.max_rollback_count}")
            
            # 第一轮：发送初始状态占位动作
            print("[INFO] 第一轮：发送初始状态占位动作")
            self.collector.send_action(self.initial_state.tolist())
            await asyncio.sleep(self.dt)
            step_count = 1
            
            # 采集第一帧
            observation = self.collector.get_observation()
            first_frame_time = time.time()
            self.frame_history.append({
                'state': observation['state'],
                'force': observation['force'],
                'images': observation['images'],
                'timestamp': first_frame_time
            })
            print(f"[INFO] 第一帧采集时间: {first_frame_time:.3f}")
            
            # 构建第一帧 payload
            frame_current = self.frame_history[-1]
            frame_minus_133ms = self.frame_history[-1]
            
            actions, predicted_states = await self._build_and_send_payload(
                frame_current, frame_minus_133ms, step_count
            )
            
            
            # 主循环
            while self.running:
                loop_start = time.time()
                
                # 检查关闭请求
                if self._shutdown_requested.is_set():
                    print("检测到关闭请求")
                    await self._safe_shutdown()
                    return
                
                # 检查复位请求
                if self._reset_requested.is_set():
                    actions = []
                    predicted_states = None
                    await self._handle_reset_request()
                    continue
                
                # 检查恢复请求
                if self._resume_requested.is_set() and self.is_paused:
                    actions, predicted_states = await self._handle_resume_request(step_count)
                    continue
                
                # 暂停状态
                if self.is_paused:
                    self.collector.send_action(self.initial_state.tolist())
                    await asyncio.sleep(self.dt)
                    continue
                
                # 计算循环间隔
                if last_loop_time is not None:
                    loop_interval = loop_start - last_loop_time
                else:
                    loop_interval = 0.0
                last_loop_time = loop_start
                
                # ========== 阶段 1: 执行动作序列 ==========
                t1 = time.time()
                
                if len(actions) > 0:
                    force_rollback_triggered = False
                    
                    for action_idx, action_to_execute in enumerate(actions):
                        action_start = time.time()
                        
                        # 检查复位请求
                        if self._reset_requested.is_set():
                            print("[WARN] 复位请求中断当前动作执行")
                            await self._handle_reset_request()
                            break
                        
                        # 检查关闭请求
                        if self._shutdown_requested.is_set():
                            print("检测到关闭请求")
                            await self._safe_shutdown()
                            return
                        
                        # 发送动作
                        if action_idx < len(actions) - 1:
                            self.collector.send_action(action_to_execute)
                        else:
                            self.collector.send_action(self.collector.get_only_state())

                        # 控制间隔
                        action_elapsed = time.time() - action_start
                        action_sleep = self.dt - action_elapsed
                        if action_sleep > 0:
                            await asyncio.sleep(action_sleep)
                        
                        # 采集帧
                        observation = self.collector.get_observation()
                        actual_force = observation['force']
                        self.frame_history.append({
                            'state': observation['state'],
                            'force': actual_force,
                            'images': observation['images'],
                            'timestamp': time.time()
                        })
                        
                        # 回滚检测
                        # 保存predicted_gripper和actual_gripper
                        if predicted_states is not None and action_idx < len(predicted_states):
                            predicted_state = predicted_states[action_idx]
                            actual_state = observation['state']
                            actual_gripper_pos = actual_state[5]  # actual_state 索引5是夹爪位置
                            
                            # 提取预测 gripper 值（predicted_state 是一维数组，gripper 在索引5）
                            if len(predicted_state) > 0:
                                predicted_gripper_pos = float(predicted_state[5])
                                
                                # 记录 gripper 数据（只要 predicted_state 有数据就记录）
                                if self.gripper_recorder is not None:
                                    self.gripper_recorder.record(step_count, actual_gripper_pos, predicted_gripper_pos)
                            
                            # 力传感器检测需要完整的 21 维数据
                            if len(predicted_state) >= 21:
                                predicted_force = predicted_state[6:21]
                                
                                # 获取当前gripper状态机状态
                                current_gripper_state = self.rollback_mgr.gripper_state.value
                                
                                # 记录力传感器数据（包含状态机状态）
                                if self.force_recorder is not None:
                                    self.force_recorder.record(
                                        step_count, actual_force, predicted_force,
                                        gripper_state=current_gripper_state
                                    )
                                
                                # 统一回滚检测接口（新版状态机）
                                need_rollback = self.rollback_mgr.check_rollback_condition(
                                    actual_gripper_pos, actual_force, predicted_force, predicted_gripper_pos
                                )
                                
                                
                                if self.rollback_mgr.update_rollback_status(need_rollback):
                                    print(
                                        f"[DO ROLLBACK] step={step_count}, action_idx={action_idx}, "
                                        f"gripper_state={current_gripper_state}, "
                                        f"rollback_limited={self.rollback_mgr.rollback_limited}, "
                                        f"rollback_happened={self.rollback_mgr.rollback_happened}, "
                                        f"g_seed={self.rollback_mgr.g_seed}"
                                    )
                                    
                                    current_obs = self.collector.get_observation()
                                    current_state = np.array(current_obs['state'], dtype=np.float64)
                                    await self._execute_reset(current_state, is_rollback=True)
                                    
                                    force_rollback_triggered = True
                                    break
                        
                        step_count += 1
                    
                    # 回滚后重新发送请求
                    if force_rollback_triggered:
                        actions, predicted_states = await self._handle_rollback(step_count)
                        
                        # print(f"[ROLLBACK] 收到 {len(actions)} 个新动作，继续执行")
                        continue
                    
                    last_action_complete_time = time.time()
                
                # t_send = time.time() - t1
                
                # ========== 阶段 2: 选取帧 ==========
                current_time = last_action_complete_time if 'last_action_complete_time' in locals() else time.time()
                frame_current, frame_minus_133ms = await self._select_frames_for_payload(current_time)
                
                if frame_current is None or frame_minus_133ms is None:
                    await asyncio.sleep(self.dt)
                    continue
                
                # ========== 阶段 3 & 4: 构建 payload 并发送 ==========
                # t2 = time.time()
                actions, predicted_states = await self._build_and_send_payload(
                    frame_current, frame_minus_133ms, step_count
                )
                # t_build_ws = time.time() - t2
                
                # # 打印统计
                # elapsed = time.time() - loop_start
                # print(
                #     f"Step {step_count} | "
                #     f"循环间隔: {loop_interval*1000:.1f}ms | "
                #     f"执行: {t_send*1000:.1f}ms | "
                #     f"构建+WS: {t_build_ws*1000:.1f}ms | "
                #     f"总耗时: {elapsed*1000:.1f}ms"
                # )
                
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
        
        # 保存力传感器数据到文件（包含状态机轨迹）
        if self.force_recorder is not None:
            # 更新动态计算的阈值和初始值
            self.force_recorder.gripper_close_threshold = self.rollback_mgr.gripper_close_threshold
            self.force_recorder.initial_gripper_value = self.rollback_mgr.initial_gripper_value
            
            state_trajectory = self.rollback_mgr.get_state_trajectory()
            self.force_recorder.save_to_file(state_trajectory=state_trajectory)
        
        # 保存 state/gripper 数据到文件
        if self.gripper_recorder is not None:
            self.gripper_recorder.save_to_file()
        
        
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            print("键盘监听器已停止")
        
        if self.wowskin_sensor is not None:
            print("正在停止 WowSkin 传感器...")
            self.wowskin_sensor.pause_streaming()
            self.wowskin_sensor.join()
        
        await self.ws_client.disconnect()
        
        try:
            self.robot.disconnect()
            print("机器人已断开")
        except Exception as e:
            logger.warning(f"断开机器人连接时出现警告: {e}")
        
        print("控制器已停止")


@draccus.wrap()
def main(cfg: ControllerConfig):
    """主函数"""
    rollback_config = RollbackConfig(
        enabled=cfg.rollback_enabled,
        max_consecutive_failures=cfg.max_consecutive_failures,
        max_rollback_count=cfg.max_rollback_count,
        reset_wait_time=cfg.reset_wait_time,
        use_force_check=cfg.use_force_check,
        use_state_check=cfg.use_state_check,
        min_start_steps=cfg.min_start_steps,
        force_filter_cutoff_freq=cfg.force_filter_cutoff_freq,
        force_sampling_rate=cfg.force_sampling_rate,
        force_delay_steps=cfg.force_delay_steps,
        gripper_velocity_threshold=cfg.gripper_velocity_threshold,
        stable_window=cfg.stable_window,
        settle_steps=cfg.settle_steps,
        sustain_steps=cfg.sustain_steps,
        max_closing_duration=cfg.max_closing_duration,
        grasp_wait_steps=cfg.grasp_wait_steps,
        force_ratio_threshold=cfg.force_ratio_threshold,
        filter_order=cfg.filter_order,
    )
    
    controller = MainController(
        robot_config=cfg.robot,
        task=cfg.task,
        server_url=cfg.server_url,
        fps=cfg.fps,
        wowskin_config=cfg.wowskin,
        num_loop=cfg.num_loop, 
        action_steps=cfg.action_steps,
        rollback_config=rollback_config,
        record_force=cfg.record_force,
        record_force_save_dir=cfg.record_force_save_dir,
        record_gripper=cfg.record_gripper,
        record_gripper_save_dir=cfg.record_gripper_save_dir,
    )
    
    asyncio.run(controller.run())


if __name__ == "__main__":
    main()