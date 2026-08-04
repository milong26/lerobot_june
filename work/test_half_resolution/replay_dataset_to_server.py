"""
从数据集回放 observation 到服务器

功能：
- 从 LeRobot 数据集读取 observation（图像、state、force）
- 发送给服务器推理
- 执行服务器返回的动作
- 数据集用完后，保持最后一个 action 继续运行，不会突然停止

使用方式（与 main_controller.py 相同的命令行参数）：
python replay_dataset_to_server.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=start_new_heihei_2 \
    --task="Grab the rectangular object" \
    --server_url=ws://10.10.16.19:9001 \
    --fps=30 \
    --action_steps=50
"""

import asyncio
import time
import logging
import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import draccus

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor
from ws_client import WSClient

# LeRobot 导入
from lerobot.datasets import LeRobotDataset
from lerobot.robots import make_robot_from_config
from lerobot.robots.config import RobotConfig

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 配置 ====================
# 数据集路径
DATASET_ROOT = "/home/qwe/.cache/huggingface/lerobot/ep10/rectangular"
EPISODE_INDEX = 0
# ==============================================


@dataclass
class ReplayConfig:
    """回放配置"""
    robot: RobotConfig
    task: str
    server_url: str = "ws://10.10.16.19:9001"
    fps: int = 30
    action_steps: Optional[int] = None
    num_loop: int = 5


class DatasetReplayer:
    """数据集回放器"""
    
    def __init__(
        self,
        robot_config: RobotConfig,
        dataset_root: str,
        episode_index: int,
        server_url: str,
        task: str,
        fps: int = 30,
        action_steps: Optional[int] = None,
        num_loop: int = 5,
    ):
        self.dataset_root = dataset_root
        self.episode_index = episode_index
        self.server_url = server_url
        self.task = task
        self.fps = fps
        self.dt = 1.0 / fps
        self.action_steps = action_steps
        self.num_loop = num_loop
        
        # 状态
        self.running = True
        self.last_observation = None
        self.last_action = None
        
        # 加载数据集
        print(f"📂 加载数据集: {dataset_root}")
        self.dataset = LeRobotDataset(repo_id="", root=Path(dataset_root))
        print(f"✅ 数据集加载完成，共 {len(self.dataset)} 帧")
        
        # 获取 episode 的起始索引
        self.episode_start = self.dataset.meta.episodes["dataset_from_index"][episode_index]
        self.episode_end = self.dataset.meta.episodes["dataset_to_index"][episode_index]
        print(f"📊 Episode {episode_index}: 帧 {self.episode_start} ~ {self.episode_end}")
        
        self.current_frame_index = self.episode_start
        
        # 获取相机 keys
        self.camera_keys = self.dataset.meta.camera_keys
        if not self.camera_keys:
            raise ValueError("数据集中没有相机数据")
        print(f"📷 相机 keys: {self.camera_keys}")
        
        # 初始化机器人（使用与 main_controller.py 相同的方式）
        print("正在连接机器人...")
        self.robot = make_robot_from_config(robot_config)
        self.robot.connect()
        print("✅ 机器人已连接")
        
        # 初始化数据处理器
        self.processor = DataProcessor(history_size=5)
        
        # 初始化 WebSocket 客户端
        self.ws_client = WSClient(server_url)
        
        # 历史帧队列
        self.frame_history = []
        
        # 记录初始状态（使用 DataCollector 的方式处理原始观测）
        raw_obs = self.robot.get_observation()
        motor_keys = ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 
                      'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
        self.initial_state = np.array([raw_obs[k] for k in motor_keys], dtype=np.float64)
        print(f"✅ 初始状态已记录: {self.initial_state}")
    
    def get_observation_from_dataset(self) -> dict:
        """从数据集获取 observation"""
        if self.current_frame_index >= self.episode_end:
            # 数据集已用完
            if self.last_observation is not None:
                print("⚠️  数据集已用完，使用最后一帧 observation")
                return self.last_observation
            else:
                raise ValueError("数据集已用完且没有历史 observation")
        
        # 从数据集读取
        frame_data = self.dataset[self.current_frame_index]
        
        # 提取图像
        images = {}
        for key in self.camera_keys:
            if key in frame_data:
                img = frame_data[key]
                if hasattr(img, "numpy"):
                    img = img.numpy()
                
                # 转换为 uint8
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                
                # CHW -> HWC
                if img.ndim == 3 and img.shape[0] == 3:
                    img = img.transpose(1, 2, 0)
                
                images[key] = img
        
        # 提取 state
        state = frame_data['observation.state']
        if hasattr(state, "numpy"):
            state = state.numpy()
        state = np.array(state, dtype=np.float64)
        
        # 提取 force（如果有）
        if 'observation.force' in frame_data:
            force = frame_data['observation.force']
            if hasattr(force, "numpy"):
                force = force.numpy()
            force = np.array(force, dtype=np.float64)
        else:
            # 如果没有 force，用零填充
            force = np.zeros(15, dtype=np.float64)
        
        observation = {
            'state': state,
            'force': force,
            'images': images,
            'timestamp': time.time()
        }
        
        # 保存最后一帧
        self.last_observation = observation
        self.current_frame_index += 1
        
        return observation
    
    def build_payload(self, frame_current: dict, frame_minus_133ms: dict) -> dict:
        """构建 payload"""
        # 处理图像
        processed_image_minus_133ms = self.processor.process_images(frame_minus_133ms['images'])
        processed_image_current = self.processor.process_images(frame_current['images'])
        
        # 构建历史状态
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
            steps=0,
            num_loop=self.num_loop,
        )
        
        return payload
    
    async def run(self):
        """运行主循环"""
        try:
            # 连接服务器
            await self.ws_client.connect()
            
            print(f"开始回放数据集到服务器")
            print(f"服务器: {self.server_url}")
            print(f"任务: {self.task}")
            print(f"控制频率: {self.fps} Hz")
            
            # 发送初始状态占位动作
            print("⏳  发送初始状态占位动作")
            initial_action = {
                'shoulder_pan.pos': self.initial_state[0],
                'shoulder_lift.pos': self.initial_state[1],
                'elbow_flex.pos': self.initial_state[2],
                'wrist_flex.pos': self.initial_state[3],
                'wrist_roll.pos': self.initial_state[4],
                'gripper.pos': self.initial_state[5],
            }
            self.robot.send_action(initial_action)
            await asyncio.sleep(self.dt)
            
            # 采集第一帧
            observation = self.get_observation_from_dataset()
            first_frame_time = time.time()
            self.frame_history.append(observation)
            print(f"📍 第一帧采集时间: {first_frame_time:.3f}")
            
            # 构建第一帧 payload 并发送
            frame_current = self.frame_history[-1]
            frame_minus_133ms = self.frame_history[-1]
            
            payload = self.build_payload(frame_current, frame_minus_133ms)
            
            print("📡 发送初始 payload，等待服务器返回动作序列...")
            response = await self.ws_client.send_and_receive(payload)
            actions = response['actions']
            
            if self.action_steps is not None and len(actions) > self.action_steps:
                actions = actions[:self.action_steps]
                print(f"⚠️  动作数量限制: {len(response['actions'])} -> {self.action_steps}")
            
            print(f"✅ 收到 {len(actions)} 个动作，开始主循环")
            
            # 主循环
            step_count = 0
            last_loop_time = None
            
            while self.running:
                loop_start = time.time()
                
                # 计算与上一轮循环的间隔时间
                if last_loop_time is not None:
                    loop_interval = loop_start - last_loop_time
                else:
                    loop_interval = 0.0
                last_loop_time = loop_start
                
                # ========== 阶段 1: 执行动作序列 ==========
                t1 = time.time()
                last_action_complete_time = None
                
                if len(actions) > 0:
                    for action_idx, action_to_execute in enumerate(actions):
                        action_start = time.time()
                        
                        # 将 numpy array 转换为字典格式
                        if isinstance(action_to_execute, np.ndarray):
                            action_dict = {
                                'shoulder_pan.pos': action_to_execute[0],
                                'shoulder_lift.pos': action_to_execute[1],
                                'elbow_flex.pos': action_to_execute[2],
                                'wrist_flex.pos': action_to_execute[3],
                                'wrist_roll.pos': action_to_execute[4],
                                'gripper.pos': action_to_execute[5],
                            }
                        else:
                            action_dict = action_to_execute
                        
                        # 发送动作到机器人
                        self.robot.send_action(action_dict)
                        
                        # 更新动作历史
                        self.processor.update_action_history(action_to_execute)
                        
                        # 保存最后一个 action
                        self.last_action = action_to_execute
                        
                        # 控制每个动作的执行频率（严格 30Hz）
                        action_elapsed = time.time() - action_start
                        action_sleep = self.dt - action_elapsed
                        if action_sleep > 0:
                            await asyncio.sleep(action_sleep)
                        
                        # 每个 action 执行完后都采集一帧
                        obs_time = time.time()
                        observation = self.get_observation_from_dataset()
                        self.frame_history.append(observation)
                        
                        step_count += 1
                    
                    # 记录最后一个 action 完成的时间
                    last_action_complete_time = time.time()
                    print(f"📍 最后一个 action 完成时间: {last_action_complete_time:.3f}")
                
                t_send = time.time() - t1
                
                # ========== 阶段 2: 选取 2 帧构建 payload ==========
                current_time = last_action_complete_time if last_action_complete_time else time.time()
                target_past_time = current_time - 0.133
                
                # 从后向前遍历队列
                frame_current = None
                frame_minus_133ms = None
                min_current_diff = float('inf')
                min_past_diff = float('inf')
                
                queue_len = len(self.frame_history)
                check_count = min(10, queue_len)
                
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
                
                # 验证选取结果
                if frame_current is None or frame_minus_133ms is None:
                    if queue_len >= 1:
                        print(f"⚠️  队列只有 {queue_len} 帧，使用同一帧")
                        frame_current = self.frame_history[-1]
                        frame_minus_133ms = self.frame_history[-1]
                    else:
                        print(f"⚠️  队列为空，跳过本轮")
                        await asyncio.sleep(self.dt)
                        continue
                
                actual_interval_ms = (frame_current['timestamp'] - frame_minus_133ms['timestamp']) * 1000
                dataset_status = "✅ 数据集" if self.current_frame_index < self.episode_end else "⚠️  已用完"
                print(
                    f"{dataset_status} | 队列 {queue_len} 帧 | "
                    f"检查 {check_count} 帧 | "
                    f"当前帧: {current_time:.3f} | "
                    f"当前帧误差: {min_current_diff*1000:.1f}ms | "
                    f"-133ms帧误差: {min_past_diff*1000:.1f}ms | "
                    f"实际间隔: {actual_interval_ms:.1f}ms"
                )
                
                # ========== 阶段 3: 构建 payload ==========
                t2 = time.time()
                payload = self.build_payload(frame_current, frame_minus_133ms)
                t_build = time.time() - t2
                
                # ========== 阶段 4: 发送请求并接收响应 ==========
                t3 = time.time()
                response = await self.ws_client.send_and_receive(payload)
                t_ws = time.time() - t3
                
                # 获取下一轮要执行的动作序列
                actions = response['actions']
                
                if self.action_steps is not None and len(actions) > self.action_steps:
                    actions = actions[:self.action_steps]
                    print(f"⚠️  动作数量限制: {len(response['actions'])} -> {self.action_steps}")
                
                # 打印本轮统计
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
            print("⚠️  用户中断执行")
        except Exception as e:
            logger.error(f"❌ 执行出错: {e}", exc_info=True)
        finally:
            await self.stop()
    
    async def stop(self):
        """停止"""
        print("🛑 开始安全关闭...")
        self.running = False
        
        # 断开 WebSocket
        await self.ws_client.disconnect()
        print("WebSocket 已断开")
        
        # 断开机器人
        self.robot.disconnect()
        print("机器人已断开")
        
        print("✅ 安全关闭完成")


@draccus.wrap()
def main(cfg: ReplayConfig):
    """主函数"""
    replayer = DatasetReplayer(
        robot_config=cfg.robot,
        dataset_root=DATASET_ROOT,
        episode_index=EPISODE_INDEX,
        server_url=cfg.server_url,
        task=cfg.task,
        fps=cfg.fps,
        action_steps=cfg.action_steps,
        num_loop=cfg.num_loop,
    )
    
    asyncio.run(replayer.run())


if __name__ == "__main__":
    main()