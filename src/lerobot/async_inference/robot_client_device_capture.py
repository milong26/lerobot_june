"""
设备信息捕获与模拟测试脚本

功能：
1. 第一次运行（capture 模式）：连接真实设备，保存设备信息，在 send_action 之前停止
2. 后续运行（replay 模式）：使用保存的设备信息模拟运行，不连接真实设备

使用方法：
# 第一次运行 - 连接真实设备并保存信息
python -m lerobot.async_inference.robot_client_device_capture \
    --mode=capture \
    --save_path=./device_info.pkl \
    --server_address=10.10.16.18:8080 \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=start_new_heihei_2 \
    --robot.cameras="{ \
        camera2:{type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
        camera1: {type: intelrealsense, serial_number_or_name: 806312060427, width: 640, height: 480, fps: 30, use_depth: False} \
    }" \
    --task="pick up the component and throw it away." \
    --policy_type=tinyvla \
    --pretrained_name_or_path=outputs/trainexm/examine72/trail1_42/checkpoints/002000/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0 \
    --aggregate_fn_name=weighted_average \
    --robot.use_degrees=True

# 后续运行 - 使用保存的设备信息模拟测试
python -m lerobot.async_inference.robot_client_device_capture \
    --mode=replay \
    --save_path=./device_info.pkl \
    --server_address=10.10.16.18:8080 \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=start_new_heihei_2 \
    --robot.cameras="{ \
        camera2:{type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
        camera1: {type: intelrealsense, serial_number_or_name: 806312060427, width: 640, height: 480, fps: 30, use_depth: False} \
    }" \
    --task="pick up the component and throw it away." \
    --policy_type=tinyvla \
    --pretrained_name_or_path=outputs/trainexm/examine72/trail1_42/checkpoints/002000/pretrained_model \
    --policy_device=cuda \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0 \
    --aggregate_fn_name=weighted_average \
    --robot.use_degrees=True
"""

import argparse
import logging
import pickle
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins

from .configs import RobotClientConfig
from .helpers import (
    Action,
    FPSTracker,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    halve_image_resolution,
    is_image_key,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)


class MockRobot:
    """模拟机器人对象，使用保存的设备信息，不连接真实硬件"""

    def __init__(self, device_info: dict):
        self.action_features = device_info["action_features"]
        self.observation_features = device_info["observation_features"]
        self.logger = get_logger("mock_robot")
        self.logger.info("MockRobot initialized with saved device info")
        self.logger.info(f"Action features: {self.action_features}")
        self.logger.info(f"Observation features: {self.observation_features}")

    def connect(self):
        """模拟连接，实际不执行任何操作"""
        self.logger.info("MockRobot: Simulated connection (no real hardware)")

    def disconnect(self):
        """模拟断开连接"""
        self.logger.info("MockRobot: Simulated disconnection")

    def get_observation(self) -> dict[str, Any]:
        """模拟获取观测数据，返回符合 observation_features 的随机数据"""
        observation = {}
        for key, shape in self.observation_features.items():
            if "image" in key.lower():
                # 图像数据：创建随机图像
                if len(shape) == 3:
                    h, w, c = shape
                    observation[key] = torch.randint(0, 255, (h, w, c), dtype=torch.uint8)
                else:
                    observation[key] = torch.randint(0, 255, (480, 640, 3), dtype=torch.uint8)
            else:
                # 状态数据：创建随机向量
                dim = shape[0] if shape else 1
                observation[key] = torch.randn(dim, dtype=torch.float32)
        return observation

    def send_action(self, action_dict: dict[str, float]) -> dict[str, Any]:
        """模拟发送动作，实际不执行任何操作"""
        self.logger.info(f"MockRobot: Simulated send_action: {action_dict}")
        return action_dict


class RobotClientWithCapture:
    """支持设备信息捕获和模拟的 RobotClient

    这个类扩展了原始的 RobotClient，增加了：
    1. capture 模式：连接真实设备，保存设备信息，在 send_action 之前停止
    2. replay 模式：使用保存的设备信息模拟运行，不连接真实设备
    """

    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(
        self,
        config: RobotClientConfig,
        mode: str = "replay",
        save_path: str = "./device_info.pkl",
    ):
        """初始化 RobotClient，支持捕获和模拟两种模式

        Args:
            config: RobotClientConfig 包含所有配置参数
            mode: "capture" 或 "replay" 模式
            save_path: 设备信息保存路径
        """
        self.config = config
        self.mode = mode
        self.save_path = Path(save_path)

        if mode == "replay" and self.save_path.exists():
            # 模拟模式：使用保存的设备信息
            self.logger.info(f"[REPLAY MODE] 从 {self.save_path} 加载设备信息")
            with open(self.save_path, "rb") as f:
                device_info = pickle.load(f)
            self.robot = MockRobot(device_info)
            self.logger.info("[REPLAY MODE] 使用 MockRobot，不连接真实设备")
        else:
            # 捕获模式或首次运行：连接真实设备
            if mode == "capture":
                self.logger.info("[CAPTURE MODE] 将连接真实设备并保存设备信息")
            else:
                self.logger.info(f"[WARNING] 保存路径 {self.save_path} 不存在，将使用真实设备")

            self.robot = make_robot_from_config(config.robot)
            self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # 如果是 capture 模式，保存设备信息
        if mode == "capture":
            self._save_device_info(lerobot_features)

        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
            half_img_resolu=config.half_img_resolu,
        )
        self.half_img_resolu = config.half_img_resolu
        if self.half_img_resolu:
            self.logger.info("half_img_resolu is enabled - images will be sent at half resolution")
        else:
            self.logger.debug("half_img_resolu is disabled - images will be sent at full resolution")
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()

        # 初始化客户端变量
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)

        # FPS 测量
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        self.must_go = threading.Event()
        self.must_go.set()

    def _save_device_info(self, lerobot_features: dict):
        """保存设备信息到文件"""
        device_info = {
            "action_features": self.robot.action_features,
            "observation_features": self.robot.observation_features,
            "lerobot_features": lerobot_features,
            "robot_type": self.config.robot.type,
            "robot_id": self.config.robot.id,
            "timestamp": time.time(),
        }

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "wb") as f:
            pickle.dump(device_info, f)

        self.logger.info(f"[CAPTURE MODE] 设备信息已保存到 {self.save_path}")
        self.logger.info(f"  - action_features: {self.robot.action_features}")
        self.logger.info(f"  - observation_features: {self.robot.observation_features}")
        self.logger.info(f"  - lerobot_features: {lerobot_features}")

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """启动客户端并连接到策略服务器"""
        try:
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            self.stub.SendPolicyInstructions(policy_setup)

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """停止客户端"""
        self.shutdown_event.set()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")

    def send_observation(self, obs: TimedObservation) -> bool:
        """发送观测数据到策略服务器"""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """聚合动作队列"""
        if aggregate_fn is None:

            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            if new_action.get_timestep() <= latest_action:
                continue

            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """从策略服务器接收动作"""
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue

                receive_time = time.time()

                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)
                deserialize_time = time.perf_counter() - deserialize_start

                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    self.logger.debug(f"Converted actions to device: {client_device}")
                else:
                    self.logger.debug(f"Actions kept on device: {client_device}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))

                if verbose and len(timed_actions) > 0:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        old_timesteps = [latest_action]

                    incoming_timesteps = [a.get_timestep() for a in timed_actions]

                    first_action_timestep = timed_actions[0].get_timestep()
                    server_to_client_latency = (receive_time - timed_actions[0].get_timestamp()) * 1000

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time

                self.must_go.set()

                if verbose:
                    new_size, new_timesteps = self._inspect_action_queue()

                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_timesteps[0]}:{old_timesteps[-1]} | "
                        f"Incoming action steps: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Updated action steps: {new_timesteps[0]}:{new_timesteps[-1]}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """检查队列中是否有可用动作"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """读取并执行本地队列中的动作

        注意：在 capture 模式下，这里会在 send_action 之前停止！
        """
        get_start = time.perf_counter()
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            timed_action = self.action_queue.get_nowait()
        get_end = time.perf_counter() - get_start

        # 在 capture 模式下，不执行真实的 send_action，在此停止
        if self.mode == "capture":
            action_dict = self._action_tensor_to_action_dict(timed_action.get_action())
            self.logger.info(
                f"[CAPTURE MODE] 准备执行 send_action，但已停止！"
            )
            self.logger.info(f"[CAPTURE MODE] 动作数据: {action_dict}")
            self.logger.info(f"[CAPTURE MODE] 设备信息已保存，可以开始使用 replay 模式测试代码")
            self.logger.info(f"[CAPTURE MODE] 停止程序，不执行真实动作")
            # 抛出异常以停止程序
            raise KeyboardInterrupt(
                "[CAPTURE MODE] 设备信息已保存，程序在 send_action 之前停止。"
                "请使用 --mode=replay 进行后续测试。"
            )
        else:
            # replay 模式：使用 MockRobot 模拟
            _performed_action = self.robot.send_action(
                self._action_tensor_to_action_dict(timed_action.get_action())
            )

        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """标记客户端是否准备好发送观测数据"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            start_time = time.perf_counter()

            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            # Apply half resolution if enabled
            if self.half_img_resolu:
                for key in list(raw_observation.keys()):
                    if is_image_key(key) and isinstance(raw_observation[key], torch.Tensor):
                        original_shape = raw_observation[key].shape
                        raw_observation[key] = halve_image_resolution(raw_observation[key])
                        self.logger.debug(
                            f"Downsampled image {key} from {original_shape} to {raw_observation[key].shape}"
                        )

            with self.latest_action_lock:
                latest_action = self.latest_action

            observation = TimedObservation(
                timestamp=time.time(),
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time

            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()

            _ = self.send_observation(observation)

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                self.must_go.clear()

            if verbose:
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            self.logger.error(f"Error in observation sender: {e}")

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """执行动作和流式观测数据的控制循环"""
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()

            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


def async_client_with_capture(cfg: RobotClientConfig, mode: str, save_path: str):
    """支持设备信息捕获和模拟的客户端入口函数"""
    logging.info(pformat(asdict(cfg)))

    client = RobotClientWithCapture(cfg, mode=mode, save_path=save_path)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()

        try:
            client.control_loop(task=cfg.task)

        except KeyboardInterrupt as e:
            # capture 模式下的预期停止
            client.logger.info(f"程序已按预期停止: {e}")
        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


def main():
    """主函数：解析命令行参数并运行"""
    # 使用 argparse 解析自定义参数
    parser = argparse.ArgumentParser(description="Robot Client with device capture/replay")
    parser.add_argument(
        "--mode",
        type=str,
        default="replay",
        choices=["capture", "replay"],
        help="运行模式：capture=连接真实设备并保存信息，replay=使用保存的信息模拟",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./device_info.pkl",
        help="设备信息保存/加载路径",
    )
    # 解析已知参数，剩余参数留给 draccus
    args, remaining = parser.parse_known_args()

    # 使用 draccus 解析剩余参数
    sys.argv = [sys.argv[0]] + remaining
    register_third_party_plugins()

    @draccus.wrap()
    def run(cfg: RobotClientConfig):
        async_client_with_capture(cfg, mode=args.mode, save_path=args.save_path)

    run()


if __name__ == "__main__":
    main()