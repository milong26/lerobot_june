"""
回滚重试模块
功能：
- 检测力传感器差异（条件A）
- 检测 gripper 减小趋势（条件B）
- 满足任一条件即判断需要回滚
"""

import numpy as np
from typing import Optional
from collections import deque
from scipy.signal import butter, filtfilt
import logging

logger = logging.getLogger(__name__)


class RollbackConfig:
    """回滚重试配置"""
    def __init__(
        self,
        enabled: bool = True,
        max_consecutive_failures: int = 3,  # 连续失败多少次才触发回滚
        max_rollback_count: int = 10,  # 最大回滚次数
        reset_wait_time: float = 2.0,  # 回滚后等待时间（秒）
        use_force_check: bool = True,  # 是否使用力传感器检测
        use_state_check: bool = True,  # 是否使用 state 检测
        # 力检查配置
        force_ratio_multiplier: float = 5.0,  # 预测力/实际力比值阈值
        force_delay_steps: int = 30,  # 力传感器延迟补偿步数
        force_filter_cutoff_freq: float = 2.0,  # Butterworth 滤波截止频率 (Hz)
        force_sampling_rate: float = 30.0,  # 采样率 (Hz)
        grasp_history_window: int = 50,  # 计算历史均值的窗口大小
        min_start_steps: int = 100,  # 排除前面多少步，避免初始阶段误判
        # Gripper 检测配置
        gripper_decrease_threshold: int = 10,  # 连续减小多少步认为 gripper 在减小趋势
        gripper_stable_threshold: float = 0.5,  # gripper 稳定阈值：变化量小于此值认为稳定
        use_gripper_stable_check: bool = False,  # 是否使用 gripper 稳定检测（True=稳定检测，False=减小趋势检测）
        use_gripper_initial_close_check: bool = False,  # 是否使用初始 gripper 闭合检测（True=当前值<初始值即满足条件B）
    ):
        self.enabled = enabled
        self.max_consecutive_failures = max_consecutive_failures
        self.max_rollback_count = max_rollback_count
        self.reset_wait_time = reset_wait_time
        self.use_force_check = use_force_check
        self.use_state_check = use_state_check
        self.force_ratio_multiplier = force_ratio_multiplier
        self.force_delay_steps = force_delay_steps
        self.force_filter_cutoff_freq = force_filter_cutoff_freq
        self.force_sampling_rate = force_sampling_rate
        self.grasp_history_window = grasp_history_window
        self.min_start_steps = min_start_steps
        self.gripper_decrease_threshold = gripper_decrease_threshold
        self.gripper_stable_threshold = gripper_stable_threshold
        self.use_gripper_stable_check = use_gripper_stable_check
        self.use_gripper_initial_close_check = use_gripper_initial_close_check


class RollbackManager:
    """回滚管理器"""
    
    def __init__(self, config: Optional[RollbackConfig] = None):
        self.config = config or RollbackConfig()
        
        # 回滚状态计数器
        self.rollback_limited = 0  # 连续检测到需要回滚的次数
        self.rollback_happened = 0  # 已触发回滚的总次数
        self.g_seed = 0  # 随机种子，用于服务器生成不同的动作序列
        self.step_counter = 0  # 总步数计数器，不回滚重置
        
        # 力传感器历史数据（用于延迟补偿和均值计算）
        self.actual_force_history: deque = deque(maxlen=self.config.force_delay_steps + 50)  # 保存实际力范数历史，用于延迟补偿
        self.predicted_force_15d_history: deque = deque(maxlen=self.config.grasp_history_window * 2)  # 保存15维预测力历史，用于滤波和均值计算
        
        # Gripper 历史数据（用于检测减小趋势）
        self.predicted_gripper_history: deque = deque(maxlen=self.config.grasp_history_window * 2)  # 保存预测 gripper 值历史
        self.actual_gripper_history: deque = deque(maxlen=self.config.grasp_history_window * 2)  # 保存实际 gripper 值历史
        self.initial_gripper_value: Optional[float] = None  # 初始 gripper 值，用于闭合检测
        
        # 回滚后的冷却期
        self.post_rollback_cooldown_steps = 100  # 回滚后多少步内不检测
        self.steps_since_rollback = 999  # 回滚后经过的步数，初始值大避免首次被冷却

    def _butterworth_lowpass_1d(self, signal, cutoff_freq=None, fs=None, order=4):
        """Butterworth 低通滤波（1D信号）"""
        if cutoff_freq is None:
            cutoff_freq = self.config.force_filter_cutoff_freq
        if fs is None:
            fs = self.config.force_sampling_rate
        
        # filtfilt 要求信号长度 > padlen (padlen = 3 * max(len(a), len(b)))
        # 对于 order 阶滤波器，len(a) = len(b) = order + 1
        # 所以 padlen = 3 * (order + 1)，需要 len(signal) > padlen
        min_length = 3 * (order + 1) + 1  # 至少需要 padlen + 1
        if len(signal) <= min_length:
            return signal
        
        nyquist = fs / 2.0
        normalized_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)

    def check_force_rollback_condition(
        self,
        actual_force: np.ndarray,
        predicted_force: np.ndarray,
    ) -> bool:
        """
        条件A：检查力传感器差异
        
        逻辑：
        - 对15维力数据分别进行 Butterworth 低通滤波
        - 计算滤波后的预测力 L2 范数
        - 如果预测力 L2 范数 / 历史均值 > 阈值，认为需要回滚
        - 同时检查预测力/实际力（延迟补偿后）> 阈值
        """
        if not self.config.enabled or not self.config.use_force_check:
            return False
        
        self.step_counter += 1
        self.steps_since_rollback += 1
        
        # 计算当前 L2 范数（原始数据）
        actual_norm = np.linalg.norm(actual_force)
        predicted_norm = np.linalg.norm(predicted_force)
        
        # 添加到历史（用于延迟补偿）
        self.actual_force_history.append(actual_norm)
        
        # 添加到 15 维力历史（用于滤波和均值计算）
        self.predicted_force_15d_history.append(predicted_force.copy())
        
        
        # 历史数据不足时，暂不判断
        if len(self.actual_force_history) < 3:
            return False
        
        # 回滚后冷却期内不检测
        if self.steps_since_rollback < self.post_rollback_cooldown_steps:
            return False
        
        # 排除初始阶段
        if self.step_counter < self.config.min_start_steps:
            return False
        
        # ===== 条件A1：预测力相对历史均值显著增大 =====
        force_detected_by_history = False
        history_size = len(self.predicted_force_15d_history)
        history_start = max(0, history_size - self.config.grasp_history_window - 1)
        history_predicted_15d = list(self.predicted_force_15d_history)[history_start:-1]
        
        if len(history_predicted_15d) >= 10:
            history_array = np.array(history_predicted_15d)
            n_dims = history_array.shape[1]
            history_filtered = np.zeros_like(history_array)
            for dim in range(n_dims):
                history_filtered[:, dim] = self._butterworth_lowpass_1d(history_array[:, dim])
            
            history_norms_filtered = np.linalg.norm(history_filtered, axis=1)
            history_mean = np.mean(history_norms_filtered)
            
            if history_mean > 1e-6:
                current_array = np.array(list(self.predicted_force_15d_history))
                if len(current_array) >= 15:
                    current_filtered = np.zeros_like(current_array)
                    for dim in range(n_dims):
                        current_filtered[:, dim] = self._butterworth_lowpass_1d(current_array[:, dim])
                    predicted_norm_filtered = np.linalg.norm(current_filtered[-1])
                else:
                    predicted_norm_filtered = predicted_norm
                
                relative_ratio = predicted_norm_filtered / history_mean
                
                if relative_ratio > self.config.force_ratio_multiplier:
                    force_detected_by_history = True
                    # print(
                    #     f"[FORCE CHECK-A1] step={self.step_counter}, "
                    #     f"predicted_norm_filtered={predicted_norm_filtered:.4f}, "
                    #     f"history_mean={history_mean:.4f}, "
                    #     f"ratio={relative_ratio:.2f} (threshold={self.config.force_ratio_multiplier})"
                    # )
        
        # ===== 条件A2：实际力/历史预测力（延迟补偿）> 阈值 =====
        # 逻辑：实际力滞后，用当前实际力 与 force_delay_steps 前的预测力比较
        # 如果实际力远小于历史预测，说明预测的力没有出现，实际环境与预期不符
        # 使用滤波后的数据进行比较
        force_detected_by_ratio = False
        
        # 对实际力历史进行滤波
        if len(self.actual_force_history) >= 15:
            actual_history_array = np.array(list(self.actual_force_history))
            actual_filtered = self._butterworth_lowpass_1d(actual_history_array)
            actual_norm_filtered = actual_filtered[-1]
        else:
            actual_norm_filtered = actual_norm
        
        # 获取延迟补偿后的预测力（使用滤波后的值）
        if len(self.predicted_force_15d_history) > self.config.force_delay_steps:
            # 获取延迟点附近的窗口进行滤波
            delay_idx = -self.config.force_delay_steps - 1
            window_size = 15
            window_start = delay_idx - window_size + 1
            window_forces = list(self.predicted_force_15d_history)[window_start:delay_idx + 1]
            
            if len(window_forces) >= window_size:
                window_array = np.array(window_forces)
                window_filtered = np.zeros_like(window_array)
                for dim in range(window_array.shape[1]):
                    window_filtered[:, dim] = self._butterworth_lowpass_1d(window_array[:, dim])
                delayed_predicted_norm = np.linalg.norm(window_filtered[-1])
            else:
                delayed_predicted_force = list(self.predicted_force_15d_history)[delay_idx]
                delayed_predicted_norm = np.linalg.norm(delayed_predicted_force)
        else:
            delayed_predicted_norm = predicted_norm
        
        if actual_norm_filtered < 1e-6:
            condition_ratio = delayed_predicted_norm > 1e-6
        else:
            condition_ratio = (delayed_predicted_norm / actual_norm_filtered) > self.config.force_ratio_multiplier
        
        if condition_ratio:
            force_detected_by_ratio = True
            force_ratio = delayed_predicted_norm / actual_norm if actual_norm >= 1e-6 else float('inf')
            # print(
            #     f"[FORCE CHECK-A2] step={self.step_counter}, "
            #     f"delayed_predicted_norm={delayed_predicted_norm:.3f}, "
            #     f"actual_norm={actual_norm:.3f}, "
            #     f"ratio={force_ratio:.3f} (threshold={self.config.force_ratio_multiplier})"
            # )
        
        # 满足任一条件即认为需要回滚（A1或A2）
        need_rollback = force_detected_by_history or force_detected_by_ratio
        
        return need_rollback

    def _check_gripper_initial_close(self, current_gripper: float) -> bool:
        """
        条件B（最新）：当前 gripper 值 < 初始值即满足条件B
        
        逻辑：
        - 以初始 gripper 值作为参考
        - 如果当前 gripper 值小于初始值，说明 gripper 已经闭合
        - 这适用于不同宽度的物体，不依赖绝对值
        """
        if self.initial_gripper_value is None:
            return False
        
        if current_gripper < self.initial_gripper_value:
            decrease_ratio = (self.initial_gripper_value - current_gripper) / (self.initial_gripper_value + 1e-6)
            # print(
            #     f"[GRIPPER CHECK-B-initial-close] step={self.step_counter}, "
            #     f"current_gripper={current_gripper:.2f}, "
            #     f"initial_gripper={self.initial_gripper_value:.2f}, "
            #     f"decrease_ratio={decrease_ratio:.2%}"
            # )
            return True
        
        return False

    def _check_gripper_decreasing_trend(self) -> bool:
        """
        条件B（旧）：检测 actual gripper 减小趋势
        
        逻辑：
        - 每一步保存一个实际 gripper 值
        - 判断从历史到当前是否呈现闭合趋势（值变小）
        - 如果当前值 < 历史均值，说明 gripper 在减小
        - 如果减小幅度超过阈值，认为 gripper 在持续闭合
        """
        if len(self.actual_gripper_history) < 3:
            return False
        
        # 获取历史值（不包括当前值）
        history_values = list(self.actual_gripper_history)[:-1]
        current_value = self.actual_gripper_history[-1]
        
        if len(history_values) < 10:
            return False
        
        # 计算历史均值
        history_mean = np.mean(history_values)
        
        # 如果当前值 < 历史均值，说明 gripper 在减小
        if current_value < history_mean:
            decrease_ratio = (history_mean - current_value) / (history_mean + 1e-6)
            
            # 减小幅度超过阈值（例如 10%）
            if decrease_ratio > 0.1:
                print(
                    f"[GRIPPER CHECK-B-decrease] step={self.step_counter}, "
                    f"current_gripper={current_value:.2f}, "
                    f"history_mean={history_mean:.2f}, "
                    f"decrease_ratio={decrease_ratio:.2%}"
                )
                return True
        
        return False

    def _check_gripper_stable(self) -> bool:
        """
        条件B（新）：检测 gripper 是否趋于稳定（不再减小或变化很小）
        
        逻辑：
        - 计算最近 N 步 gripper 的平均变化量
        - 如果变化量小于阈值，认为 gripper 已经稳定
        - 这适用于不同宽度的物体，不依赖绝对值
        """
        if len(self.actual_gripper_history) < 10:
            return False
        
        # 获取最近 10 步的 gripper 值
        recent_values = list(self.actual_gripper_history)[-10:]
        
        # 计算相邻差分的绝对值
        diffs = [abs(recent_values[i+1] - recent_values[i]) for i in range(len(recent_values)-1)]
        
        # 平均变化量
        avg_change = np.mean(diffs)
        
        # 如果平均变化量小于阈值，认为 gripper 稳定
        if avg_change < self.config.gripper_stable_threshold:
            print(
                f"[GRIPPER CHECK-B-stable] step={self.step_counter}, "
                f"avg_change={avg_change:.4f}, "
                f"threshold={self.config.gripper_stable_threshold}"
            )
            return True
        
        return False

    def check_state_rollback_condition(
        self,
        actual_gripper_pos: float,
        predicted_gripper_pos: float,
    ) -> bool:
        """
        检查 state 维度的回滚条件
        
        条件B：根据配置选择检测模式
            - use_gripper_initial_close_check=True: 当前值 < 初始值即满足条件B
            - use_gripper_stable_check=True: gripper 趋于稳定（不再减小或变化很小）
            - use_gripper_stable_check=False: gripper 减小趋势检测（旧逻辑）
        """
        if not self.config.enabled or not self.config.use_state_check:
            return False
        
        # 保存初始 gripper 值（第一次调用时）
        if self.initial_gripper_value is None:
            self.initial_gripper_value = actual_gripper_pos
            print(f"[GRIPPER] 初始 gripper 值: {self.initial_gripper_value:.2f}")
        
        # 保存实际 gripper 值到历史（用于检测）
        self.actual_gripper_history.append(actual_gripper_pos)
        
        # ===== 条件B：根据配置选择检测模式 =====
        if self.config.use_gripper_initial_close_check:
            # 新模式：当前值 < 初始值即满足条件B
            condition_b_met = self._check_gripper_initial_close(actual_gripper_pos)
        elif self.config.use_gripper_stable_check:
            # 中模式：检测 gripper 是否趋于稳定
            condition_b_met = self._check_gripper_stable()
        else:
            # 旧模式：检测 gripper 减小趋势
            condition_b_met = self._check_gripper_decreasing_trend()
        
        if condition_b_met:
            print(
                f"[STATE ROLLBACK-B] actual_gripper_pos={actual_gripper_pos:.2f}, "
                f"predicted_gripper_pos={predicted_gripper_pos:.2f}"
            )
        
        return condition_b_met

    def check_rollback_condition(
        self,
        actual_gripper_pos: float,
        actual_force: np.ndarray,
        predicted_force: np.ndarray,
        predicted_gripper_pos: float,
    ) -> bool:
        """
        统一回滚检测接口
        
        条件A：力传感器差异检测
        条件B：Gripper 减小趋势检测
        
        条件A和条件B同时满足才判断需要回滚
        """
        if not self.config.enabled:
            return False
        
        need_rollback_force = False
        need_rollback_state = False
        
        # 力传感器检测（条件A）
        if self.config.use_force_check:
            need_rollback_force = self.check_force_rollback_condition(actual_force, predicted_force)
        
        # State 检测（条件B：gripper 减小趋势）
        if self.config.use_state_check:
            need_rollback_state = self.check_state_rollback_condition(actual_gripper_pos, predicted_gripper_pos)
        
        # 条件A和条件B同时满足才触发回滚
        need_rollback = need_rollback_force and need_rollback_state
        
        if need_rollback:
            print(
                f"[ROLLBACK TRIGGERED] 条件A(force)={need_rollback_force}, "
                f"条件B(gripper)={need_rollback_state} -> 同时满足，触发回滚"
            )
        
        return need_rollback
    
    def update_rollback_status(self, need_rollback: bool) -> bool:
        """
        更新回滚状态并判断是否触发回滚
        
        Returns:
            True 表示需要执行回滚，False 表示正常
        """
        if need_rollback:
            self.rollback_limited += 1
        else:
            self.rollback_limited = 0

        # 连续失败次数超过阈值且未达到最大回滚次数
        if (self.rollback_limited >= self.config.max_consecutive_failures 
            and self.rollback_happened < self.config.max_rollback_count):
            return True
        
        return False
    
    def reset_after_rollback(self):
        """回滚后重置状态"""
        self.rollback_limited = 0
        self.rollback_happened += 1
        self.g_seed += 1
        self.actual_force_history.clear()
        self.steps_since_rollback = 0
        
        print(
            f"[ROLLBACK RESET] 回滚 #{self.rollback_happened}, "
            f"进入冷却期 {self.post_rollback_cooldown_steps} 步, "
            f"g_seed={self.g_seed}"
        )
    
    def reset_all(self):
        """完全重置所有状态"""
        self.rollback_limited = 0
        self.rollback_happened = 0
        self.g_seed = 0
        self.actual_force_history.clear()
        self.predicted_force_15d_history.clear()
        self.predicted_gripper_history.clear()
        self.actual_gripper_history.clear()
        self.step_counter = 0
        self.steps_since_rollback = 999
        
        print("[ROLLBACK RESET ALL] 完全重置所有状态")