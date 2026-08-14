"""
回滚重试模块 - 重构版
功能：
- Gripper 状态机：CLOSING → STABLE_CANDIDATE → GRASP_CONFIRMED
- Force 判据：统一滤波顺序（15维逐维因果滤波 → L2范数）
- 失败检测：一点没抓住 / 抓住一半松了
"""

import numpy as np
from typing import Optional, Tuple
from collections import deque
from scipy.signal import butter, filtfilt, sosfilt, sosfilt_zi
from enum import Enum


class GripperState(Enum):
    """Gripper 状态机状态"""
    CLOSING = "closing"                    # 正在闭合
    STABLE_CANDIDATE = "stable_candidate"  # 速度稳定，等待力信号确认
    GRASP_CONFIRMED = "grasp_confirmed"    # 已确认抓取成功
    FAILED_NO_GRASP = "failed_no_grasp"    # 一点没抓住（超时未稳定）


class RollbackConfig:
    def __init__(
        self,
        enabled: bool = True,
        max_consecutive_failures: int = 3,  # 连续失败多少次才触发回滚
        max_rollback_count: int = 10,  # 最大回滚次数
        reset_wait_time: float = 2.0,  # 回滚后等待时间（秒）
        use_force_check: bool = True,  # 是否使用力传感器检测
        use_state_check: bool = True,  # 是否使用 state 检测
        # 基础配置
        min_start_steps: int = 100,  # 排除前面多少步，避免初始阶段误判
        # 滤波配置
        force_filter_cutoff_freq: float = 2.0,  # Butterworth 滤波截止频率 (Hz)
        force_sampling_rate: float = 30.0,  # 采样率 (Hz)
        force_delay_steps: int = 50,  # 力传感器延迟补偿步数
        # 状态机配置
        gripper_velocity_threshold: float = 0.5,  # gripper速度噪声阈值（相邻差分绝对值均值）
        stable_window: int = 10,  # 连续多少步速度低于阈值认为进入稳定候选
        settle_steps: int = 30,  # 进入STABLE_CANDIDATE后等待力信号建立的步数
        sustain_steps: int = 10,  # 力信号需要连续达标多少步才确认接触/滑脱
        max_closing_duration: int = 100,  # 最大闭合步数（从STABLE_CANDIDATE开始计时），超时未稳定判定"一点没抓住"
        grasp_wait_steps: int = 10,  # 进入STABLE_CANDIDATE后等待夹爪闭合的步数（此期间不检测力失败）
        # 力阈值配置
        force_ratio_threshold: float = 0.5,  # 实际力/预测力的比值阈值（判断是否匹配）
        filter_order: int = 4,  # Butterworth滤波器阶数
    ):
        self.enabled = enabled
        self.max_consecutive_failures = max_consecutive_failures
        self.max_rollback_count = max_rollback_count
        self.reset_wait_time = reset_wait_time
        self.use_force_check = use_force_check
        self.use_state_check = use_state_check
        self.min_start_steps = min_start_steps
        self.force_filter_cutoff_freq = force_filter_cutoff_freq
        self.force_sampling_rate = force_sampling_rate
        self.force_delay_steps = force_delay_steps
        self.gripper_velocity_threshold = gripper_velocity_threshold
        self.stable_window = stable_window
        self.settle_steps = settle_steps
        self.sustain_steps = sustain_steps
        self.max_closing_duration = max_closing_duration
        self.grasp_wait_steps = grasp_wait_steps
        self.force_ratio_threshold = force_ratio_threshold
        self.filter_order = filter_order


class CausalFilter:
    """因果滤波器（15维，逐维sosfilt + zi状态）"""
    
    def __init__(self, n_dims: int = 15, cutoff_freq: float = 2.0, 
                 fs: float = 30.0, order: int = 4):
        self.n_dims = n_dims
        nyquist = fs / 2.0
        normalized_cutoff = cutoff_freq / nyquist
        self.sos = butter(order, normalized_cutoff, btype='low', output='sos')
        # sos shape: (n_sections, 6), where n_sections = order // 2
        # zi shape for each channel: (n_sections, 2) - required by sosfilt for 1D input
        n_sections = self.sos.shape[0]
        # 为每个维度初始化 zi: (n_dims, n_sections, 2)
        self.zi = np.zeros((n_dims, n_sections, 2))
    
    def filter_step(self, force_15d: np.ndarray) -> np.ndarray:
        """单步因果滤波，更新zi状态"""
        # force_15d shape: (15,)
        filtered = np.zeros(self.n_dims)
        for dim in range(self.n_dims):
            # 取出该维度的zi状态 (n_sections, 2)
            zi_dim = self.zi[dim]  # (n_sections, 2)
            # 输入信号 (1,) 单样本
            # sosfilt 返回 (output_array, new_zi_array)
            result, new_zi = sosfilt(
                self.sos, force_15d[dim:dim+1], zi=zi_dim
            )
            # result shape: (1,), 提取标量
            filtered[dim] = result[0]
            # 更新该维度的 zi 状态
            self.zi[dim] = new_zi
        return filtered  # (15,)
    
    def reset(self):
        """重置滤波器状态"""
        n_sections = self.sos.shape[0]
        self.zi = np.zeros((self.n_dims, n_sections, 2))
    
    def filtfilt_offline(self, signal_15d: np.ndarray) -> np.ndarray:
        """离线双向滤波（用于check_force_dim9.py）"""
        # signal_15d shape: (N, 15)
        filtered = np.zeros_like(signal_15d)
        for dim in range(self.n_dims):
            filtered[:, dim] = filtfilt(self.sos, signal_15d[:, dim])
        return filtered


class RollbackManager:
    """回滚管理器 - 重构版"""
    
    def __init__(self, config: Optional[RollbackConfig] = None):
        self.config = config or RollbackConfig()
        
        # 回滚状态计数器
        self.rollback_limited = 0  # 连续检测到需要回滚的次数
        self.rollback_happened = 0  # 已触发回滚的总次数
        self.g_seed = 0  # 随机种子，用于服务器生成不同的动作序列
        self.step_counter = 0  # 总步数计数器，不回滚重置
        
        # Gripper 状态机
        self.gripper_state = GripperState.CLOSING
        self.stable_step_count = 0  # 连续稳定步数计数器
        self.force_sustain_count = 0  # 力达标连续步数
        self.force_fail_count = 0  # 力不达标连续步数
        self.closing_start_step = 0  # CLOSING状态开始步数
        self.settle_step_count = 0  # settle等待步数
        self.grasp_wait_counter = 0  # 进入STABLE_CANDIDATE后的闭合等待步数
        
        # Gripper 历史数据（用于速度计算和闭合检测）
        self.actual_gripper_history: deque = deque(maxlen=self.config.stable_window + 10)
        self.initial_gripper_value: Optional[float] = None
        self.gripper_max_value: Optional[float] = None  # 记录夹爪初始最大值（闭合前）
        self.gripper_has_closed = False  # 是否检测到夹爪闭合动作（位置下降）
        self.gripper_close_threshold: Optional[float] = None  # 动态计算的闭合阈值（基于initial_gripper）
        self.closing_detected_step: Optional[int] = None  # 检测到稳定下降的步数
        self.stable_candidate_start_step: Optional[int] = None  # 进入STABLE_CANDIDATE的步数
        
        # 因果滤波器（actual和predicted分开维护）
        self.actual_force_filter = CausalFilter(
            n_dims=15,
            cutoff_freq=self.config.force_filter_cutoff_freq,
            fs=self.config.force_sampling_rate,
            order=self.config.filter_order
        )
        self.predicted_force_filter = CausalFilter(
            n_dims=15,
            cutoff_freq=self.config.force_filter_cutoff_freq,
            fs=self.config.force_sampling_rate,
            order=self.config.filter_order
        )
        
        # 延迟补偿历史（保存滤波后的predicted力范数）
        self.predicted_norm_history: deque = deque(maxlen=self.config.force_delay_steps + 50)
        
        # 状态机轨迹记录（用于保存到npz文件）
        self.state_trajectory = []  # List[Tuple[step, gripper_state_str, actual_norm, predicted_norm]]
        
        # 回滚后的冷却期
        self.post_rollback_cooldown_steps = 100  # 回滚后多少步内不检测
        self.steps_since_rollback = 999  # 回滚后经过的步数，初始值大避免首次被冷却

    def _compute_gripper_velocity(self, current_gripper: float) -> float:
        """计算gripper速度（最近K步相邻差分绝对值均值）"""
        if len(self.actual_gripper_history) < 2:
            return float('inf')
        
        recent_values = list(self.actual_gripper_history)[-self.config.stable_window:]
        if len(recent_values) < 2:
            return float('inf')
        
        diffs = [abs(recent_values[i+1] - recent_values[i]) for i in range(len(recent_values)-1)]
        return float(np.mean(diffs))

    def _detect_closing_start(self, current_gripper: float, step: int) -> bool:
        """
        检测夹爪是否开始稳定下降过程
        判定标准：actual gripper 连续稳定下降（相邻差分都为负）
        """
        # 先添加到历史
        self.actual_gripper_history.append(current_gripper)
        
        if len(self.actual_gripper_history) < 5:
            return False
        
        # 检查最近5步是否都在下降
        recent_values = list(self.actual_gripper_history)[-5:]
        for i in range(len(recent_values) - 1):
            if recent_values[i+1] >= recent_values[i]:
                return False
        
        return True

    def _update_gripper_state(self, current_gripper: float, step: int) -> None:
        """更新gripper状态机"""
        # 先检测是否开始稳定下降（会添加 current_gripper 到历史）
        if not self.gripper_has_closed:
            if self._detect_closing_start(current_gripper, step):
                self.gripper_has_closed = True
                self.closing_detected_step = step
                # 动态计算闭合阈值：基于 initial gripper 的 15%
                if self.initial_gripper_value is not None:
                    self.gripper_close_threshold = self.initial_gripper_value * 0.15
                    print(
                        f"[GRIPPER] ✓ 检测到稳定下降开始,gripper_has_closed设置成true， step={step}, "
                        f"initial_gripper={self.initial_gripper_value:.2f}, "
                        f"close_threshold={self.gripper_close_threshold:.2f}"
                    )
        
        # 计算速度（使用已更新的历史）
        velocity = self._compute_gripper_velocity(current_gripper)
        
        # 更新夹爪最大值（闭合前的初始位置）
        if self.gripper_max_value is None:
            self.gripper_max_value = current_gripper
        else:
            self.gripper_max_value = max(self.gripper_max_value, current_gripper)
        
        if self.gripper_state == GripperState.CLOSING:
            # 检测到稳定下降后，速度低于阈值即可进入 STABLE_CANDIDATE
            if self.gripper_has_closed and velocity < self.config.gripper_velocity_threshold:
                self.stable_step_count += 1
                if self.stable_step_count >= self.config.stable_window:
                    # 进入稳定候选
                    self.gripper_state = GripperState.STABLE_CANDIDATE
                    self.stable_candidate_start_step = step  # 记录进入 STABLE_CANDIDATE 的步数
                    self.stable_step_count = 0
                    self.settle_step_count = 0
                    self.force_sustain_count = 0
                    self.force_fail_count = 0
                    self.grasp_wait_counter = 0
                    print(
                        f"[GRIPPER] ✓ 进入STABLE_CANDIDATE, step={step}, "
                        f"velocity={velocity:.4f} < threshold={self.config.gripper_velocity_threshold}, "
                        f"stable_count={self.stable_step_count} >= window={self.config.stable_window}, "
                        f"将等待 {self.config.grasp_wait_steps} 步用于夹爪闭合"
                    )
            else:
                self.stable_step_count = 0
                    
        elif self.gripper_state == GripperState.STABLE_CANDIDATE:
            if velocity >= self.config.gripper_velocity_threshold:
                # 稳定被打破，回到CLOSING重新评估
                self.gripper_state = GripperState.CLOSING
                self.stable_step_count = 0
                self.closing_start_step = step
                print(
                    f"[GRIPPER] → 稳定被打破，回到CLOSING, step={step}, "
                    f"velocity={velocity:.4f} >= threshold={self.config.gripper_velocity_threshold}"
                )
            else:
                self.settle_step_count += 1
                
        elif self.gripper_state == GripperState.GRASP_CONFIRMED:
            if velocity >= self.config.gripper_velocity_threshold:
                # 位置变化，回到CLOSING重新评估
                self.gripper_state = GripperState.CLOSING
                self.stable_step_count = 0
                self.closing_start_step = step
                self.force_sustain_count = 0
                self.force_fail_count = 0
                print(
                    f"[GRIPPER] → 位置变化，回到CLOSING, step={step}, "
                    f"velocity={velocity:.4f} >= threshold={self.config.gripper_velocity_threshold}"
                )

    def _update_force_condition(self, actual_force_15d: np.ndarray, 
                                predicted_force_15d: np.ndarray, 
                                actual_gripper_pos: float,
                                predicted_gripper_pos: float,
                                step: int) -> Optional[str]:
        """更新力判据，返回失败原因或None"""
        # 因果滤波
        actual_filtered = self.actual_force_filter.filter_step(actual_force_15d)
        predicted_filtered = self.predicted_force_filter.filter_step(predicted_force_15d)
        
        actual_norm = float(np.linalg.norm(actual_filtered))
        predicted_norm = float(np.linalg.norm(predicted_filtered))
        
        # 保存predicted范数到延迟历史
        self.predicted_norm_history.append(predicted_norm)
        
        if self.gripper_state == GripperState.STABLE_CANDIDATE:
            # 检查 max_closing_duration 超时（从 STABLE_CANDIDATE 开始计时）
            if self.stable_candidate_start_step is not None:
                duration_in_stable = step - self.stable_candidate_start_step
                if duration_in_stable >= self.config.max_closing_duration:
                    self.gripper_state = GripperState.FAILED_NO_GRASP
                    print(
                        f"[GRIPPER] ✗ FAILED_NO_GRASP (超时), step={step}, "
                        f"duration_in_stable={duration_in_stable} >= max={self.config.max_closing_duration}"
                    )
                    return "FAILED_NO_GRASP"
            
            # 递增闭合等待计数器
            self.grasp_wait_counter += 1
            
            # 在闭合等待期间，不检测力失败，给夹爪时间闭合
            if self.grasp_wait_counter < self.config.grasp_wait_steps:
                # 仅记录轨迹，不做力判据
                if self.grasp_wait_counter % 20 == 0 or self.grasp_wait_counter == 1:
                    print(
                        f"[FORCE] 闭合等待中... step={step}, grasp_wait_counter={self.grasp_wait_counter}/{self.config.grasp_wait_steps}, "
                        f"actual_norm={actual_norm:.4f}, predicted_norm={predicted_norm:.4f}"
                    )
            else:
                # 闭合等待结束后，开始力判据检测
                # 计算实际力与预测力的比值
                if predicted_norm > 1e-6:
                    force_ratio = actual_norm / predicted_norm
                else:
                    force_ratio = 0.0
                
                # 检查比值是否达标
                if force_ratio >= self.config.force_ratio_threshold:
                    # 实际力与预测力匹配 → 抓取正常
                    self.force_sustain_count += 1
                    if self.grasp_wait_counter % 10 == 0 or self.force_sustain_count == 1:
                        print(
                            f"[FORCE] 力比值达标计数+1, step={step}, force_ratio={force_ratio:.4f} >= {self.config.force_ratio_threshold}, "
                            f"actual_norm={actual_norm:.4f}, predicted_norm={predicted_norm:.4f}, "
                            f"sustain_count={self.force_sustain_count}/{self.config.sustain_steps}"
                        )
                    if self.force_sustain_count >= self.config.sustain_steps:
                        self.gripper_state = GripperState.GRASP_CONFIRMED
                        self.force_sustain_count = 0
                        print(
                            f"[FORCE] ✓ GRASP_CONFIRMED, step={step}, "
                            f"force_ratio={force_ratio:.4f} >= {self.config.force_ratio_threshold}, "
                            f"actual_norm={actual_norm:.4f}, predicted_norm={predicted_norm:.4f}, "
                            f"sustain_count={self.force_sustain_count} >= {self.config.sustain_steps}"
                        )
                else:
                    # 比值不达标 → 实际力远小于预测力 → 失败
                    self.force_fail_count += 1
                    if self.force_fail_count == 1 or self.force_fail_count % 5 == 0:
                        print(
                            f"[FORCE] 力比值不达标，失败计数+1, step={step}, force_ratio={force_ratio:.4f} < {self.config.force_ratio_threshold}, "
                            f"actual_norm={actual_norm:.4f}, predicted_norm={predicted_norm:.4f}, "
                            f"fail_count={self.force_fail_count}/{self.config.sustain_steps}"
                        )
                    if self.force_fail_count >= self.config.sustain_steps:
                        print(
                            f"[FORCE] ✗ FAILED_NO_GRASP (闭合等待后没接触), step={step}, "
                            f"force_ratio={force_ratio:.4f} < {self.config.force_ratio_threshold}, "
                            f"actual_norm={actual_norm:.4f}, predicted_norm={predicted_norm:.4f}, "
                            f"grasp_wait_counter={self.grasp_wait_counter}, "
                            f"fail_count={self.force_fail_count} >= {self.config.sustain_steps}"
                        )
                        return "FAILED_NO_GRASP"
        
        # 记录状态机轨迹
        self.state_trajectory.append((
            step,
            self.gripper_state.value,
            actual_norm,
            predicted_norm
        ))
        
        return None

    def check_rollback_condition(
        self,
        actual_gripper_pos: float,
        actual_force: np.ndarray,
        predicted_force: np.ndarray,
        predicted_gripper_pos: float,
    ) -> bool:
        """
        统一回滚检测接口（新版状态机）
        
        返回：是否需要回滚
        """
        if not self.config.enabled:
            return False
        
        self.step_counter += 1
        self.steps_since_rollback += 1
        
        # 回滚后冷却期内不检测
        if self.steps_since_rollback < self.post_rollback_cooldown_steps:
            return False
        
        # 排除初始阶段
        if self.step_counter < self.config.min_start_steps:
            return False
        
        # 保存初始 gripper 值（第一次调用时）
        if self.initial_gripper_value is None:
            self.initial_gripper_value = actual_gripper_pos
            # 动态计算闭合阈值：基于 initial gripper 的 15%
            self.gripper_close_threshold = self.initial_gripper_value
            print(f"[GRIPPER] 初始 gripper 值: {self.initial_gripper_value:.2f}, close_threshold={self.gripper_close_threshold:.2f}")
        
        # 更新gripper状态机
        self._update_gripper_state(actual_gripper_pos, self.step_counter)
        
        # 更新力判据
        failure_reason = None
        if self.config.use_force_check:
            failure_reason = self._update_force_condition(
                actual_force, predicted_force, actual_gripper_pos, predicted_gripper_pos, self.step_counter
            )
        
        # 判断是否需要回滚
        need_rollback = False
        
        # 情况1：状态机直接判定失败
        if failure_reason is not None:
            need_rollback = True
            print(
                f"[ROLLBACK] ✗ 状态机失败: {failure_reason}, "
                f"step={self.step_counter}"
            )
        
        # 情况2：gripper状态为FAILED_NO_GRASP
        if self.gripper_state == GripperState.FAILED_NO_GRASP:
            need_rollback = True
            print(
                f"[ROLLBACK]gripper状态: {self.gripper_state.value}, "
                f"step={self.step_counter}"
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
        self.steps_since_rollback = 0
        
        # 重置状态机
        self.gripper_state = GripperState.CLOSING
        self.stable_step_count = 0
        self.force_sustain_count = 0
        self.force_fail_count = 0
        self.closing_start_step = self.step_counter
        self.settle_step_count = 0
        self.grasp_wait_counter = 0
        self.gripper_max_value = None
        self.gripper_has_closed = False
        self.gripper_close_threshold = None
        self.closing_detected_step = None
        self.stable_candidate_start_step = None
        
        # 重置滤波器
        self.actual_force_filter.reset()
        self.predicted_force_filter.reset()
        
        # 清空历史
        self.actual_gripper_history.clear()
        self.predicted_norm_history.clear()
        
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
        self.step_counter = 0
        self.steps_since_rollback = 999
        
        # 重置状态机
        self.gripper_state = GripperState.CLOSING
        self.stable_step_count = 0
        self.force_sustain_count = 0
        self.force_fail_count = 0
        self.closing_start_step = 0
        self.settle_step_count = 0
        self.initial_gripper_value = None
        self.grasp_wait_counter = 0
        self.gripper_max_value = None
        self.gripper_has_closed = False
        self.gripper_close_threshold = None
        self.closing_detected_step = None
        self.stable_candidate_start_step = None
        
        # 重置滤波器
        self.actual_force_filter.reset()
        self.predicted_force_filter.reset()
        
        # 清空历史
        self.actual_gripper_history.clear()
        self.predicted_norm_history.clear()
        self.state_trajectory.clear()
        
        print("[ROLLBACK RESET ALL] 完全重置所有状态")
    
    def get_state_trajectory(self) -> list:
        """获取状态机轨迹（用于保存到npz文件）"""
        return self.state_trajectory.copy()