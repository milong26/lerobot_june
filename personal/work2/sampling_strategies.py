"""数据配方采样器。见 SPEC.md 4.3 节。"""

from typing import Protocol
import numpy as np
from personal.work2.mw_common.task_ranges import KNOWN_RANGES
from personal.work2.mw_common.state_injection import validate_pick_place_pair


class SamplingStrategy(Protocol):
    def sample(self, n: int, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
        """返回 n 个 (obj_pos_3d, goal_pos_3d) 对,且必须满足该任务的拒绝采样约束。"""
        ...


class UniformRandomStrategy:
    """在 obj_low/high 和 goal_low/high 各自独立均匀采样,拒绝不满足约束的组合。"""

    def __init__(self, task_name: str = "pick-place-v3"):
        self.task_name = task_name
        info = KNOWN_RANGES[task_name]
        self.obj_low = np.array(info["obj_low"])
        self.obj_high = np.array(info["obj_high"])
        self.goal_low = np.array(info["goal_low"])
        self.goal_high = np.array(info["goal_high"])
        self.min_planar_dist = 0.15

    def sample(self, n: int, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
        results = []
        max_attempts = n * 50
        attempts = 0
        while len(results) < n and attempts < max_attempts:
            obj = rng.uniform(self.obj_low, self.obj_high)
            goal = rng.uniform(self.goal_low, self.goal_high)
            attempts += 1
            if validate_pick_place_pair(obj[:2], goal[:2], self.min_planar_dist):
                results.append((obj, goal))
        if len(results) < n:
            raise RuntimeError(
                f"UniformRandomStrategy: 只采样到 {len(results)}/{n} 个合法样本 "
                f"(尝试了 {max_attempts} 次),可能需要调整约束或范围。"
            )
        return results


class GridStrategy:
    """在 obj (x,y) 和 goal (x,y,z) 上打规则网格,过滤非法组合。"""

    def __init__(self, task_name: str = "pick-place-v3",
                 n_obj_per_axis: int = 3, n_goal_per_axis: int = 3):
        self.task_name = task_name
        info = KNOWN_RANGES[task_name]
        self.obj_low = np.array(info["obj_low"])
        self.obj_high = np.array(info["obj_high"])
        self.goal_low = np.array(info["goal_low"])
        self.goal_high = np.array(info["goal_high"])
        self.n_obj_per_axis = n_obj_per_axis
        self.n_goal_per_axis = n_goal_per_axis
        self.min_planar_dist = 0.15

    def sample(self, n: int, rng: np.random.Generator | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
        obj_x = np.linspace(self.obj_low[0], self.obj_high[0], self.n_obj_per_axis)
        obj_y = np.linspace(self.obj_low[1], self.obj_high[1], self.n_obj_per_axis)
        obj_z = np.array([self.obj_low[2]])  # z 通常是常数

        goal_x = np.linspace(self.goal_low[0], self.goal_high[0], self.n_goal_per_axis)
        goal_y = np.linspace(self.goal_low[1], self.goal_high[1], self.n_goal_per_axis)
        goal_z = np.linspace(self.goal_low[2], self.goal_high[2], self.n_goal_per_axis)

        results = []
        for ox in obj_x:
            for oy in obj_y:
                for oz in obj_z:
                    for gx in goal_x:
                        for gy in goal_y:
                            for gz in goal_z:
                                obj = np.array([ox, oy, oz])
                                goal = np.array([gx, gy, gz])
                                if validate_pick_place_pair(obj[:2], goal[:2], self.min_planar_dist):
                                    results.append((obj, goal))

        if n > 0 and len(results) > n:
            if rng is None:
                rng = np.random.default_rng(42)
            indices = rng.choice(len(results), n, replace=False)
            results = [results[i] for i in indices]

        return results


class BoundaryBiasedStrategy:
    """以一定概率偏向 box 边界采样,其余均匀。"""

    def __init__(self, task_name: str = "pick-place-v3",
                 boundary_prob: float = 0.3, boundary_width: float = 0.02):
        self.task_name = task_name
        info = KNOWN_RANGES[task_name]
        self.obj_low = np.array(info["obj_low"])
        self.obj_high = np.array(info["obj_high"])
        self.goal_low = np.array(info["goal_low"])
        self.goal_high = np.array(info["goal_high"])
        self.boundary_prob = boundary_prob
        self.boundary_width = boundary_width
        self.min_planar_dist = 0.15

    def _sample_with_bias(self, low, high, rng):
        if rng.random() < self.boundary_prob:
            if rng.random() < 0.5:
                return rng.uniform(low, low + self.boundary_width)
            else:
                return rng.uniform(high - self.boundary_width, high)
        return rng.uniform(low, high)

    def sample(self, n: int, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
        results = []
        max_attempts = n * 50
        attempts = 0
        while len(results) < n and attempts < max_attempts:
            obj = np.array([
                self._sample_with_bias(self.obj_low[i], self.obj_high[i], rng)
                for i in range(3)
            ])
            goal = np.array([
                self._sample_with_bias(self.goal_low[i], self.goal_high[i], rng)
                for i in range(3)
            ])
            attempts += 1
            if validate_pick_place_pair(obj[:2], goal[:2], self.min_planar_dist):
                results.append((obj, goal))
        if len(results) < n:
            raise RuntimeError(
                f"BoundaryBiasedStrategy: 只采样到 {len(results)}/{n} 个合法样本"
            )
        return results


class DistanceStratifiedStrategy:
    """按 obj-goal 平面距离分层,各层等量采样。"""

    def __init__(self, task_name: str = "pick-place-v3",
                 bins: list[float] | None = None):
        self.task_name = task_name
        info = KNOWN_RANGES[task_name]
        self.obj_low = np.array(info["obj_low"])
        self.obj_high = np.array(info["obj_high"])
        self.goal_low = np.array(info["goal_low"])
        self.goal_high = np.array(info["goal_high"])
        self.min_planar_dist = 0.15
        self.bins = bins or [0.15, 0.25, 0.35, 0.5]

    def sample(self, n: int, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
        per_bin = max(1, n // (len(self.bins) - 1))
        results = []
        max_attempts = per_bin * 100

        for i in range(len(self.bins) - 1):
            low_d, high_d = self.bins[i], self.bins[i + 1]
            bin_results = []
            attempts = 0
            while len(bin_results) < per_bin and attempts < max_attempts:
                obj = rng.uniform(self.obj_low, self.obj_high)
                goal = rng.uniform(self.goal_low, self.goal_high)
                attempts += 1
                d = np.linalg.norm(obj[:2] - goal[:2])
                if low_d <= d < high_d and d >= self.min_planar_dist:
                    bin_results.append((obj, goal))
            results.extend(bin_results)

        if len(results) < n:
            raise RuntimeError(
                f"DistanceStratifiedStrategy: 只采样到 {len(results)}/{n} 个合法样本"
            )
        return results


STRATEGY_MAP = {
    "uniform": UniformRandomStrategy,
    "grid": GridStrategy,
    "boundary": BoundaryBiasedStrategy,
    "distance_stratified": DistanceStratifiedStrategy,
}


def get_strategy(name: str, **kwargs):
    if name not in STRATEGY_MAP:
        raise ValueError(f"未知策略: {name}, 可选: {list(STRATEGY_MAP.keys())}")
    return STRATEGY_MAP[name](**kwargs)