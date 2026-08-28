"""
Layout扰动模块（目标位置扰动）

在Meta-World pick-place-v3任务中扰动goal位置，保持物体初始位置不变。
这是方案A：goal/托盘位置和物体初始位置是两个独立可控参数。

扰动定义：
- L0: 原始目标位置（无扰动）
- L1: 轻度偏移（±0.02m）
- L2: 重度偏移（±0.04m）
"""

import numpy as np
from typing import Dict, Any, Optional


def perturb_layout(
    env,
    level: str,
    obj_pos: np.ndarray,
    original_goal_pos: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    扰动目标位置（goal position）

    在env reset前调用，修改env的goal位置。

    参数:
        env: Meta-World环境实例
        level: 扰动等级，"L0"表示无扰动
        obj_pos: 物体初始位置 [x, y, z]（保持不变）
        original_goal_pos: 原始目标位置 [x, y, z]
        params: 扰动参数字典，来自probe_config.yaml
        rng: 随机数生成器（可选，用于可复现性）

    返回:
        new_goal_pos: 扰动后的目标位置
    """
    if level == "L0" or params is None:
        return original_goal_pos.copy()

    if rng is None:
        rng = np.random.default_rng()

    # 获取偏移范围
    goal_offset_range = params.get("goal_offset_range", [0.02, 0.02, 0.02])
    goal_offset_range = np.array(goal_offset_range)

    # 在偏移范围内随机采样
    offset = rng.uniform(-goal_offset_range, goal_offset_range)
    new_goal_pos = original_goal_pos + offset

    # 确保新goal与物体位置的距离 >= 0.15m（避免太近导致任务不合理）
    min_dist = 0.15
    current_dist = np.linalg.norm(new_goal_pos[:2] - obj_pos[:2])
    if current_dist < min_dist:
        # 沿原方向推远
        direction = new_goal_pos[:2] - obj_pos[:2]
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([1.0, 0.0])
        direction = direction / np.linalg.norm(direction)
        new_goal_pos[:2] = obj_pos[:2] + direction * min_dist

    # 应用扰动：修改 _last_rand_vec 并冻结随机向量
    # Meta-World的 SawyerPickPlaceEnvV3 在 reset_model() 中会调用 _get_state_rand_vec()
    # 如果 _freeze_rand_vec=True，会直接返回 _last_rand_vec，不会重新采样
    # rand_vec 的结构是 [obj_pos(3), goal_pos(3)]，共6维
    new_rand_vec = np.hstack([obj_pos.copy(), new_goal_pos.copy()]).astype(np.float64)
    
    if hasattr(env, '_last_rand_vec') and hasattr(env, '_freeze_rand_vec'):
        env._last_rand_vec = new_rand_vec
        env._freeze_rand_vec = True
    
    # 同时更新 env.goal，确保 reset_model() 中的 self._target_pos = self.goal.copy() 也正确
    if hasattr(env, 'goal'):
        env.goal = new_goal_pos.copy()
    
    # 备选方案：如果环境没有这些属性，尝试修改 _get_state_rand_vec
    if not hasattr(env, '_last_rand_vec') and hasattr(env, '_get_state_rand_vec'):
        env._get_state_rand_vec = lambda: new_rand_vec.copy()

    return new_goal_pos


def get_layout_config(level: str) -> Dict[str, Any]:
    """
    获取预定义的layout扰动配置

    参数:
        level: "L0", "L1", 或 "L2"

    返回:
        扰动参数字典
    """
    configs = {
        "L0": None,
        "L1": {
            "goal_offset_range": [0.02, 0.02, 0.02],
        },
        "L2": {
            "goal_offset_range": [0.04, 0.04, 0.04],
        },
    }
    return configs.get(level, None)


if __name__ == "__main__":
    print("Layout扰动模块测试")
    print("L0:", get_layout_config("L0"))
    print("L1:", get_layout_config("L1"))
    print("L2:", get_layout_config("L2"))
    print("\nLayout扰动需要在Meta-World环境中测试，请运行run_probe_rollout.py进行集成测试")