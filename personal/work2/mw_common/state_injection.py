"""通用状态注入器:强制 Meta-World 环境在 reset() 时使用指定的 (obj_pos, goal_pos)。
见 SPEC.md 3.4 节。
"""

import numpy as np
import metaworld


class RandVecExhaustedError(RuntimeError):
    """任务内部的拒绝采样循环拒绝了我们指定的状态(比如 obj/goal 靠得太近),
    而不是死循环或静默换成别的随机状态。"""


def make_env_with_fixed_state(
    task_name: str,
    rand_vec,
    seed: int = 42,
    camera_name: str = "corner2",
    render_mode: str = "rgb_array",
):
    """创建一个 reset() 被强制使用 `rand_vec` 的 Meta-World 环境。

    rand_vec: 长度/语义要匹配 env._random_reset_space 的形状。
        对 pick-place-v3 是 6 个数: [obj_x,obj_y,obj_z, goal_x,goal_y,goal_z]。
        其他任务用 `env._random_reset_space.low` / `.high` 自省。

    注意: _get_state_rand_vec 会被任务内部的拒绝采样循环多次调用，
    我们每次都返回相同的固定值，这样拒绝采样会一直用同一个状态。
    """
    mt1 = metaworld.MT1(task_name, seed=seed)
    env = mt1.train_classes[task_name](render_mode=render_mode, camera_name=camera_name)
    env.set_task(mt1.train_tasks[0])
    env._freeze_rand_vec = False

    def _patched():
        return rand_vec.copy()

    env._get_state_rand_vec = _patched
    obs, info = env.reset()
    return env, obs, info


def validate_pick_place_pair(obj_xy, goal_xy, min_planar_dist=0.15, margin=0.01):
    """在真正调用 make_env_with_fixed_state 之前,先检查是否会触发拒绝采样。"""
    d = np.linalg.norm(np.asarray(obj_xy) - np.asarray(goal_xy))
    return d >= (min_planar_dist + margin)


if __name__ == "__main__":
    print("=== state_injection.py smoke test ===")
    test_cases = [
        ([0.0, 0.65, 0.02], [-0.05, 0.85, 0.1]),
        ([0.05, 0.62, 0.02], [0.0, 0.88, 0.15]),
        ([-0.08, 0.68, 0.02], [0.08, 0.82, 0.25]),
    ]

    for obj, goal in test_cases:
        rand_vec = np.array(obj + goal)
        if not validate_pick_place_pair(obj[:2], goal[:2]):
            print(f"  [SKIP] obj={obj}, goal={goal} -> 不满足平面距离约束")
            continue
        env, obs, info = make_env_with_fixed_state("pick-place-v3", rand_vec)
        actual_obj = env.obj_init_pos
        actual_goal = env.goal
        obj_ok = np.allclose(actual_obj, obj, atol=1e-6)
        goal_ok = np.allclose(actual_goal, goal, atol=1e-6)
        status = "OK" if (obj_ok and goal_ok) else "FAIL"
        print(f"  [{status}] obj: 指定={obj}, 实际={actual_obj.tolist()}")
        print(f"         goal: 指定={goal}, 实际={actual_goal.tolist()}")
        env.close()

    print("\n=== smoke test 完成 ===")