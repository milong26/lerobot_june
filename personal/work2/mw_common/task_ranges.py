"""每个任务 _random_reset_space 的数值范围工具。见 SPEC.md 4.2 节。"""

import numpy as np
import metaworld


KNOWN_RANGES = {
    "pick-place-v3": {
        "obj_low": (-0.1, 0.6, 0.02),
        "obj_high": (0.1, 0.7, 0.02),
        "goal_low": (-0.1, 0.8, 0.05),
        "goal_high": (0.1, 0.9, 0.30),
        "rand_vec_layout": "obj(3)+goal(3)",
        "rejection_constraint": "planar_dist(obj_xy, goal_xy) >= 0.15",
    },
}


def introspect_range(task_name: str, seed: int = 42):
    """自省任意任务的 _random_reset_space,不用一个个去读源码。"""
    mt1 = metaworld.MT1(task_name, seed=seed)
    env = mt1.train_classes[task_name](render_mode="rgb_array")
    env.set_task(mt1.train_tasks[0])
    space = env._random_reset_space
    result = {
        "low": space.low.tolist(),
        "high": space.high.tolist(),
        "shape": space.shape,
    }
    env.close()
    return result


def empirical_estimate_range(task_name: str, n_samples: int = 200, seed: int = 42):
    """经验性交叉验证:多次 reset 用默认随机机制,统计 obj_init_pos 实际落点范围,
    跟 introspect_range() 的解析值做交叉核对。"""
    rng = np.random.default_rng(seed)
    obj_positions = []
    goal_positions = []

    for i in range(n_samples):
        mt1 = metaworld.MT1(task_name, seed=int(rng.integers(0, 100000)))
        env = mt1.train_classes[task_name](render_mode="rgb_array")
        env.set_task(mt1.train_tasks[0])
        env._freeze_rand_vec = True
        env.reset()
        obj_positions.append(env.obj_init_pos.copy())
        goal_positions.append(env.goal.copy())
        env.close()

    obj_arr = np.array(obj_positions)
    goal_arr = np.array(goal_positions)

    return {
        "obj_low": obj_arr.min(axis=0).tolist(),
        "obj_high": obj_arr.max(axis=0).tolist(),
        "obj_mean": obj_arr.mean(axis=0).tolist(),
        "goal_low": goal_arr.min(axis=0).tolist(),
        "goal_high": goal_arr.max(axis=0).tolist(),
        "goal_mean": goal_arr.mean(axis=0).tolist(),
        "n_samples": n_samples,
    }


if __name__ == "__main__":
    print("=== task_ranges.py 验证 ===")
    print("\n--- pick-place-v3 已知范围 (来自源码) ---")
    info = KNOWN_RANGES["pick-place-v3"]
    for k, v in info.items():
        print(f"  {k}: {v}")

    print("\n--- introspect_range 结果 ---")
    result = introspect_range("pick-place-v3")
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n--- empirical_estimate_range (n=50) ---")
    emp = empirical_estimate_range("pick-place-v3", n_samples=50)
    for k, v in emp.items():
        print(f"  {k}: {v}")