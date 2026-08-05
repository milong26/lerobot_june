#!/usr/bin/env python
"""Smoke test: 精确指定状态 + expert 策略验证。"""

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
from personal.work2.mw_common.state_injection import make_env_with_fixed_state, validate_pick_place_pair
import metaworld.policies as policies


def get_expert_policy(task_name):
    policy_class_name = f"Sawyer{task_name.replace('-', ' ').title().replace(' ', '')}Policy"
    policy_class = getattr(policies, policy_class_name)
    return policy_class()


def run_episode(env, expert_policy, max_steps=500):
    obs, info = env.reset()
    success = False
    for step in range(max_steps):
        action = expert_policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("success", 0):
            success = True
        if terminated or truncated:
            break
    return success


def main():
    print("=== state_injection smoke test ===")
    task_name = "pick-place-v3"
    expert_policy = get_expert_policy(task_name)

    test_cases = [
        ([0.0, 0.65, 0.02], [0.0, 0.85, 0.2]),
        ([0.05, 0.62, 0.02], [-0.05, 0.88, 0.15]),
        ([-0.08, 0.68, 0.02], [0.08, 0.82, 0.25]),
    ]

    all_passed = True
    for i, (obj, goal) in enumerate(test_cases):
        rand_vec = np.array(obj + goal)
        if not validate_pick_place_pair(obj[:2], goal[:2]):
            print(f"  测试 {i+1}: [SKIP] obj={obj}, goal={goal} -> 不满足约束")
            continue

        env, obs, info = make_env_with_fixed_state(task_name, rand_vec)
        actual_obj = env.obj_init_pos
        actual_goal = env.goal
        obj_ok = np.allclose(actual_obj, obj, atol=1e-6)
        goal_ok = np.allclose(actual_goal, goal, atol=1e-6)

        success = run_episode(env, expert_policy)

        status = "PASS" if (obj_ok and goal_ok and success) else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"  测试 {i+1}: [{status}] obj={obj} -> {actual_obj.tolist()}, "
              f"goal={goal} -> {actual_goal.tolist()}, success={success}")
        env.close()

    if all_passed:
        print("\n全部 smoke test 通过!")
    else:
        print("\n有测试失败!")


if __name__ == "__main__":
    main()