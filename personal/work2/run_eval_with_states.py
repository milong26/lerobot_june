"""用固定评测集跑 eval,记录逐局结果。见 SPEC.md 4.6 节。"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import numpy as np
import metaworld
import metaworld.policies as policies


def load_eval_set(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def make_env_with_fixed_state(task_name, rand_vec, seed=42, camera_name="corner2"):
    mt1 = metaworld.MT1(task_name, seed=seed)
    env = mt1.train_classes[task_name](render_mode="rgb_array", camera_name=camera_name)
    env.set_task(mt1.train_tasks[0])
    env._freeze_rand_vec = False

    call_count = [0]

    def _patched():
        if call_count[0] > 0:
            raise RuntimeError(f"_get_state_rand_vec called {call_count[0] + 1} times")
        call_count[0] += 1
        return rand_vec.copy()

    env._get_state_rand_vec = _patched
    obs, info = env.reset()
    return env, obs, info


def get_expert_policy(task_name):
    policy_class_name = f"Sawyer{task_name.replace('-', ' ').title().replace(' ', '')}Policy"
    try:
        policy_class = getattr(policies, policy_class_name)
        return policy_class()
    except AttributeError:
        print(f"错误: 找不到任务 {task_name} 的专家策略 {policy_class_name}")
        sys.exit(1)


def run_episode_with_state(env, expert_policy, max_steps=500):
    obs, info = env.reset()
    obj_pos = env.obj_init_pos.copy()
    goal_pos = env.goal.copy()

    total_reward = 0.0
    success = False

    for step in range(max_steps):
        action = expert_policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if info.get("success", 0):
            success = True
        if terminated or truncated:
            break

    return obj_pos, goal_pos, success, total_reward


def main():
    parser = argparse.ArgumentParser(description="用固定评测集跑 eval")
    parser.add_argument("--eval-set", type=str, required=True, help="fixed_eval_set.json 路径")
    parser.add_argument("--out-csv", type=str, default="personal/work2/results/eval_results.csv")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    eval_set = load_eval_set(args.eval_set)
    task_name = eval_set["task"]
    states = eval_set["states"]

    expert_policy = get_expert_policy(task_name)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    success_count = 0

    print(f"=== 评测集: {task_name}, {len(states)} 个状态 ===")
    print("-" * 80)

    for i, state in enumerate(states):
        obj = np.array(state["obj_pos"])
        goal = np.array(state["goal_pos"])
        rand_vec = np.concatenate([obj, goal])

        try:
            env, _, _ = make_env_with_fixed_state(task_name, rand_vec, seed=args.seed)
        except Exception as e:
            print(f"  [{i:3d}] 创建环境失败: {e}")
            results.append({
                "episode_index": i,
                "obj_pos_x": obj[0], "obj_pos_y": obj[1], "obj_pos_z": obj[2],
                "goal_pos_x": goal[0], "goal_pos_y": goal[1], "goal_pos_z": goal[2],
                "success": False, "sum_reward": 0.0, "error": str(e),
            })
            continue

        obj_pos, goal_pos, success, total_reward = run_episode_with_state(env, expert_policy, args.max_steps)
        env.close()

        if success:
            success_count += 1

        results.append({
            "episode_index": i,
            "obj_pos_x": obj_pos[0], "obj_pos_y": obj_pos[1], "obj_pos_z": obj_pos[2],
            "goal_pos_x": goal_pos[0], "goal_pos_y": goal_pos[1], "goal_pos_z": goal_pos[2],
            "success": success, "sum_reward": total_reward, "error": "",
        })

        status = "✓" if success else "✗"
        print(f"  [{i:3d}/{len(states)}] {status} | obj=[{obj[0]:.3f},{obj[1]:.3f},{obj[2]:.3f}] | reward={total_reward:.2f}")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "episode_index", "obj_pos_x", "obj_pos_y", "obj_pos_z",
            "goal_pos_x", "goal_pos_y", "goal_pos_z", "success", "sum_reward", "error",
        ])
        writer.writeheader()
        writer.writerows(results)

    rate = success_count / len(states) * 100
    print(f"\n=== 评测完成 ===")
    print(f"成功率: {success_count}/{len(states)} = {rate:.1f}%")
    print(f"结果已保存至: {out_path}")


if __name__ == "__main__":
    main()