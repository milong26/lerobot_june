"""生成一份固定的、所有数据配方共用的评测集。见 SPEC.md 4.5 节。"""

import argparse
import json
import numpy as np
from pathlib import Path

from personal.work2.sampling_strategies import UniformRandomStrategy


def main():
    parser = argparse.ArgumentParser(description="生成固定评测集")
    parser.add_argument("--task", type=str, default="pick-place-v3")
    parser.add_argument("--n-states", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="personal/work2/fixed_eval_set.json")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    strategy = UniformRandomStrategy(task_name=args.task)
    samples = strategy.sample(args.n_states, rng)

    states = []
    for obj, goal in samples:
        states.append({
            "obj_pos": obj.tolist(),
            "goal_pos": goal.tolist(),
        })

    output = {
        "task": args.task,
        "n_states": len(states),
        "seed": args.seed,
        "states": states,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"已生成 {len(states)} 个评测状态，保存至 {output_path}")


if __name__ == "__main__":
    main()