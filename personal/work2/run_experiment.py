"""编排脚本: 生成 → 训练 → 评测 → 汇总。见 SPEC.md 4.8 节。"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

RECIPES = {
    "uniform_50": dict(strategy="uniform", n_episodes=50),
    "grid_50": dict(strategy="grid", n_episodes=50),
    "boundary_50": dict(strategy="boundary", n_episodes=50),
}

WORK_DIR = Path(__file__).parent
GENERATED_DIR = WORK_DIR / "generated"
RESULTS_DIR = WORK_DIR / "results"
LOGS_DIR = WORK_DIR / "logs"


def run_cmd(cmd: list[str], log_path: Path, check: bool = True) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True, check=check)


def main():
    parser = argparse.ArgumentParser(description="编排实验: 生成 → 训练 → 评测")
    parser.add_argument("--recipes", type=str, default=None,
                        help="逗号分隔的配方名,如 uniform_50,grid_50 (默认全部)")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--task", type=str, default="pick-place-v3")
    parser.add_argument("--eval-set", type=str, default=str(WORK_DIR / "fixed_eval_set.json"))
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    if args.recipes:
        recipe_names = [r.strip() for r in args.recipes.split(",")]
        recipes = {k: v for k, v in RECIPES.items() if k in recipe_names}
    else:
        recipes = RECIPES

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    for name, cfg in recipes.items():
        print(f"\n{'=' * 60}")
        print(f"Recipe: {name}")
        print(f"{'=' * 60}")

        dataset_root = GENERATED_DIR / name
        train_output = Path(f"outputs/exp_{name}")
        log_prefix = LOGS_DIR / name

        # Step 1: Generate
        if not args.skip_generate:
            gen_log = log_prefix.with_suffix(".generate.log")
            if dataset_root.exists() and (dataset_root / "meta" / "info.json").exists():
                print(f"  [SKIP] 数据集已存在: {dataset_root}")
            else:
                print(f"  [GENERATE] {name} -> {dataset_root}")
                cmd = [
                    sys.executable, str(WORK_DIR / "generate_dataset.py"),
                    "--task", args.task,
                    "--strategy", cfg["strategy"],
                    "--num-episodes", str(cfg["n_episodes"]),
                    "--output-dir", str(dataset_root),
                    "--repo-id", f"test/{name}",
                ]
                run_cmd(cmd, gen_log)
                print(f"  [DONE] 生成完成, 日志: {gen_log}")

        # Step 2: Train
        if not args.skip_train:
            train_log = log_prefix.with_suffix(".train.log")
            if (train_output / "checkpoints" / "last").exists():
                print(f"  [SKIP] 训练产物已存在: {train_output}")
            else:
                print(f"  [TRAIN] {name} -> {train_output}")
                cmd = [
                    "lerobot-train",
                    f"--dataset.root={dataset_root}",
                    f"--output_dir={train_output}",
                    f"--steps={args.steps}",
                    '--remove_features=["observation.environment_state"]',
                ]
                run_cmd(cmd, train_log)
                print(f"  [DONE] 训练完成, 日志: {train_log}")

        # Step 3: Eval
        if not args.skip_eval:
            eval_log = log_prefix.with_suffix(".eval.log")
            out_csv = RESULTS_DIR / f"{name}.csv"
            if out_csv.exists():
                print(f"  [SKIP] 评测结果已存在: {out_csv}")
            else:
                policy_path = train_output / "checkpoints" / "last"
                if not policy_path.exists():
                    print(f"  [WARN] 策略不存在: {policy_path}, 跳过评测")
                    continue
                print(f"  [EVAL] {name} -> {out_csv}")
                cmd = [
                    sys.executable, str(WORK_DIR / "run_eval_with_states.py"),
                    "--policy-path", str(policy_path),
                    "--eval-set", args.eval_set,
                    "--out-csv", str(out_csv),
                ]
                run_cmd(cmd, eval_log)
                print(f"  [DONE] 评测完成, 日志: {eval_log}")

    print(f"\n{'=' * 60}")
    print("所有配方处理完成!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()