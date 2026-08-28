"""统计分析脚本。见 SPEC.md 4.9 节。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_csv(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def load_episode_metadata(dataset_dir: Path) -> list[dict]:
    meta_file = dataset_dir / "episode_initial_states.json"
    if not meta_file.exists():
        return []
    with open(meta_file) as f:
        return json.load(f).get("episodes", [])


def compute_nearest_dist(eval_row, train_samples):
    """计算评测点到最近训练样本的欧氏距离(仅 obj_pos)。"""
    eval_obj = np.array([float(eval_row["obj_pos_x"]), float(eval_row["obj_pos_y"]), float(eval_row["obj_pos_z"])])
    if not train_samples:
        return float("inf")
    train_objs = np.array([[s["obj_init_pos"][0], s["obj_init_pos"][1], s["obj_init_pos"][2]] for s in train_samples])
    dists = np.linalg.norm(train_objs - eval_obj, axis=1)
    return float(dists.min())


def main():
    parser = argparse.ArgumentParser(description="分析实验结果")
    parser.add_argument("--results-dir", type=str, default="personal/work2/results")
    parser.add_argument("--generated-dir", type=str, default="personal/work2/generated")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    generated_dir = Path(args.generated_dir)

    if not results_dir.exists():
        print(f"错误: 结果目录不存在: {results_dir}")
        return

    csv_files = list(results_dir.glob("*.csv"))
    if not csv_files:
        print(f"错误: 没有找到 CSV 文件 in {results_dir}")
        return

    print("=" * 60)
    print("配方 × 成功率汇总表")
    print("=" * 60)
    print(f"{'配方':<20} | {'成功率':>8} | {'评测数':>6}")
    print("-" * 40)

    summary = []
    for csv_file in sorted(csv_files):
        rows = load_csv(csv_file)
        if not rows:
            continue
        recipe = csv_file.stem
        n_total = len(rows)
        n_success = sum(1 for r in rows if r["success"].lower() in ("true", "1"))
        rate = n_success / n_total * 100 if n_total > 0 else 0
        summary.append({"recipe": recipe, "rate": rate, "n_success": n_success, "n_total": n_total})
        print(f"{recipe:<20} | {rate:>7.1f}% | {n_total:>6}")

    print("=" * 60)

    # 距离分析
    print("\n失败点到最近训练样本的距离分析:")
    print("-" * 60)
    for s in summary:
        csv_file = results_dir / f"{s['recipe']}.csv"
        rows = load_csv(csv_file)
        train_dir = generated_dir / s["recipe"]
        train_samples = [ep for ep in load_episode_metadata(train_dir) if ep.get("included", True)]

        fail_dists = []
        success_dists = []
        for row in rows:
            d = compute_nearest_dist(row, train_samples)
            if row["success"].lower() in ("true", "1"):
                success_dists.append(d)
            else:
                fail_dists.append(d)

        if fail_dists:
            print(f"  {s['recipe']}:")
            print(f"    成功样本平均最近距离: {np.mean(success_dists):.4f} (n={len(success_dists)})")
            print(f"    失败样本平均最近距离: {np.mean(fail_dists):.4f} (n={len(fail_dists)})")
        else:
            print(f"  {s['recipe']}: 无失败样本")


if __name__ == "__main__":
    main()