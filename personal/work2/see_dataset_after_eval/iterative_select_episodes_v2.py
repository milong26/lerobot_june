#!/usr/bin/env python
"""
迭代式 SIC Episode 选择 V2

与 V1 的核心区别：
1. 固定 reference anchor universe A（从所有 embeddings 构建，全程不变）
2. dbar_global / dbar_wrist 只计算一次，所有 candidate 共享
3. 预计算 kernel matrix，避免重复 pairwise distance 计算
4. 真正的 sequential greedy：每步只选一个 episode，立即更新 sigma
5. 支持 random / FPS 两种 B0 初始化策略
6. 输出 normalized SIC（除以 N*(1+lambda_wrist)）
7. 详细的诊断信息输出

输出文件：
- iterative_selection_result.json
- subset.json
- sic_curve.json
"""

import sys
import os
import json
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

sys.stdout.reconfigure(line_buffering=True)

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sic_v2 import FixedAnchorSIC, tau


def load_embeddings(embeddings_dir: Path) -> Dict[int, Dict]:
    """
    加载缓存的嵌入

    参数:
        embeddings_dir: 嵌入缓存目录

    返回:
        Dict[episode_index, {"phi_global": ..., "phi_wrist": ...}]
    """
    embeddings = {}

    for f in embeddings_dir.glob("*.npy"):
        try:
            data = np.load(f, allow_pickle=True).item()
            ep_idx = data.get("episode_index")
            if ep_idx is not None:
                embeddings[ep_idx] = {
                    "phi_global": data["phi_global"],
                    "phi_wrist": data["phi_wrist"]
                }
        except Exception as e:
            print(f"  跳过无效文件: {f.name} ({e})")

    print(f"加载了 {len(embeddings)} 个 episode 嵌入")
    return embeddings


def select_initial_b0_random(
    all_episode_indices: List[int],
    b0_size: int,
    seed: int = 42
) -> List[int]:
    """
    随机采样初始 B0 episodes

    必须完全复现 V1 的 seed=42 行为，方便公平 ablation。

    参数:
        all_episode_indices: 所有可用的 episode 索引
        b0_size: B0 大小
        seed: 随机种子

    返回:
        B0 episode 索引列表
    """
    rng = np.random.RandomState(seed)
    selected = rng.choice(all_episode_indices, size=b0_size, replace=False).tolist()
    selected.sort()
    return selected


def select_initial_b0_fps(
    all_episode_indices: List[int],
    phi_globals: np.ndarray,
    phi_wrists: np.ndarray,
    b0_size: int,
    lambda_wrist: float = 1.0,
    seed: int = 42
) -> List[int]:
    """
    基于 embedding 的 farthest-point sampling

    使用 combined distance: distance_global + lambda_wrist * distance_wrist

    参数:
        all_episode_indices: 所有可用的 episode 索引
        phi_globals: shape (N, D_global)
        phi_wrists: shape (N, D_wrist)
        b0_size: B0 大小
        lambda_wrist: wrist 权重
        seed: 随机种子（用于选择第一个点）

    返回:
        B0 episode 索引列表
    """
    rng = np.random.RandomState(seed)
    n = len(all_episode_indices)
    episode_to_idx = {ep: i for i, ep in enumerate(all_episode_indices)}

    phi_g_norm = phi_globals / (np.linalg.norm(phi_globals, axis=1, keepdims=True) + 1e-8)
    phi_w_norm = phi_wrists / (np.linalg.norm(phi_wrists, axis=1, keepdims=True) + 1e-8)

    first_idx = rng.randint(0, n)
    selected_indices = [first_idx]
    remaining = set(range(n))
    remaining.remove(first_idx)

    min_dist = np.full(n, np.inf)

    for _ in range(b0_size - 1):
        last_idx = selected_indices[-1]

        dist_g = np.linalg.norm(phi_g_norm - phi_g_norm[last_idx], axis=1)
        dist_w = np.linalg.norm(phi_w_norm - phi_w_norm[last_idx], axis=1)
        combined_dist = dist_g + lambda_wrist * dist_w

        for idx in remaining:
            if combined_dist[idx] < min_dist[idx]:
                min_dist[idx] = combined_dist[idx]

        best_idx = max(remaining, key=lambda i: min_dist[i])
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    b0_episodes = sorted([all_episode_indices[i] for i in selected_indices])
    return b0_episodes


def sequential_greedy_select(
    sic_calculator: FixedAnchorSIC,
    all_episode_indices: List[int],
    b0_episodes: List[int],
    target_size: int,
    n_add_per_round: int = 9
) -> Dict:
    """
    真正的 sequential greedy episode selection

    每步只选择一个 episode，立即更新 sigma，然后重新评价所有 remaining candidates。

    参数:
        sic_calculator: 固定 anchor SIC 计算器
        all_episode_indices: 所有 episode 索引
        b0_episodes: B0 episode 索引
        target_size: 目标 episode 数量
        n_add_per_round: 每多少个 selection 记为一个 round（用于日志）

    返回:
        选择结果字典
    """
    sic_calculator.initialize_b0(b0_episodes)

    selected_set = set(b0_episodes)
    remaining_episodes = [ep for ep in all_episode_indices if ep not in selected_set]

    selection_steps = []
    sic_history = [sic_calculator.get_current_sic()]
    round_log = []

    current_sic = sic_calculator.get_current_sic()
    round_start_sic = current_sic
    round_start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Sequential Greedy Episode Selection V2")
    print(f"{'='*60}")
    print(f"Reference anchor count: {sic_calculator.reference_anchor_count}")
    print(f"dbar_global: {sic_calculator.dbar_global:.6f}")
    print(f"dbar_wrist: {sic_calculator.dbar_wrist:.6f}")
    print(f"dbar_fallback_used: {sic_calculator.dbar_fallback_used}")
    print(f"Initial B0 SIC: {current_sic:.6f}")
    print(f"Target size: {target_size}")
    print(f"{'='*60}")

    step = 0
    while len(selected_set) < target_size and remaining_episodes:
        step += 1
        step_start = time.time()

        candidate_scores = sic_calculator.score_candidates(remaining_episodes)

        if not candidate_scores:
            print(f"  Step {step}: No valid candidates remaining")
            break

        gains = np.array(list(candidate_scores.values()))
        episodes = list(candidate_scores.keys())

        best_idx = np.argmax(gains)
        best_episode = episodes[best_idx]
        best_gain = gains[best_idx]

        select_info = sic_calculator.select_episode(best_episode)

        selected_set.add(best_episode)
        remaining_episodes.remove(best_episode)

        new_sic = sic_calculator.get_current_sic()

        step_time = time.time() - step_start

        step_record = {
            "step": step,
            "episode_index": best_episode,
            "marginal_gain": float(best_gain),
            "sic_before": float(select_info["sic_before"]),
            "sic_after": float(select_info["sic_after"]),
            "max_candidate_gain": float(np.max(gains)),
            "median_candidate_gain": float(np.median(gains)),
            "min_candidate_gain": float(np.min(gains)),
            "time_seconds": float(step_time)
        }
        selection_steps.append(step_record)
        sic_history.append(new_sic)

        is_round_boundary = (len(selected_set) - len(b0_episodes)) % n_add_per_round == 0
        is_final = len(selected_set) >= target_size

        if is_round_boundary or is_final:
            round_num = (len(selected_set) - len(b0_episodes) + n_add_per_round - 1) // n_add_per_round
            round_gain = new_sic - round_start_sic
            round_time = time.time() - round_start_time

            round_record = {
                "round": round_num,
                "n_selected": len(selected_set),
                "round_sic_gain": float(round_gain),
                "time_seconds": float(round_time)
            }
            round_log.append(round_record)

            print(f"  Round {round_num}: selected={len(selected_set)}, "
                  f"SIC={new_sic:.4f}, gain={round_gain:.4f}, "
                  f"time={round_time:.2f}s")

            round_start_sic = new_sic
            round_start_time = time.time()

        if step % 10 == 0 or is_final:
            print(f"    Step {step}: ep={best_episode}, "
                  f"gain={best_gain:.6f}, SIC={new_sic:.4f}, "
                  f"remaining={len(remaining_episodes)}")

    total_time = time.time() - round_start_time

    final_sic = sic_calculator.get_current_sic()
    n_episodes = sic_calculator.reference_anchor_count
    normalized_sic = final_sic / (n_episodes * (1 + sic_calculator.lambda_wrist))

    result = {
        "selection_method": "fixed_anchor_sequential_sic_v2",
        "selected_episodes": sic_calculator.get_selected_episodes(),
        "b0_episodes": b0_episodes,
        "b0_strategy": "random",
        "reference_anchor_count": sic_calculator.reference_anchor_count,
        "dbar_global": sic_calculator.dbar_global,
        "dbar_wrist": sic_calculator.dbar_wrist,
        "dbar_fallback_used": sic_calculator.dbar_fallback_used,
        "alpha": sic_calculator.alpha,
        "lambda_wrist": sic_calculator.lambda_wrist,
        "target_size": target_size,
        "n_add_per_round": n_add_per_round,
        "selection_steps": selection_steps,
        "round_log": round_log,
        "sic_history": sic_history,
        "initial_sic": sic_history[0],
        "final_sic": final_sic,
        "normalized_sic": normalized_sic,
        "total_steps": len(selection_steps),
        "sigma_stats": sic_calculator.get_sigma_stats(),
        "tau_1": sic_calculator.tau_1,
        "note": "repeat softening currently inactive for unique episode-level selection"
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="迭代式 SIC Episode 选择 V2")
    parser.add_argument("--embeddings-dir", type=str, required=True, help="嵌入缓存目录")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--b0-size", type=int, default=18, help="初始 B0 大小 (默认: 18)")
    parser.add_argument("--target-size", type=int, default=112, help="目标 episode 数量 (默认: 112)")
    parser.add_argument("--n-add-per-round", type=int, default=9, help="每轮添加的 episodes 数 (默认: 9, 用于日志分组)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--alpha", type=float, default=1.0, help="SIC 次数软化系数")
    parser.add_argument("--lambda-wrist", type=float, default=1.0, help="wrist 权重")
    parser.add_argument("--b0-strategy", type=str, default="random", choices=["random", "fps"],
                       help="B0 初始化策略 (默认: random)")

    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)

    if not embeddings_dir.exists():
        print(f"错误: 嵌入目录不存在: {embeddings_dir}")
        return

    print(f"\n加载嵌入...")
    embeddings = load_embeddings(embeddings_dir)

    if not embeddings:
        print("错误: 没有找到嵌入文件")
        return

    all_episode_indices = sorted(embeddings.keys())
    phi_globals = np.array([embeddings[ep]["phi_global"] for ep in all_episode_indices])
    phi_wrists = np.array([embeddings[ep]["phi_wrist"] for ep in all_episode_indices])

    print(f"\n构建 FixedAnchorSIC 计算器...")
    sic_calc = FixedAnchorSIC(
        episode_indices=all_episode_indices,
        phi_globals=phi_globals,
        phi_wrists=phi_wrists,
        alpha=args.alpha,
        lambda_wrist=args.lambda_wrist
    )

    print(f"Reference anchor count: {sic_calc.reference_anchor_count}")
    print(f"dbar_global: {sic_calc.dbar_global:.6f}")
    print(f"dbar_wrist: {sic_calc.dbar_wrist:.6f}")
    print(f"dbar_fallback_used: {sic_calc.dbar_fallback_used}")

    print(f"\n选择 B0 (strategy={args.b0_strategy})...")
    if args.b0_strategy == "random":
        b0_episodes = select_initial_b0_random(
            all_episode_indices, args.b0_size, args.seed
        )
    elif args.b0_strategy == "fps":
        b0_episodes = select_initial_b0_fps(
            all_episode_indices, phi_globals, phi_wrists,
            args.b0_size, args.lambda_wrist, args.seed
        )
    else:
        raise ValueError(f"Unknown b0_strategy: {args.b0_strategy}")

    print(f"B0 episodes: {b0_episodes}")

    overall_start = time.time()

    result = sequential_greedy_select(
        sic_calculator=sic_calc,
        all_episode_indices=all_episode_indices,
        b0_episodes=b0_episodes,
        target_size=args.target_size,
        n_add_per_round=args.n_add_per_round
    )

    overall_time = time.time() - overall_start
    result["total_runtime_seconds"] = overall_time
    result["b0_strategy"] = args.b0_strategy

    print(f"\n{'='*60}")
    print(f"选择完成！")
    print(f"{'='*60}")
    print(f"最终 episodes 数: {len(result['selected_episodes'])}")
    print(f"最终 SIC: {result['final_sic']:.4f}")
    print(f"Normalized SIC: {result['normalized_sic']:.4f}")
    print(f"总耗时: {overall_time:.2f}s")

    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / "iterative_selection_result.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n保存结果到: {result_file}")

    subset_file = output_dir / "subset.json"
    subset_data = {
        "selected_episode_indices": result["selected_episodes"],
        "b0_episodes": result["b0_episodes"],
        "num_episodes": len(result["selected_episodes"]),
        "selection_method": result["selection_method"],
        "parameters": {
            "b0_size": args.b0_size,
            "target_size": args.target_size,
            "n_add_per_round": args.n_add_per_round,
            "seed": args.seed,
            "alpha": args.alpha,
            "lambda_wrist": args.lambda_wrist,
            "b0_strategy": args.b0_strategy
        }
    }
    with open(subset_file, "w") as f:
        json.dump(subset_data, f, indent=2)
    print(f"保存子集到: {subset_file}")

    n_ref = result["reference_anchor_count"]
    sic_curve_data = {
        "steps": list(range(len(result["sic_history"]))),
        "sic_scores": result["sic_history"],
        "normalized_sic_scores": [
            s / (n_ref * (1 + result["lambda_wrist"]))
            for s in result["sic_history"]
        ],
        "initial_sic": result["initial_sic"],
        "final_sic": result["final_sic"],
        "normalized_final_sic": result["normalized_sic"],
        "reference_anchor_count": n_ref
    }
    sic_curve_file = output_dir / "sic_curve.json"
    with open(sic_curve_file, "w") as f:
        json.dump(sic_curve_data, f, indent=2)
    print(f"保存 SIC 曲线到: {sic_curve_file}")

    print(f"\n{'='*60}")
    print(f"完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()