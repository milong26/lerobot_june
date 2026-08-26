#!/usr/bin/env python
"""
迭代式 SIC Episode 选择

基于原始 SIC 函数，迭代选择 episodes：
1. 初始 B0：均匀采样 18 个 episodes
2. 迭代：每轮选择 SIC 增益最大的 episode
3. 停止：达到 112 个 episodes

输出：每个迭代轮次的 episode 子集
"""

import sys
import os
import json
import argparse
import time
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

# 强制禁用 Python 输出缓冲
sys.stdout.reconfigure(line_buffering=True)

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# 添加 project root 到 Python 路径
# 文件路径: /data/zhonglinye/jun/lerobot/personal/work2/our/experiments/iterative_select_episodes.py
# 需要找到: /data/zhonglinye/jun/lerobot/personal/work2/our/
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.anchors import AnchorSystem
from core.sic import compute_sic_score


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


def select_initial_b0(all_episode_indices: List[int], b0_size: int, seed: int = 42) -> List[int]:
    """
    均匀采样初始 B0 episodes
    
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


def compute_sic_for_candidate(
    selected_episodes: List[int],
    candidate_idx: int,
    embeddings: Dict[int, Dict],
    alpha: float = 1.0,
    lambda_wrist: float = 1.0
) -> float:
    """
    计算添加候选 episode 后的 SIC 分数
    
    参数:
        selected_episodes: 已选择的 episodes
        candidate_idx: 候选 episode 索引
        embeddings: 嵌入字典
        alpha: 次数软化系数
        lambda_wrist: wrist 权重
    
    返回:
        SIC 分数
    """
    # 构建候选集（每个 episode 作为 grid_coord 使用 episode_index 代替）
    candidate_set = {}
    for ep_idx in selected_episodes:
        candidate_set[ep_idx] = candidate_set.get(ep_idx, 0) + 1
    
    # 添加候选
    candidate_set[candidate_idx] = candidate_set.get(candidate_idx, 0) + 1
    
    # 构建临时 AnchorSystem
    anchor_system = AnchorSystem()
    for ep_idx in set(list(selected_episodes) + [candidate_idx]):
        if ep_idx in embeddings:
            anchor_system.add_anchor(
                ep_idx,
                embeddings[ep_idx]["phi_global"],
                embeddings[ep_idx]["phi_wrist"]
            )
    
    anchor_system.compute_dbar()
    
    return compute_sic_score(candidate_set, anchor_system, alpha, lambda_wrist)


def iterative_select_episodes(
    embeddings: Dict[int, Dict],
    b0_size: int = 18,
    target_size: int = 112,
    n_add_per_round: int = 5,
    seed: int = 42,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0
) -> Dict:
    """
    迭代式选择 episodes
    
    参数:
        embeddings: 嵌入字典
        b0_size: 初始 B0 大小
        target_size: 目标 episode 数量
        n_add_per_round: 每轮添加的 episodes 数
        seed: 随机种子
        alpha: SIC 次数软化系数
        lambda_wrist: wrist 权重
    
    返回:
        选择结果字典
    """
    all_episode_indices = sorted(embeddings.keys())
    n_total = len(all_episode_indices)
    
    print(f"\n{'='*60}")
    print(f"迭代式 SIC Episode 选择")
    print(f"{'='*60}")
    print(f"总 episodes: {n_total}")
    print(f"初始 B0 大小: {b0_size}")
    print(f"目标大小: {target_size}")
    print(f"每轮添加: {n_add_per_round}")
    print(f"alpha: {alpha}, lambda_wrist: {lambda_wrist}")
    print(f"{'='*60}")
    
    # Step 1: 选择初始 B0
    print(f"\n[Step 1] 选择初始 B0 ({b0_size} episodes)...")
    b0_episodes = select_initial_b0(all_episode_indices, b0_size, seed)
    selected_episodes = list(b0_episodes)
    remaining_episodes = [ep for ep in all_episode_indices if ep not in selected_episodes]
    
    print(f"  B0 episodes: {b0_episodes}")
    
    # 计算初始 SIC
    candidate_set_b0 = {ep: 1 for ep in selected_episodes}
    anchor_system_b0 = AnchorSystem()
    for ep_idx in selected_episodes:
        if ep_idx in embeddings:
            anchor_system_b0.add_anchor(
                ep_idx,
                embeddings[ep_idx]["phi_global"],
                embeddings[ep_idx]["phi_wrist"]
            )
    anchor_system_b0.compute_dbar()
    initial_sic = compute_sic_score(candidate_set_b0, anchor_system_b0, alpha, lambda_wrist)
    
    print(f"  初始 SIC 分数: {initial_sic:.4f}")
    
    # Step 2: 迭代选择
    print(f"\n[Step 2] 开始迭代选择...")
    selection_log = []
    sic_history = [initial_sic]
    round_num = 0
    
    while len(selected_episodes) < target_size and remaining_episodes:
        round_num += 1
        round_start = time.time()
        
        n_to_add = min(n_add_per_round, target_size - len(selected_episodes))
        
        print(f"\n  Round {round_num}: 已选择 {len(selected_episodes)}, 需要添加 {n_to_add}")
        
        # 计算每个候选 episode 的 SIC 增益
        best_candidates = []
        candidate_scores = []
        
        for candidate_idx in remaining_episodes:
            sic_with_candidate = compute_sic_for_candidate(
                selected_episodes, candidate_idx, embeddings, alpha, lambda_wrist
            )
            candidate_scores.append((candidate_idx, sic_with_candidate))
        
        # 按 SIC 分数排序，选择最好的 n_to_add 个
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i in range(n_to_add):
            if i >= len(candidate_scores):
                break
            best_idx, best_sic = candidate_scores[i]
            selected_episodes.append(best_idx)
            remaining_episodes.remove(best_idx)
            best_candidates.append({
                "episode_index": best_idx,
                "sic_score": best_sic
            })
        
        # 重新计算当前 SIC
        candidate_set_current = {ep: 1 for ep in selected_episodes}
        anchor_system_current = AnchorSystem()
        for ep_idx in selected_episodes:
            if ep_idx in embeddings:
                anchor_system_current.add_anchor(
                    ep_idx,
                    embeddings[ep_idx]["phi_global"],
                    embeddings[ep_idx]["phi_wrist"]
                )
        anchor_system_current.compute_dbar()
        current_sic = compute_sic_score(candidate_set_current, anchor_system_current, alpha, lambda_wrist)
        sic_history.append(current_sic)
        
        round_time = time.time() - round_start
        
        # 记录日志
        selection_log.append({
            "round": round_num,
            "n_selected": len(selected_episodes),
            "n_added": len(best_candidates),
            "sic_score": current_sic,
            "sic_gain": current_sic - sic_history[-2] if len(sic_history) > 1 else 0,
            "time_seconds": round_time,
            "added_episodes": best_candidates
        })
        
        print(f"    添加 {len(best_candidates)} 个 episodes")
        print(f"    当前 SIC: {current_sic:.4f} (增益: {current_sic - sic_history[-2]:.4f})")
        print(f"    耗时: {round_time:.2f}s")
    
    # Step 3: 输出结果
    print(f"\n{'='*60}")
    print(f"迭代选择完成！")
    print(f"{'='*60}")
    print(f"最终选择 episodes 数: {len(selected_episodes)}")
    print(f"最终 SIC 分数: {sic_history[-1]:.4f}")
    print(f"SIC 增益: {sic_history[-1] - initial_sic:.4f}")
    print(f"总迭代轮次: {round_num}")
    
    return {
        "selected_episodes": sorted(selected_episodes),
        "b0_episodes": b0_episodes,
        "selection_log": selection_log,
        "sic_history": sic_history,
        "initial_sic": initial_sic,
        "final_sic": sic_history[-1],
        "total_rounds": round_num,
        "target_size": target_size,
        "b0_size": b0_size,
        "n_add_per_round": n_add_per_round,
        "alpha": alpha,
        "lambda_wrist": lambda_wrist,
        "seed": seed
    }


def main():
    parser = argparse.ArgumentParser(description="迭代式 SIC Episode 选择")
    parser.add_argument("--embeddings-dir", type=str, required=True, help="嵌入缓存目录")
    parser.add_argument("--output-dir", type=str, required=True, help="输出目录")
    parser.add_argument("--b0-size", type=int, default=18, help="初始 B0 大小 (默认: 18)")
    parser.add_argument("--target-size", type=int, default=112, help="目标 episode 数量 (默认: 112)")
    parser.add_argument("--n-add-per-round", type=int, default=5, help="每轮添加的 episodes 数 (默认: 5)")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--alpha", type=float, default=1.0, help="SIC 次数软化系数")
    parser.add_argument("--lambda-wrist", type=float, default=1.0, help="wrist 权重")
    
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
    
    # 运行迭代选择
    result = iterative_select_episodes(
        embeddings=embeddings,
        b0_size=args.b0_size,
        target_size=args.target_size,
        n_add_per_round=args.n_add_per_round,
        seed=args.seed,
        alpha=args.alpha,
        lambda_wrist=args.lambda_wrist
    )
    
    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存完整结果
    result_file = output_dir / "iterative_selection_result.json"
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n保存结果到: {result_file}")
    
    # 保存选中的 episodes 列表（用于训练）
    subset_file = output_dir / "subset.json"
    subset_data = {
        "selected_episode_indices": result["selected_episodes"],
        "b0_episodes": result["b0_episodes"],
        "num_episodes": len(result["selected_episodes"]),
        "selection_method": "iterative_sic",
        "parameters": {
            "b0_size": args.b0_size,
            "target_size": args.target_size,
            "n_add_per_round": args.n_add_per_round,
            "seed": args.seed,
            "alpha": args.alpha,
            "lambda_wrist": args.lambda_wrist
        }
    }
    with open(subset_file, "w") as f:
        json.dump(subset_data, f, indent=2)
    print(f"保存子集到: {subset_file}")
    
    # 保存 SIC 曲线数据
    sic_curve_file = output_dir / "sic_curve.json"
    sic_curve_data = {
        "rounds": list(range(len(result["sic_history"]))),
        "sic_scores": result["sic_history"],
        "initial_sic": result["initial_sic"],
        "final_sic": result["final_sic"]
    }
    with open(sic_curve_file, "w") as f:
        json.dump(sic_curve_data, f, indent=2)
    print(f"保存 SIC 曲线到: {sic_curve_file}")
    
    print(f"\n{'='*60}")
    print(f"完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()