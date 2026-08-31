#!/usr/bin/env python
"""
V1 vs V2 offline comparison script

在同一个 fixed reference universe 下公平比较 V1 和 V2 选出的 episode subsets。

注意：
- 不直接比较 V1 原始 SIC 和 V2 SIC（因为 objective definition 不同）
- V1 的最终 subset 用 V2 的固定 reference universe 重新计算 fixed-universe SIC
- 输出 overlap、redundancy、coverage 等诊断指标
"""

import sys
import os
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent))

from sic_v2 import FixedAnchorSIC


def load_embeddings(embeddings_dir: Path) -> Dict[int, Dict]:
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
        except Exception:
            pass
    return embeddings


def compute_fixed_universe_sic(
    selected_episodes: List[int],
    all_episode_indices: List[int],
    phi_globals: np.ndarray,
    phi_wrists: np.ndarray,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0
) -> Dict:
    """
    在固定 reference universe 下计算一个 subset 的 SIC
    """
    sic_calc = FixedAnchorSIC(
        episode_indices=all_episode_indices,
        phi_globals=phi_globals,
        phi_wrists=phi_wrists,
        alpha=alpha,
        lambda_wrist=lambda_wrist
    )

    sic_calc.initialize_b0(selected_episodes)

    n_ref = sic_calc.reference_anchor_count
    final_sic = sic_calc.get_current_sic()
    normalized_sic = final_sic / (n_ref * (1 + lambda_wrist))

    return {
        "fixed_universe_sic": final_sic,
        "normalized_sic": normalized_sic,
        "reference_anchor_count": n_ref,
        "dbar_global": sic_calc.dbar_global,
        "dbar_wrist": sic_calc.dbar_wrist,
        "sigma_stats": sic_calc.get_sigma_stats()
    }


def compute_pairwise_redundancy(
    episodes: List[int],
    episode_to_idx: Dict[int, int],
    K_global: np.ndarray,
    K_wrist: np.ndarray
) -> Dict:
    """
    计算 subset 内的 pairwise redundancy
    """
    indices = [episode_to_idx[ep] for ep in episodes if ep in episode_to_idx]
    if len(indices) < 2:
        return {"mean_global_redundancy": 0.0, "mean_wrist_redundancy": 0.0}

    idx_arr = np.array(indices)
    K_sub_g = K_global[np.ix_(idx_arr, idx_arr)]
    K_sub_w = K_wrist[np.ix_(idx_arr, idx_arr)]

    n = len(indices)
    mask = ~np.eye(n, dtype=bool)

    mean_global = float(np.mean(K_sub_g[mask]))
    mean_wrist = float(np.mean(K_sub_w[mask]))

    return {
        "mean_global_redundancy": mean_global,
        "mean_wrist_redundancy": mean_wrist
    }


def compute_mean_nearest_selected_distance(
    all_episode_indices: List[int],
    selected_episodes: List[int],
    episode_to_idx: Dict[int, int],
    K_global: np.ndarray,
    K_wrist: np.ndarray
) -> Dict:
    """
    计算每个未选 episode 到最近已选 episode 的平均 kernel 距离
    """
    selected_set = set(selected_episodes)
    unselected = [ep for ep in all_episode_indices if ep not in selected_set]

    if not unselected or not selected_episodes:
        return {"mean_nearest_global": 0.0, "mean_nearest_wrist": 0.0}

    selected_indices = np.array([episode_to_idx[ep] for ep in selected_episodes])
    unselected_indices = [episode_to_idx[ep] for ep in unselected]

    nearest_global = []
    nearest_wrist = []

    for ui in unselected_indices:
        sim_g = K_global[ui, selected_indices]
        sim_w = K_wrist[ui, selected_indices]
        nearest_global.append(float(np.max(sim_g)))
        nearest_wrist.append(float(np.max(sim_w)))

    return {
        "mean_nearest_global": float(np.mean(nearest_global)),
        "mean_nearest_wrist": float(np.mean(nearest_wrist))
    }


def main():
    parser = argparse.ArgumentParser(description="V1 vs V2 offline comparison")
    parser.add_argument("--embeddings-dir", type=str, required=True)
    parser.add_argument("--v1-subset", type=str, required=True, help="V1 subset.json path")
    parser.add_argument("--v2-subset", type=str, required=True, help="V2 subset.json path")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda-wrist", type=float, default=1.0)

    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings_dir)
    embeddings = load_embeddings(embeddings_dir)

    if not embeddings:
        print("错误: 没有找到嵌入文件")
        return

    all_episode_indices = sorted(embeddings.keys())
    phi_globals = np.array([embeddings[ep]["phi_global"] for ep in all_episode_indices])
    phi_wrists = np.array([embeddings[ep]["phi_wrist"] for ep in all_episode_indices])
    episode_to_idx = {ep: i for i, ep in enumerate(all_episode_indices)}

    from sic_v2 import compute_dbar_from_embeddings, build_kernel_matrices
    dbar_g, dbar_w, _ = compute_dbar_from_embeddings(phi_globals, phi_wrists)
    K_global, K_wrist = build_kernel_matrices(phi_globals, phi_wrists, dbar_g, dbar_w)

    with open(args.v1_subset) as f:
        v1_data = json.load(f)
    with open(args.v2_subset) as f:
        v2_data = json.load(f)

    v1_episodes = v1_data["selected_episode_indices"]
    v2_episodes = v2_data["selected_episode_indices"]

    v1_set = set(v1_episodes)
    v2_set = set(v2_episodes)

    overlap = v1_set & v2_set
    only_v1 = v1_set - v2_set
    only_v2 = v2_set - v1_set

    print(f"\n{'='*60}")
    print(f"V1 vs V2 Offline Comparison")
    print(f"{'='*60}")
    print(f"V1 episodes: {len(v1_episodes)}")
    print(f"V2 episodes: {len(v2_episodes)}")
    print(f"Overlap: {len(overlap)}")
    print(f"Only in V1: {len(only_v1)}")
    print(f"Only in V2: {len(only_v2)}")

    v1_sic_info = compute_fixed_universe_sic(
        v1_episodes, all_episode_indices, phi_globals, phi_wrists,
        args.alpha, args.lambda_wrist
    )
    v2_sic_info = compute_fixed_universe_sic(
        v2_episodes, all_episode_indices, phi_globals, phi_wrists,
        args.alpha, args.lambda_wrist
    )

    print(f"\n--- Fixed-Universe SIC (fair comparison) ---")
    print(f"V1 fixed-universe SIC: {v1_sic_info['fixed_universe_sic']:.4f}")
    print(f"V1 normalized SIC: {v1_sic_info['normalized_sic']:.4f}")
    print(f"V2 fixed-universe SIC: {v2_sic_info['fixed_universe_sic']:.4f}")
    print(f"V2 normalized SIC: {v2_sic_info['normalized_sic']:.4f}")

    v1_redundancy = compute_pairwise_redundancy(v1_episodes, episode_to_idx, K_global, K_wrist)
    v2_redundancy = compute_pairwise_redundancy(v2_episodes, episode_to_idx, K_global, K_wrist)

    print(f"\n--- Pairwise Redundancy ---")
    print(f"V1 global redundancy: {v1_redundancy['mean_global_redundancy']:.4f}")
    print(f"V1 wrist redundancy: {v1_redundancy['mean_wrist_redundancy']:.4f}")
    print(f"V2 global redundancy: {v2_redundancy['mean_global_redundancy']:.4f}")
    print(f"V2 wrist redundancy: {v2_redundancy['mean_wrist_redundancy']:.4f}")

    v1_coverage = compute_mean_nearest_selected_distance(
        all_episode_indices, v1_episodes, episode_to_idx, K_global, K_wrist
    )
    v2_coverage = compute_mean_nearest_selected_distance(
        all_episode_indices, v2_episodes, episode_to_idx, K_global, K_wrist
    )

    print(f"\n--- Coverage (mean nearest kernel similarity) ---")
    print(f"V1 global coverage: {v1_coverage['mean_nearest_global']:.4f}")
    print(f"V1 wrist coverage: {v1_coverage['mean_nearest_wrist']:.4f}")
    print(f"V2 global coverage: {v2_coverage['mean_nearest_global']:.4f}")
    print(f"V2 wrist coverage: {v2_coverage['mean_nearest_wrist']:.4f}")

    comparison = {
        "v1_episodes_count": len(v1_episodes),
        "v2_episodes_count": len(v2_episodes),
        "overlap_count": len(overlap),
        "overlap_ratio": len(overlap) / len(v1_episodes) if v1_episodes else 0,
        "only_v1_count": len(only_v1),
        "only_v2_count": len(only_v2),
        "v1_fixed_universe_sic": v1_sic_info["fixed_universe_sic"],
        "v1_normalized_sic": v1_sic_info["normalized_sic"],
        "v2_fixed_universe_sic": v2_sic_info["fixed_universe_sic"],
        "v2_normalized_sic": v2_sic_info["normalized_sic"],
        "v1_global_redundancy": v1_redundancy["mean_global_redundancy"],
        "v1_wrist_redundancy": v1_redundancy["mean_wrist_redundancy"],
        "v2_global_redundancy": v2_redundancy["mean_global_redundancy"],
        "v2_wrist_redundancy": v2_redundancy["mean_wrist_redundancy"],
        "v1_global_coverage": v1_coverage["mean_nearest_global"],
        "v1_wrist_coverage": v1_coverage["mean_nearest_wrist"],
        "v2_global_coverage": v2_coverage["mean_nearest_global"],
        "v2_wrist_coverage": v2_coverage["mean_nearest_wrist"],
        "reference_anchor_count": v1_sic_info["reference_anchor_count"],
        "dbar_global": v1_sic_info["dbar_global"],
        "dbar_wrist": v1_sic_info["dbar_wrist"]
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "v1_vs_v2_comparison.json"
    with open(output_file, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n保存比较结果到: {output_file}")


if __name__ == "__main__":
    main()