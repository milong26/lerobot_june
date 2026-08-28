"""
V3: 显式权衡贪心

在V1基础上加一个超参数γ，
score(c) = Δ_SIC(c) + γ·𝟙[c 是首次访问]
把"探索奖励"显式暴露成一个可扫描的超参数。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from our.core.anchors import AnchorSystem
from our.core.candidate_pool import CandidatePool
from our.core.greedy_planner import GreedyPlanner
from our.core.sic import compute_sic_score, compute_marginal_gain


def v3_score_fn(
    coord,
    candidate_pool: CandidatePool,
    anchor_system: AnchorSystem,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0,
    gamma: float = 0.0
) -> float:
    """
    V3打分函数
    
    score(c) = Δ_SIC(c) + γ·𝟙[c 是首次访问]
    """
    delta_sic = compute_marginal_gain(
        candidate_pool.get_candidate_set(),
        coord,
        anchor_system,
        alpha,
        lambda_wrist
    )
    
    is_first_visit = coord not in candidate_pool.visited
    
    return delta_sic + gamma * (1.0 if is_first_visit else 0.0)


def run_v3(
    anchor_system: AnchorSystem,
    budget: int = 112,
    t_max: int = 10,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0,
    gamma: float = 0.0,
    verbose: bool = True
) -> dict:
    """
    运行V3显式权衡贪心
    
    参数:
        anchor_system: 已初始化的锚点系统
        budget: 总采集预算
        t_max: 单配置最大采集次数
        alpha: 次数软化系数
        lambda_wrist: wrist视角权重
        gamma: 探索奖励超参数
        verbose: 是否打印进度
    
    返回:
        规划结果 Dict[grid_coord, repeat_count]
    """
    candidate_pool = CandidatePool(grid_shape=(14, 8), t_max=t_max)
    
    for coord in anchor_system.anchors.keys():
        candidate_pool.mark_visited(coord)
        candidate_pool.rep_count[coord] = 1
    
    def score_fn(coord):
        return v3_score_fn(
            coord, candidate_pool, anchor_system,
            alpha, lambda_wrist, gamma
        )
    
    planner = GreedyPlanner(
        candidate_pool=candidate_pool,
        anchor_system=anchor_system,
        alpha=alpha,
        lambda_wrist=lambda_wrist,
        score_fn=score_fn
    )
    
    result = planner.plan(budget, verbose=verbose)
    
    stats = candidate_pool.get_coverage_stats()
    final_sic = compute_sic_score(result, anchor_system, alpha, lambda_wrist)
    
    if verbose:
        print(f"\nV3 (gamma={gamma}) 结果:")
        print(f"  唯一配置数: {stats['n_unique']}")
        print(f"  重复采集数: {stats['n_repeats']}")
        print(f"  覆盖率: {stats['coverage_rate']:.1f}%")
        print(f"  SIC分数: {final_sic:.4f}")
    
    return {
        "result": result,
        "stats": stats,
        "sic_score": final_sic
    }


if __name__ == "__main__":
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description="V3 显式权衡贪心")
    parser.add_argument("--embeddings-dir", type=str, required=True)
    parser.add_argument("--budget", type=int, default=112)
    parser.add_argument("--t-max", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda-wrist", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.0,
                       help="探索奖励超参数")
    parser.add_argument("--output", type=str, default="results/tables/offline_sim_v3.csv")
    args = parser.parse_args()
    
    embeddings_dir = Path(args.embeddings_dir)
    
    embeddings = {}
    for f in embeddings_dir.glob("*.npy"):
        coord_str = f.stem
        parts = coord_str.replace("(", "").replace(")", "").split(",")
        coord = (int(parts[0].strip()), int(parts[1].strip()))
        data = np.load(f, allow_pickle=True).item()
        embeddings[coord] = data
    
    anchor_system = AnchorSystem()
    for coord, data in embeddings.items():
        anchor_system.add_anchor(coord, data["phi_global"], data["phi_wrist"])
    anchor_system.compute_dbar()
    
    result = run_v3(
        anchor_system=anchor_system,
        budget=args.budget,
        t_max=args.t_max,
        alpha=args.alpha,
        lambda_wrist=args.lambda_wrist,
        gamma=args.gamma
    )
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("budget,variant,sic_score,n_unique_configs,n_repeats\n")
        f.write(f"{args.budget},v3_gamma{args.gamma},{result['sic_score']:.4f},"
                f"{result['stats']['n_unique']},{result['stats']['n_repeats']}\n")
    
    print(f"\n结果已保存到 {output_path}")