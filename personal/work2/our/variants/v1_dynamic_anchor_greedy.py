"""
V1: 动态锚点贪心

候选池 = 全部未达 T_max 的格点（含从未访问过的）。
每步计算 Δ(c) = SIC(D∪c) − SIC(D)
首次访问的候选因为会新增一整个饱和项，天然获得较高边际增益；
随着覆盖变密，这个优势自动衰减，转而与"重复候选"公平竞争。
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from our.core.anchors import AnchorSystem
from our.core.candidate_pool import CandidatePool
from our.core.greedy_planner import GreedyPlanner
from our.core.sic import compute_sic_score


def run_v1(
    anchor_system: AnchorSystem,
    budget: int = 112,
    t_max: int = 10,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0,
    verbose: bool = True
) -> dict:
    """
    运行V1动态锚点贪心
    
    参数:
        anchor_system: 已初始化的锚点系统（包含B0的嵌入）
        budget: 总采集预算
        t_max: 单配置最大采集次数
        alpha: 次数软化系数
        lambda_wrist: wrist视角权重
        verbose: 是否打印进度
    
    返回:
        规划结果 Dict[grid_coord, repeat_count]
    """
    candidate_pool = CandidatePool(grid_shape=(14, 8), t_max=t_max)
    
    for coord in anchor_system.anchors.keys():
        candidate_pool.mark_visited(coord)
        candidate_pool.rep_count[coord] = 1
    
    planner = GreedyPlanner(
        candidate_pool=candidate_pool,
        anchor_system=anchor_system,
        alpha=alpha,
        lambda_wrist=lambda_wrist
    )
    
    result = planner.plan(budget, verbose=verbose)
    
    stats = candidate_pool.get_coverage_stats()
    final_sic = compute_sic_score(result, anchor_system, alpha, lambda_wrist)
    
    if verbose:
        print(f"\nV1 结果:")
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
    import json
    import numpy as np
    
    parser = argparse.ArgumentParser(description="V1 动态锚点贪心")
    parser.add_argument("--embeddings-dir", type=str, required=True,
                       help="嵌入缓存目录路径")
    parser.add_argument("--budget", type=int, default=112)
    parser.add_argument("--t-max", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda-wrist", type=float, default=1.0)
    parser.add_argument("--output", type=str, default="results/tables/offline_sim_v1.csv")
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
    
    result = run_v1(
        anchor_system=anchor_system,
        budget=args.budget,
        t_max=args.t_max,
        alpha=args.alpha,
        lambda_wrist=args.lambda_wrist
    )
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("budget,variant,sic_score,n_unique_configs,n_repeats\n")
        f.write(f"{args.budget},v1,{result['sic_score']:.4f},"
                f"{result['stats']['n_unique']},{result['stats']['n_repeats']}\n")
    
    print(f"\n结果已保存到 {output_path}")