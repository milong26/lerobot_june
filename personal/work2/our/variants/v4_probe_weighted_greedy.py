"""
V4: 探针加权贪心

在V1基础上引入w_a，从personal/work2/tanzhen/results/weakness_scores.json读取。
如果该文件不存在，V4自动退化为V1并在日志里打印警告。
"""

import sys
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from our.core.anchors import AnchorSystem
from our.core.candidate_pool import CandidatePool
from our.core.greedy_planner import GreedyPlanner
from our.core.sic import compute_sic_score, compute_marginal_gain


def load_weakness_scores(weakness_file: str) -> Optional[Dict]:
    """
    加载弱点分数文件
    
    参数:
        weakness_file: weakness_scores.json路径
    
    返回:
        Dict[grid_coord_str, weakness_score] 或 None
    """
    path = Path(weakness_file)
    if not path.exists():
        warnings.warn(
            f"弱点分数文件不存在: {weakness_file}，V4将退化为V1"
        )
        return None
    
    with open(path) as f:
        return json.load(f)


def v4_score_fn(
    coord,
    candidate_pool: CandidatePool,
    anchor_system: AnchorSystem,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0,
    weakness_scores: Optional[Dict] = None
) -> float:
    """
    V4打分函数
    
    score(c) = w_a · Δ_SIC(c)
    如果weakness_scores为None，退化为V1
    """
    delta_sic = compute_marginal_gain(
        candidate_pool.get_candidate_set(),
        coord,
        anchor_system,
        alpha,
        lambda_wrist
    )
    
    if weakness_scores is None:
        return delta_sic
    
    coord_str = f"({coord[0]}, {coord[1]})"
    w_a = weakness_scores.get(coord_str, 1.0)
    
    return w_a * delta_sic


def run_v4(
    anchor_system: AnchorSystem,
    budget: int = 112,
    t_max: int = 10,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0,
    weakness_file: str = None,
    verbose: bool = True
) -> dict:
    """
    运行V4探针加权贪心
    
    参数:
        anchor_system: 已初始化的锚点系统
        budget: 总采集预算
        t_max: 单配置最大采集次数
        alpha: 次数软化系数
        lambda_wrist: wrist视角权重
        weakness_file: weakness_scores.json路径
        verbose: 是否打印进度
    
    返回:
        规划结果 Dict[grid_coord, repeat_count]
    """
    weakness_scores = None
    if weakness_file:
        weakness_scores = load_weakness_scores(weakness_file)
    
    if weakness_scores is None:
        if verbose:
            print("V4: 弱点分数不可用，退化为V1")
    
    candidate_pool = CandidatePool(grid_shape=(14, 8), t_max=t_max)
    
    for coord in anchor_system.anchors.keys():
        candidate_pool.mark_visited(coord)
        candidate_pool.rep_count[coord] = 1
    
    def score_fn(coord):
        return v4_score_fn(
            coord, candidate_pool, anchor_system,
            alpha, lambda_wrist, weakness_scores
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
        print(f"\nV4 结果:")
        print(f"  唯一配置数: {stats['n_unique']}")
        print(f"  重复采集数: {stats['n_repeats']}")
        print(f"  覆盖率: {stats['coverage_rate']:.1f}%")
        print(f"  SIC分数: {final_sic:.4f}")
        print(f"  使用弱点分数: {weakness_scores is not None}")
    
    return {
        "result": result,
        "stats": stats,
        "sic_score": final_sic
    }


if __name__ == "__main__":
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(description="V4 探针加权贪心")
    parser.add_argument("--embeddings-dir", type=str, required=True)
    parser.add_argument("--budget", type=int, default=112)
    parser.add_argument("--t-max", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda-wrist", type=float, default=1.0)
    parser.add_argument("--weakness-file", type=str, 
                       default="personal/work2/tanzhen/results/weakness_scores.json")
    parser.add_argument("--output", type=str, default="results/tables/offline_sim_v4.csv")
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
    
    result = run_v4(
        anchor_system=anchor_system,
        budget=args.budget,
        t_max=args.t_max,
        alpha=args.alpha,
        lambda_wrist=args.lambda_wrist,
        weakness_file=args.weakness_file
    )
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("budget,variant,sic_score,n_unique_configs,n_repeats\n")
        variant_name = "v4" if result.get("used_weakness", False) else "v4_degraded_to_v1"
        f.write(f"{args.budget},{variant_name},{result['sic_score']:.4f},"
                f"{result['stats']['n_unique']},{result['stats']['n_repeats']}\n")
    
    print(f"\n结果已保存到 {output_path}")