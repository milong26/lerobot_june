"""
离线模拟主入口

在离线池上模拟 V1-V3，扫描多个预算点。
输出 results/tables/offline_sim_{variant}.csv
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from our.core.anchors import AnchorSystem
from our.core.candidate_pool import CandidatePool
from our.core.sic import compute_sic_score
from our.variants.v1_dynamic_anchor_greedy import run_v1
from our.variants.v2_two_phase_explore_exploit import run_v2
from our.variants.v3_explicit_tradeoff_greedy import run_v3


def load_embeddings(embeddings_dir: Path) -> Dict[Tuple[int, int], Dict]:
    """
    加载缓存的嵌入
    
    参数:
        embeddings_dir: 嵌入缓存目录
    
    返回:
        Dict[grid_coord, {"phi_global": ..., "phi_wrist": ...}]
    """
    embeddings = {}
    
    for f in embeddings_dir.glob("*.npy"):
        coord_str = f.stem
        try:
            parts = coord_str.replace("(", "").replace(")", "").split(",")
            coord = (int(parts[0].strip()), int(parts[1].strip()))
            data = np.load(f, allow_pickle=True).item()
            embeddings[coord] = data
        except (ValueError, IndexError) as e:
            print(f"  跳过无效文件名: {f.name} ({e})")
    
    print(f"加载了 {len(embeddings)} 个嵌入")
    return embeddings


def build_anchor_system(embeddings: Dict) -> AnchorSystem:
    """
    构建锚点系统
    
    参数:
        embeddings: 嵌入字典
    
    返回:
        初始化好的AnchorSystem
    """
    anchor_system = AnchorSystem()
    
    for coord, data in embeddings.items():
        anchor_system.add_anchor(coord, data["phi_global"], data["phi_wrist"])
    
    anchor_system.compute_dbar()
    
    print(f"锚点系统: {len(anchor_system.anchors)} 个锚点")
    print(f"  dbar_global: {anchor_system.dbar_global:.4f}")
    print(f"  dbar_wrist: {anchor_system.dbar_wrist:.4f}")
    
    return anchor_system


def run_offline_simulation(
    embeddings_dir: Path,
    output_dir: Path,
    budgets: List[int] = None,
    t_max: int = 10,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0,
    b_cov: int = 50,
    gammas: List[float] = None
):
    """
    运行离线模拟
    
    参数:
        embeddings_dir: 嵌入缓存目录
        output_dir: 输出目录
        budgets: 预算点列表
        t_max: 单配置最大采集次数
        alpha: 次数软化系数
        lambda_wrist: wrist视角权重
        b_cov: V2阶段一覆盖数
        gammas: V3的gamma扫描值列表
    """
    if budgets is None:
        budgets = [50, 100, 112, 150, 200]
    
    if gammas is None:
        gammas = [0.0, 0.1, 0.5, 1.0, 2.0]
    
    embeddings = load_embeddings(embeddings_dir)
    
    if not embeddings:
        print("错误: 没有找到嵌入文件")
        return
    
    print("\n构建锚点系统...")
    anchor_system = build_anchor_system(embeddings)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    print("\n" + "=" * 60)
    print("V1 动态锚点贪心")
    print("=" * 60)
    
    v1_results = []
    best_v1_result = None
    best_v1_sic = -float('inf')
    
    for budget in budgets:
        print(f"\n--- V1: 预算 {budget} ---")
        result = run_v1(
            anchor_system=anchor_system,
            budget=budget,
            t_max=t_max,
            alpha=alpha,
            lambda_wrist=lambda_wrist,
            verbose=True
        )
        v1_results.append({
            "budget": budget,
            "sic_score": result["sic_score"],
            "n_unique": result["stats"]["n_unique"],
            "n_repeats": result["stats"]["n_repeats"]
        })
        print(f"  V1 预算 {budget} 完成: SIC={result['sic_score']:.4f}, 唯一配置={result['stats']['n_unique']}, 重复={result['stats']['n_repeats']}")
        
        if result["sic_score"] > best_v1_sic:
            best_v1_sic = result["sic_score"]
            best_v1_result = result["result"]
    
    v1_output = output_dir / "offline_sim_v1.csv"
    with open(v1_output, "w") as f:
        f.write("budget,variant,sic_score,n_unique_configs,n_repeats\n")
        for r in v1_results:
            f.write(f"{r['budget']},v1,{r['sic_score']:.4f},{r['n_unique']},{r['n_repeats']}\n")
    print(f"\nV1结果已保存到: {v1_output}")
    print(f"V1 最佳 SIC 分数: {best_v1_sic:.4f}")
    
    # Save best V1 planning result as JSON
    if best_v1_result:
        planning_result_json = output_dir / "planning_result.json"
        planning_result_serializable = {
            f"({k[0]},{k[1]})": v
            for k, v in best_v1_result.items()
        }
        with open(planning_result_json, "w") as f:
            json.dump(planning_result_serializable, f, indent=2)
        print(f"最佳V1规划结果已保存到: {planning_result_json}")
    
    print("\n" + "=" * 60)
    print("V2 两阶段探索-利用")
    print("=" * 60)
    
    v2_results = []
    for budget in budgets:
        print(f"\n--- V2: 预算 {budget} ---")
        result = run_v2(
            anchor_system=anchor_system,
            budget=budget,
            b_cov=b_cov,
            t_max=t_max,
            alpha=alpha,
            lambda_wrist=lambda_wrist,
            verbose=True
        )
        v2_results.append({
            "budget": budget,
            "sic_score": result["sic_score"],
            "n_unique": result["stats"]["n_unique"],
            "n_repeats": result["stats"]["n_repeats"]
        })
        print(f"  V2 预算 {budget} 完成: SIC={result['sic_score']:.4f}, 唯一配置={result['stats']['n_unique']}, 重复={result['stats']['n_repeats']}")
    
    v2_output = output_dir / "offline_sim_v2.csv"
    with open(v2_output, "w") as f:
        f.write("budget,variant,sic_score,n_unique_configs,n_repeats\n")
        for r in v2_results:
            f.write(f"{r['budget']},v2,{r['sic_score']:.4f},{r['n_unique']},{r['n_repeats']}\n")
    print(f"\nV2结果已保存到: {v2_output}")
    
    print("\n" + "=" * 60)
    print("V3 显式权衡贪心 (gamma扫描)")
    print("=" * 60)
    
    v3_results = []
    for gamma in gammas:
        print(f"\nGamma: {gamma}")
        for budget in budgets:
            result = run_v3(
                anchor_system=anchor_system,
                budget=budget,
                t_max=t_max,
                alpha=alpha,
                lambda_wrist=lambda_wrist,
                gamma=gamma,
                verbose=False
            )
            v3_results.append({
                "budget": budget,
                "gamma": gamma,
                "sic_score": result["sic_score"],
                "n_unique": result["stats"]["n_unique"],
                "n_repeats": result["stats"]["n_repeats"]
            })
    
    v3_output = output_dir / "offline_sim_v3.csv"
    with open(v3_output, "w") as f:
        f.write("budget,gamma,variant,sic_score,n_unique_configs,n_repeats\n")
        for r in v3_results:
            f.write(f"{r['budget']},{r['gamma']},v3_gamma{r['gamma']},{r['sic_score']:.4f},{r['n_unique']},{r['n_repeats']}\n")
    print(f"\nV3结果已保存到: {v3_output}")
    
    print("\n" + "=" * 60)
    print("离线模拟完成")
    print("=" * 60)
    
    print("\n各变体最佳SIC分数对比:")
    print(f"{'预算':<8} | {'V1':>10} | {'V2':>10} | {'V3_best':>10}")
    print("-" * 50)
    
    for budget in budgets:
        v1_best = max([r for r in v1_results if r["budget"] == budget], key=lambda x: x["sic_score"])
        v2_best = max([r for r in v2_results if r["budget"] == budget], key=lambda x: x["sic_score"])
        v3_best = max([r for r in v3_results if r["budget"] == budget], key=lambda x: x["sic_score"])
        
        print(f"{budget:<8} | {v1_best['sic_score']:>10.4f} | {v2_best['sic_score']:>10.4f} | {v3_best['sic_score']:>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离线模拟V1-V3")
    parser.add_argument("--embeddings-dir", type=str, required=True,
                       help="嵌入缓存目录")
    parser.add_argument("--output-dir", type=str,
                       default="personal/work2/our/results/tables",
                       help="输出目录")
    parser.add_argument("--budgets", type=int, nargs="+",
                       default=[50, 100, 112, 150, 200],
                       help="预算点列表")
    parser.add_argument("--t-max", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--lambda-wrist", type=float, default=1.0)
    parser.add_argument("--b-cov", type=int, default=50)
    parser.add_argument("--gammas", type=float, nargs="+",
                       default=[0.0, 0.1, 0.5, 1.0, 2.0])
    args = parser.parse_args()
    
    run_offline_simulation(
        embeddings_dir=Path(args.embeddings_dir),
        output_dir=Path(args.output_dir),
        budgets=args.budgets,
        t_max=args.t_max,
        alpha=args.alpha,
        lambda_wrist=args.lambda_wrist,
        b_cov=args.b_cov,
        gammas=args.gammas
    )