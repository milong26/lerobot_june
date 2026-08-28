"""
GreedyPlanner: 通用贪心主循环

score_fn 可插拔，供 V1-V4 复用同一份循环代码。
"""

from typing import Callable, Dict, Tuple, List, Optional
import numpy as np

from .candidate_pool import CandidatePool
from .anchors import AnchorSystem
from .sic import compute_sic_score, compute_marginal_gain


class GreedyPlanner:
    """
    通用贪心规划器
    
    参数:
        candidate_pool: CandidatePool实例
        anchor_system: AnchorSystem实例
        alpha: 次数软化系数
        lambda_wrist: wrist视角权重
        score_fn: 自定义打分函数 (coord, pool, anchor_system) -> float
                  如果为None，使用默认SIC边际增益
    """
    
    def __init__(
        self,
        candidate_pool: CandidatePool,
        anchor_system: AnchorSystem,
        alpha: float = 1.0,
        lambda_wrist: float = 1.0,
        score_fn: Optional[Callable] = None
    ):
        self.candidate_pool = candidate_pool
        self.anchor_system = anchor_system
        self.alpha = alpha
        self.lambda_wrist = lambda_wrist
        self.score_fn = score_fn
    
    def default_score_fn(self, coord: Tuple[int, int]) -> float:
        """
        默认打分函数：SIC边际增益
        
        Δ(c) = SIC(D∪c) - SIC(D)
        """
        return compute_marginal_gain(
            self.candidate_pool.get_candidate_set(),
            coord,
            self.anchor_system,
            self.alpha,
            self.lambda_wrist
        )
    
    def plan(self, budget: int, verbose: bool = True) -> Dict[Tuple[int, int], int]:
        """
        执行贪心规划
        
        参数:
            budget: 总采集预算
            verbose: 是否打印进度
        
        返回:
            Dict[grid_coord, repeat_count] - 每个格点的推荐采集次数
        """
        score_fn = self.score_fn if self.score_fn else self.default_score_fn
        
        while self.candidate_pool.get_total_budget_used() < budget:
            available = self.candidate_pool.get_available_candidates()
            
            if not available:
                if verbose:
                    print(f"  预算 {budget}: 所有候选已达t_max，提前终止")
                break
            
            best_coord = None
            best_score = -float("inf")
            
            for coord in available:
                score = score_fn(coord)
                if score > best_score:
                    best_score = score
                    best_coord = coord
            
            if best_coord is None:
                break
            
            self.candidate_pool.increment_rep(best_coord)
            
            if verbose and self.candidate_pool.get_total_budget_used() % 10 == 0:
                stats = self.candidate_pool.get_coverage_stats()
                current_sic = compute_sic_score(
                    self.candidate_pool.get_candidate_set(),
                    self.anchor_system,
                    self.alpha,
                    self.lambda_wrist
                )
                print(f"  预算={self.candidate_pool.get_total_budget_used()}: "
                      f"唯一配置={stats['n_unique']}, "
                      f"SIC={current_sic:.4f}, "
                      f"最新选择={best_coord}")
        
        return self.candidate_pool.get_candidate_set()