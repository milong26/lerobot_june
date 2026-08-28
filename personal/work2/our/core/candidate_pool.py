"""
CandidatePool: 112格点候选池状态机

管理 visited/unvisited/rep_count 状态
"""

from typing import Dict, Tuple, List, Set


class CandidatePool:
    """
    候选池状态机
    
    属性:
        grid_shape: (14, 8) 网格形状
        rep_count: Dict[grid_coord, int] 每个格点的重复次数
        visited: Set[grid_coord] 已访问的格点集合
        t_max: 单配置最大采集次数限制
    """
    
    def __init__(self, grid_shape: Tuple[int, int] = (14, 8), t_max: int = 10):
        self.grid_shape = grid_shape
        self.rep_count: Dict[Tuple[int, int], int] = {}
        self.visited: Set[Tuple[int, int]] = set()
        self.t_max = t_max
        self._init_all_coords()
    
    def _init_all_coords(self):
        """初始化所有网格坐标"""
        self.all_coords: List[Tuple[int, int]] = [
            (i, j) for i in range(self.grid_shape[0]) for j in range(self.grid_shape[1])
        ]
    
    def get_all_coords(self) -> List[Tuple[int, int]]:
        """获取所有候选网格坐标"""
        return self.all_coords.copy()
    
    def get_unvisited(self) -> List[Tuple[int, int]]:
        """获取未访问的网格坐标"""
        return [c for c in self.all_coords if c not in self.visited]
    
    def get_visited(self) -> List[Tuple[int, int]]:
        """获取已访问的网格坐标"""
        return list(self.visited)
    
    def get_rep_count(self, coord: Tuple[int, int]) -> int:
        """获取指定格点的重复次数"""
        return self.rep_count.get(coord, 0)
    
    def mark_visited(self, coord: Tuple[int, int]):
        """标记格点为已访问"""
        self.visited.add(coord)
        if coord not in self.rep_count:
            self.rep_count[coord] = 0
    
    def increment_rep(self, coord: Tuple[int, int]) -> int:
        """
        增加格点的重复次数
        
        返回:
            增加后的重复次数
        """
        if coord not in self.rep_count:
            self.rep_count[coord] = 0
            self.visited.add(coord)
        
        self.rep_count[coord] += 1
        return self.rep_count[coord]
    
    def can_add(self, coord: Tuple[int, int]) -> bool:
        """检查是否还能继续添加该格点（未超过t_max）"""
        return self.rep_count.get(coord, 0) < self.t_max
    
    def get_available_candidates(self) -> List[Tuple[int, int]]:
        """获取所有未达到t_max的候选格点"""
        return [c for c in self.all_coords if self.can_add(c)]
    
    def get_candidate_set(self) -> Dict[Tuple[int, int], int]:
        """获取当前候选方案 {coord: rep_count}"""
        return {c: self.rep_count[c] for c in self.visited if self.rep_count[c] > 0}
    
    def get_total_budget_used(self) -> int:
        """获取已使用的总预算"""
        return sum(self.rep_count.values())
    
    def get_n_unique_configs(self) -> int:
        """获取唯一配置数（访问过的格点数）"""
        return len(self.visited)
    
    def get_n_repeats(self) -> int:
        """获取重复采集数（总预算 - 唯一配置数）"""
        return self.get_total_budget_used() - self.get_n_unique_configs()
    
    def is_full(self) -> bool:
        """检查是否所有格点都达到t_max"""
        return all(self.rep_count.get(c, 0) >= self.t_max for c in self.all_coords)
    
    def get_coverage_stats(self) -> Dict:
        """获取覆盖统计信息"""
        n_total = len(self.all_coords)
        n_visited = len(self.visited)
        n_unvisited = n_total - n_visited
        coverage_rate = n_visited / n_total * 100 if n_total > 0 else 0
        
        return {
            "n_total": n_total,
            "n_visited": n_visited,
            "n_unvisited": n_unvisited,
            "coverage_rate": coverage_rate,
            "total_budget": self.get_total_budget_used(),
            "n_unique": self.get_n_unique_configs(),
            "n_repeats": self.get_n_repeats()
        }