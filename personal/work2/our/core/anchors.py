"""
AnchorSystem: 管理锚点集、计算距离尺度参数 d̄_global/d̄_wrist
"""

import numpy as np
from typing import Dict, Tuple, Optional


class AnchorSystem:
    """
    锚点参考系管理系统
    
    属性:
        anchors: Dict[grid_coord, {"phi_global": np.ndarray, "phi_wrist": np.ndarray}]
        dbar_global: float - global视角距离尺度参数
        dbar_wrist: float - wrist视角距离尺度参数
    """
    
    def __init__(self, grid_coords: list = None):
        self.anchors: Dict[Tuple[int, int], Dict] = {}
        self.dbar_global: Optional[float] = None
        self.dbar_wrist: Optional[float] = None
        self.grid_coords = grid_coords or []
    
    def add_anchor(self, grid_coord: Tuple[int, int], 
                   phi_global: np.ndarray, 
                   phi_wrist: np.ndarray):
        """添加一个锚点到锚点集"""
        self.anchors[grid_coord] = {
            "phi_global": phi_global.copy(),
            "phi_wrist": phi_wrist.copy()
        }
    
    def has_anchor(self, grid_coord: Tuple[int, int]) -> bool:
        """检查锚点是否已存在"""
        return grid_coord in self.anchors
    
    def get_anchor(self, grid_coord: Tuple[int, int]) -> Optional[Dict]:
        """获取指定锚点的嵌入"""
        return self.anchors.get(grid_coord)
    
    def compute_dbar(self):
        """
        计算距离尺度参数 d̄_global 和 d̄_wrist
        
        δv(a) = min_{a'≠a} ||φv(a) - φv(a')||_2
        d̄v = (1/|A|) Σ_a δv(a)
        """
        if len(self.anchors) < 2:
            self.dbar_global = 1.0
            self.dbar_wrist = 1.0
            return
        
        anchor_list = list(self.anchors.values())
        phi_globals = np.array([a["phi_global"] for a in anchor_list])
        phi_wrists = np.array([a["phi_wrist"] for a in anchor_list])
        
        n = len(anchor_list)
        d_global_list = []
        d_wrist_list = []
        
        for i in range(n):
            dists_global = np.linalg.norm(phi_globals - phi_globals[i], axis=1)
            dists_wrist = np.linalg.norm(phi_wrists - phi_wrists[i], axis=1)
            
            dists_global[i] = np.inf
            dists_wrist[i] = np.inf
            
            d_global_list.append(dists_global.min())
            d_wrist_list.append(dists_wrist.min())
        
        self.dbar_global = float(np.mean(d_global_list))
        self.dbar_wrist = float(np.mean(d_wrist_list))
        
        if self.dbar_global < 1e-6:
            self.dbar_global = 1.0
        if self.dbar_wrist < 1e-6:
            self.dbar_wrist = 1.0
    
    def recompute_dbar(self):
        """重新计算距离尺度参数（当新增首次访问的锚点时调用）"""
        self.compute_dbar()
    
    def get_dbar_global(self) -> float:
        if self.dbar_global is None:
            self.compute_dbar()
        return self.dbar_global
    
    def get_dbar_wrist(self) -> float:
        if self.dbar_wrist is None:
            self.compute_dbar()
        return self.dbar_wrist