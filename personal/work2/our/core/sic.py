"""
SIC (State Information Coverage) 计算模块

实现双视角空间接近度核函数、次数软化函数、支持度计算、饱和函数以及SIC总分计算。
"""

import numpy as np
from typing import Dict, Tuple, List, Optional


def tau(t: int, alpha: float = 1.0) -> float:
    """
    次数软化函数
    
    τ(t) = α · log((t+1)/t)
    
    参数:
        t: 同一配置下第t次采集 (t >= 1)
        alpha: 控制重复采集权重的系数
    
    返回:
        软化权重值
    """
    if t < 1:
        raise ValueError(f"t must be >= 1, got {t}")
    return alpha * np.log((t + 1) / t)


def compute_kernel_distance(phi_a: np.ndarray, phi_c: np.ndarray) -> float:
    """计算两个嵌入之间的L2距离"""
    return float(np.linalg.norm(phi_a - phi_c))


def laplacian_kernel(distance: float, dbar: float) -> float:
    """
    Laplacian核函数
    
    K = exp(-||φ(a) - φ(c)||_2 / d̄)
    
    参数:
        distance: 嵌入空间的L2距离
        dbar: 距离尺度参数
    
    返回:
        核函数值 [0, 1]
    """
    if dbar < 1e-6:
        return 1.0
    return float(np.exp(-distance / dbar))


def compute_sic_score(
    candidate_set: Dict[Tuple[int, int], int],
    anchor_system,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0
) -> float:
    """
    计算SIC总分
    
    SIC(D) = Σ_a σ_global(a,D)/(1+σ_global(a,D)) + λ·Σ_a σ_wrist(a,D)/(1+σ_wrist(a,D))
    
    参数:
        candidate_set: Dict[grid_coord, repeat_count] - 候选方案
        anchor_system: AnchorSystem实例
        alpha: 次数软化系数
        lambda_wrist: wrist视角权重
    
    返回:
        SIC总分
    """
    if not anchor_system.anchors:
        return 0.0
    
    dbar_global = anchor_system.get_dbar_global()
    dbar_wrist = anchor_system.get_dbar_wrist()
    
    sigma_global_all = np.zeros(len(anchor_system.anchors))
    sigma_wrist_all = np.zeros(len(anchor_system.anchors))
    
    anchor_list = list(anchor_system.anchors.items())
    
    for grid_coord, repeat_count in candidate_set.items():
        if grid_coord not in anchor_system.anchors:
            continue
        
        phi_c_global = anchor_system.anchors[grid_coord]["phi_global"]
        phi_c_wrist = anchor_system.anchors[grid_coord]["phi_wrist"]
        
        for t in range(1, repeat_count + 1):
            tau_val = tau(t, alpha)
            
            for idx, (anchor_coord, anchor_data) in enumerate(anchor_list):
                dist_global = compute_kernel_distance(
                    anchor_data["phi_global"], phi_c_global
                )
                dist_wrist = compute_kernel_distance(
                    anchor_data["phi_wrist"], phi_c_wrist
                )
                
                k_global = laplacian_kernel(dist_global, dbar_global)
                k_wrist = laplacian_kernel(dist_wrist, dbar_wrist)
                
                sigma_global_all[idx] += tau_val * k_global
                sigma_wrist_all[idx] += tau_val * k_wrist
    
    sat_global = sigma_global_all / (1 + sigma_global_all)
    sat_wrist = sigma_wrist_all / (1 + sigma_wrist_all)
    
    sic_score = float(np.sum(sat_global) + lambda_wrist * np.sum(sat_wrist))
    
    return sic_score


def compute_marginal_gain(
    candidate_set: Dict[Tuple[int, int], int],
    new_coord: Tuple[int, int],
    anchor_system,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0
) -> float:
    """
    计算边际增益 Δ(c) = SIC(D∪c) - SIC(D)
    
    参数:
        candidate_set: 当前候选方案
        new_coord: 新增的候选配置
        anchor_system: AnchorSystem实例
        alpha: 次数软化系数
        lambda_wrist: wrist视角权重
    
    返回:
        边际增益值
    """
    current_sic = compute_sic_score(candidate_set, anchor_system, alpha, lambda_wrist)
    
    new_set = candidate_set.copy()
    new_set[new_coord] = new_set.get(new_coord, 0) + 1
    
    new_sic = compute_sic_score(new_set, anchor_system, alpha, lambda_wrist)
    
    return new_sic - current_sic


def compute_support_for_anchor(
    anchor_coord: Tuple[int, int],
    candidate_set: Dict[Tuple[int, int], int],
    anchor_system,
    alpha: float = 1.0
) -> Tuple[float, float]:
    """
    计算单个锚点的支持度
    
    σ_global(a,D) = Σ_(p,r)∈D Σ_(t=1)^Tp,r τ(t)K_global(a,p,r)
    σ_wrist(a,D) = Σ_(p,r)∈D Σ_(t=1)^Tp,r τ(t)K_wrist(a,p,r)
    
    返回:
        (sigma_global, sigma_wrist)
    """
    if anchor_coord not in anchor_system.anchors:
        return 0.0, 0.0
    
    anchor_data = anchor_system.anchors[anchor_coord]
    dbar_global = anchor_system.get_dbar_global()
    dbar_wrist = anchor_system.get_dbar_wrist()
    
    sigma_global = 0.0
    sigma_wrist = 0.0
    
    for grid_coord, repeat_count in candidate_set.items():
        if grid_coord not in anchor_system.anchors:
            continue
        
        phi_c_global = anchor_system.anchors[grid_coord]["phi_global"]
        phi_c_wrist = anchor_system.anchors[grid_coord]["phi_wrist"]
        
        for t in range(1, repeat_count + 1):
            tau_val = tau(t, alpha)
            
            dist_global = compute_kernel_distance(
                anchor_data["phi_global"], phi_c_global
            )
            dist_wrist = compute_kernel_distance(
                anchor_data["phi_wrist"], phi_c_wrist
            )
            
            k_global = laplacian_kernel(dist_global, dbar_global)
            k_wrist = laplacian_kernel(dist_wrist, dbar_wrist)
            
            sigma_global += tau_val * k_global
            sigma_wrist += tau_val * k_wrist
    
    return sigma_global, sigma_wrist