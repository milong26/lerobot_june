#!/usr/bin/env python
"""
Common analysis utilities for V1/V2 comparison and sanity checks

Centralizes functions that are shared across multiple scripts to avoid
code duplication and circular imports.
"""

import numpy as np
from typing import Dict, List

from sic_v2 import FixedAnchorSIC


def compute_fixed_universe_sic(
    selected_episodes: List[int],
    all_episode_indices: List[int],
    phi_globals: np.ndarray,
    phi_wrists: np.ndarray,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0
) -> Dict:
    """
    Compute SIC for a subset under fixed reference universe
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
    Compute pairwise redundancy within a subset
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


def compute_fixed_universe_sic_from_indices(
    selected_episodes: List[int],
    all_episode_indices: List[int],
    K_global: np.ndarray,
    K_wrist: np.ndarray,
    dbar_global: float,
    dbar_wrist: float,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0
) -> Dict:
    """
    Compute SIC for a subset from pre-computed kernel matrices and dbar.
    This allows reusing the same fixed universe across multiple bootstrap subsets.
    """
    from sic_v2 import tau

    episode_to_idx = {ep: i for i, ep in enumerate(all_episode_indices)}
    n_episodes = len(all_episode_indices)

    selected_indices = set()
    sigma_global = np.zeros(n_episodes)
    sigma_wrist = np.zeros(n_episodes)

    tau_1 = tau(1, alpha)

    for ep in selected_episodes:
        if ep in episode_to_idx:
            idx = episode_to_idx[ep]
            selected_indices.add(idx)
            sigma_global += tau_1 * K_global[:, idx]
            sigma_wrist += tau_1 * K_wrist[:, idx]

    def sat(x):
        return x / (1.0 + x)

    final_sic = float(np.sum(sat(sigma_global)) + lambda_wrist * np.sum(sat(sigma_wrist)))
    n_ref = n_episodes
    normalized_sic = final_sic / (n_ref * (1 + lambda_wrist))

    return {
        "fixed_universe_sic": final_sic,
        "normalized_sic": normalized_sic,
        "reference_anchor_count": n_ref,
        "dbar_global": dbar_global,
        "dbar_wrist": dbar_wrist,
    }


def compute_mean_nearest_selected_distance(
    all_episode_indices: List[int],
    selected_episodes: List[int],
    episode_to_idx: Dict[int, int],
    K_global: np.ndarray,
    K_wrist: np.ndarray
) -> Dict:
    """
    Compute mean nearest kernel distance from each unselected episode to selected episodes
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