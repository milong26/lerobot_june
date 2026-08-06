"""SIC Framework - SIC score computation."""

import numpy as np


def laplacian_kernel(phi_a: np.ndarray, phi_pr: np.ndarray, d_bar: float) -> float:
    """K_v(a, p, r) = exp(-||phi_v(a) - phi_v(p,r)||_2 / d_bar_v)"""
    return np.exp(-np.linalg.norm(phi_a - phi_pr) / (d_bar + 1e-8))


def tau(t: int, alpha: float) -> float:
    """tau(t) = alpha * log((t+1)/t)"""
    return alpha * np.log((t + 1) / t)


def tau_cumsum(T_pr: int, alpha: float) -> float:
    """Sum_{t=1}^{T_pr} tau(t) = alpha * log(T_pr + 1)"""
    return alpha * np.log(T_pr + 1)


def compute_anchor_support(
    anchor_key: tuple,
    collection_plan: dict,
    anchors: dict,
    d_bar_global: float,
    d_bar_wrist: float,
    alpha: float
) -> dict:
    """Compute sigma_global(a, D) and sigma_wrist(a, D) for one anchor."""
    phi_a_g = anchors[anchor_key]['phi_global']
    phi_a_w = anchors[anchor_key]['phi_wrist']

    sigma_g = 0.0
    sigma_w = 0.0

    for (pos_id, rot_id), n_times in collection_plan.items():
        if (pos_id, rot_id) not in anchors:
            continue
        phi_pr_g = anchors[(pos_id, rot_id)]['phi_global']
        phi_pr_w = anchors[(pos_id, rot_id)]['phi_wrist']

        weight = tau_cumsum(n_times, alpha)
        sigma_g += weight * laplacian_kernel(phi_a_g, phi_pr_g, d_bar_global)
        sigma_w += weight * laplacian_kernel(phi_a_w, phi_pr_w, d_bar_wrist)

    return {'sigma_global': sigma_g, 'sigma_wrist': sigma_w}


def saturation(sigma: float) -> float:
    """sigma / (1 + sigma), maps [0, inf) -> [0, 1)"""
    return sigma / (1.0 + sigma)


def compute_sic(
    collection_plan: dict,
    anchor_ref: dict,
    alpha: float = 0.05,
    lambda_weight: float = 0.5
) -> dict:
    """
    SIC(D) = SIC_global(D) + lambda * SIC_wrist(D)
    SIC_v(D) = Sum_{a in A} saturation(sigma_v(a, D))
    """
    anchors = anchor_ref['anchors']
    sic_g, sic_w = 0.0, 0.0
    anchor_supports = {}

    for anchor_key in anchors:
        support = compute_anchor_support(
            anchor_key, collection_plan, anchors,
            anchor_ref['d_bar_global'], anchor_ref['d_bar_wrist'], alpha
        )
        sic_g += saturation(support['sigma_global'])
        sic_w += saturation(support['sigma_wrist'])
        anchor_supports[anchor_key] = support

    total_sic = sic_g + lambda_weight * sic_w
    return {
        'sic': total_sic,
        'sic_global': sic_g,
        'sic_wrist': sic_w,
        'anchor_supports': anchor_supports
    }


def compute_marginal_gain(
    current_plan: dict,
    candidate_config: tuple,
    anchor_ref: dict,
    alpha: float,
    lambda_weight: float
) -> float:
    """Compute SIC(D + 1 more of candidate) - SIC(D) efficiently."""
    new_plan = dict(current_plan)
    new_plan[candidate_config] = new_plan.get(candidate_config, 0) + 1

    current_sic = compute_sic(current_plan, anchor_ref, alpha, lambda_weight)['sic']
    new_sic = compute_sic(new_plan, anchor_ref, alpha, lambda_weight)['sic']

    return new_sic - current_sic