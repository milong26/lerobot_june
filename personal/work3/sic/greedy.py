"""SIC Framework - Forward greedy planning algorithm."""

import numpy as np
from tqdm import tqdm
from .score import compute_sic, compute_marginal_gain


def greedy_plan(
    anchor_ref: dict,
    budget_B: int,
    t_max: int = 4,
    alpha: float = 0.05,
    lambda_weight: float = 0.5
) -> dict:
    """
    Forward greedy planning algorithm.
    Starts from B0 (all configs with n_times=1), adds one collection
    at a time to the config with highest SIC marginal gain.
    """
    all_configs = list(anchor_ref['anchors'].keys())

    current_plan = {cfg: 1 for cfg in all_configs}
    current_sic = compute_sic(current_plan, anchor_ref, alpha, lambda_weight)['sic']

    sic_history = [current_sic]
    gain_history = []
    selected_configs = []
    stopping_step = None

    n_steps = budget_B - len(all_configs)

    for step in tqdm(range(n_steps), desc="Greedy planning"):
        candidates = [cfg for cfg in all_configs if current_plan[cfg] < t_max]

        if not candidates:
            print(f"All configs reached t_max={t_max} at step {step}")
            break

        gains = {cfg: compute_marginal_gain(
            current_plan, cfg, anchor_ref, alpha, lambda_weight
        ) for cfg in candidates}

        best_cfg = max(gains, key=gains.get)
        best_gain = gains[best_cfg]

        current_plan[best_cfg] += 1
        new_sic = compute_sic(current_plan, anchor_ref, alpha, lambda_weight)['sic']

        sic_history.append(new_sic)
        gain_history.append(best_gain)
        selected_configs.append(best_cfg)

        if best_gain < 0.005 * new_sic and stopping_step is None:
            stopping_step = step
            print(f"Stopping criterion met at step {step}: "
                  f"marginal gain {best_gain:.6f} < 0.5% of SIC {new_sic:.4f}")

    return {
        'final_plan': current_plan,
        'sic_history': sic_history,
        'gain_history': gain_history,
        'selected_configs': selected_configs,
        'stopping_step': stopping_step if stopping_step else n_steps
    }