#!/usr/bin/env python
"""
Step 3: SIC incremental iteration main loop.

SIC formula (from cankao_419code.py):
  SIC(S) = sum_a f( sum_{i in S} K(a, i) )
  f(x) = x / (1 + x)          (saturation function, marginal diminishing)
  K(a, i) = exp(-||a - i|| / d_bar)   (RBF kernel)

For per-cell scoring:
  cell_coverage(c) = f( sum_{i in B_t} K(emb(c), emb(i)) )
  cell_sic(c) = 1 - cell_coverage(c)   (high = poorly covered = "hole")

Adaptation for our setting:
  - Each cell's anchor = embedding of the episode physically closest to cell center
  - d_bar = mean pairwise distance among currently selected episodes B_t (dynamic, recomputed each round)
  - We use the episode embedding (PCA-32) directly as the feature vector
"""
import json
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

K_TOP_CELLS = 3
N_ADD_PER_ROUND = 10
MAX_PER_CELL = 5
CONVERGENCE_THRESHOLD = 0.01
BUDGETS = [50, 100, 150, 200, 300]


def sat(x):
    """Saturation function f(x) = x / (1 + x)"""
    return x / (1.0 + x)


def rbf_kernel(x, y, d_bar):
    """RBF kernel K(x, y) = exp(-||x - y|| / d_bar)"""
    return float(np.exp(-np.linalg.norm(x - y) / d_bar))


def compute_d_bar(b_t_embs):
    """Compute characteristic distance scale from currently selected episodes B_t."""
    embs = np.array(b_t_embs, dtype=np.float64)
    n = len(embs)
    if n < 2:
        return 1.0
    dists = cdist(embs, embs, metric="euclidean")
    triu_dists = dists[np.triu_indices(n, k=1)]
    d_bar = float(triu_dists.mean())
    if d_bar < 1e-10:
        d_bar = 1.0
    return d_bar


def get_cell_anchor_emb(cell_center_xy, episode_indices, obj_positions, episode_embs):
    """Get the embedding of the episode physically closest to the cell center."""
    phys_2d = obj_positions[:, :2]
    dists = np.sqrt((phys_2d[:, 0] - cell_center_xy[0]) ** 2 + (phys_2d[:, 1] - cell_center_xy[1]) ** 2)
    nearest_idx = int(np.argmin(dists))
    return episode_embs[nearest_idx], nearest_idx, int(episode_indices[nearest_idx])


def compute_cell_sic(cell_anchor_emb, b_t_embs, d_bar):
    """
    Compute SIC score for a single cell.
    
    cell_coverage = f( sum_{i in B_t} K(anchor, emb_i) )
    cell_sic = 1 - cell_coverage  (higher = more uncovered = bigger hole)
    """
    if len(b_t_embs) == 0:
        return 1.0

    kernel_sum = 0.0
    for emb_i in b_t_embs:
        kernel_sum += rbf_kernel(cell_anchor_emb, emb_i, d_bar)

    coverage = sat(kernel_sum)
    return 1.0 - coverage


def select_episodes_from_cell(cell_center_xy, episode_indices, obj_positions, b_t_set, n_select, rng):
    """Select episodes from a cell area, with perturbation to avoid duplicates."""
    phys_2d = obj_positions[:, :2]
    dists = np.sqrt((phys_2d[:, 0] - cell_center_xy[0]) ** 2 + (phys_2d[:, 1] - cell_center_xy[1]) ** 2)

    available_mask = ~np.isin(episode_indices, list(b_t_set))
    available_dists = dists.copy()
    available_dists[~available_mask] = np.inf

    selected = []
    for _ in range(n_select):
        best_idx = int(np.argmin(available_dists))
        if available_dists[best_idx] == np.inf:
            break
        selected.append(int(episode_indices[best_idx]))
        noise = rng.uniform(0.001, 0.01, size=2)
        available_dists[best_idx] = np.inf
        perturbed = phys_2d[best_idx] + noise
        available_dists += np.sqrt((phys_2d[:, 0] - perturbed[0]) ** 2 + (phys_2d[:, 1] - perturbed[1]) ** 2) * 0.1

    return selected


def main():
    print("=" * 60)
    print("Step 3: SIC Incremental Iteration")
    print("=" * 60)

    with open(RESULTS_DIR / "step0_all_data.pkl", "rb") as f:
        data = pickle.load(f)

    with open(RESULTS_DIR / "step1_grid_info.json", "r") as f:
        grid_info = json.load(f)

    with open(RESULTS_DIR / "step2_b0_episodes.json", "r") as f:
        b0_info = json.load(f)

    episode_indices = data["episode_indices"]
    obj_positions = data["obj_positions"]
    episode_embs = data["episode_embs"]
    cell_centers = grid_info["cell_centers"]

    b_t_set = set(b0_info["b0_episodes"])
    b_t_list = sorted(b_t_set)
    b_t_embs = [episode_embs[i] for i in range(len(episode_indices)) if episode_indices[i] in b_t_set]
    print(f"B0 size: {len(b_t_list)}")

    rng = np.random.RandomState(42)

    cell_sic_history = []
    overall_sic_history = []
    d_bar_history = []
    selection_log = []
    budget_subsets = {}

    round_num = 0
    prev_overall_sic = None
    prev_prev_overall_sic = None

    # Compute initial d_bar from B0
    d_bar = compute_d_bar(b_t_embs)
    d_bar_history.append(d_bar)
    print(f"Round 0: d_bar = {d_bar:.6f}")

    cell_sic_values = []
    for cell in cell_centers:
        cx, cy = cell["x_center"], cell["y_center"]
        anchor_emb, _, _ = get_cell_anchor_emb((cx, cy), episode_indices, obj_positions, episode_embs)
        sic = compute_cell_sic(anchor_emb, b_t_embs, d_bar)
        cell_sic_values.append(sic)
    overall_sic = float(np.mean(cell_sic_values))
    initial_overall_sic = overall_sic
    overall_sic_history.append(overall_sic)
    cell_sic_history.append(cell_sic_values.copy())

    print(f"Round 0 (B0): overall_SIC = {overall_sic:.6f}")

    cell_add_count = {cell["cell_id"]: 0 for cell in cell_centers}

    while True:
        round_num += 1

        # Recompute d_bar from current B_t
        d_bar = compute_d_bar(b_t_embs)
        d_bar_history.append(d_bar)

        cell_sic_values = []
        cell_anchor_eps = []
        for cell in cell_centers:
            cx, cy = cell["x_center"], cell["y_center"]
            anchor_emb, _, nearest_ep = get_cell_anchor_emb(
                (cx, cy), episode_indices, obj_positions, episode_embs
            )
            sic = compute_cell_sic(anchor_emb, b_t_embs, d_bar)
            cell_sic_values.append(sic)
            cell_anchor_eps.append(nearest_ep)

        overall_sic = float(np.mean(cell_sic_values))
        overall_sic_history.append(overall_sic)
        cell_sic_history.append(cell_sic_values.copy())

        print(f"Round {round_num}: d_bar = {d_bar:.6f}, overall_SIC = {overall_sic:.6f}")

        if prev_overall_sic is not None and prev_prev_overall_sic is not None:
            delta1 = prev_overall_sic - overall_sic
            delta2 = prev_prev_overall_sic - prev_overall_sic
            threshold = CONVERGENCE_THRESHOLD * initial_overall_sic
            if delta1 < threshold and delta2 < threshold:
                print(f"\nConverged at round {round_num} (delta1={delta1:.6f}, delta2={delta2:.6f}, threshold={threshold:.6f})")
                break

        prev_prev_overall_sic = prev_overall_sic
        prev_overall_sic = overall_sic

        sic_array = np.array(cell_sic_values)
        sorted_cell_indices = np.argsort(-sic_array)

        round_selections = []
        remaining_add = N_ADD_PER_ROUND

        for rank_idx in sorted_cell_indices:
            if remaining_add <= 0:
                break

            cell = cell_centers[rank_idx]
            cell_id = cell["cell_id"]

            if cell_add_count[cell_id] >= MAX_PER_CELL:
                continue

            n_to_add = min(remaining_add, MAX_PER_CELL - cell_add_count[cell_id])

            cx, cy = cell["x_center"], cell["y_center"]
            new_eps = select_episodes_from_cell(
                (cx, cy), episode_indices, obj_positions, b_t_set, n_to_add, rng
            )

            for ep in new_eps:
                b_t_set.add(ep)
                ep_idx_in_data = int(np.where(np.array(episode_indices) == ep)[0][0])
                b_t_embs.append(episode_embs[ep_idx_in_data])
                round_selections.append(ep)
                cell_add_count[cell_id] += 1

            remaining_add -= len(new_eps)
            selection_log.append({
                "round": round_num,
                "cell_id": cell_id,
                "cell_sic": float(sic_array[rank_idx]),
                "added_episodes": new_eps,
            })

        b_t_list = sorted(b_t_set)
        print(f"  Added {len(round_selections)} episodes, total B_t = {len(b_t_list)}")

        for budget in BUDGETS:
            if budget not in budget_subsets and len(b_t_list) >= budget:
                budget_subsets[budget] = b_t_list[:budget]
                print(f"  Budget N={budget} reached, saved subset")

        if len(b_t_list) >= max(BUDGETS):
            print(f"\nReached maximum budget {max(BUDGETS)}, stopping")
            break

    final_cell_sic = cell_sic_history[-1]

    sic_curve_path = RESULTS_DIR / "sic_curves.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(len(overall_sic_history)), overall_sic_history, "b-o", linewidth=2, markersize=6)
    ax.set_xlabel("Round")
    ax.set_ylabel("Overall SIC (1 - coverage)")
    ax.set_title("Overall SIC vs Iteration Round")
    ax.grid(True, alpha=0.3)
    plt.savefig(sic_curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved SIC curve to {sic_curve_path}")

    n_grid = grid_info["n_grid_x"]
    sic_matrix = np.array(final_cell_sic).reshape(n_grid, -1)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(sic_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("Grid J (Y)")
    ax.set_ylabel("Grid I (X)")
    ax.set_title("Final Cell SIC Heatmap (1 - coverage)")
    plt.colorbar(im, ax=ax, label="SIC Value")
    for i in range(n_grid):
        for j in range(n_grid):
            ax.text(j, i, f"{sic_matrix[i, j]:.3f}", ha="center", va="center", fontsize=8, color="black")
    heatmap_path = RESULTS_DIR / "cell_sic_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved cell SIC heatmap to {heatmap_path}")

    selection_log_path = RESULTS_DIR / "selection_log.json"
    log_data = {
        "d_bar_history": d_bar_history,
        "initial_overall_sic": initial_overall_sic,
        "final_overall_sic": overall_sic_history[-1],
        "convergence_round": round_num,
        "overall_sic_history": overall_sic_history,
        "cell_sic_history": cell_sic_history,
        "selection_log": selection_log,
        "final_b_t": b_t_list,
        "cell_add_count": cell_add_count,
    }
    with open(selection_log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"Saved selection log to {selection_log_path}")

    budget_path = RESULTS_DIR / "budget_subsets.pkl"
    with open(budget_path, "wb") as f:
        pickle.dump(budget_subsets, f)
    print(f"Saved budget subsets to {budget_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()