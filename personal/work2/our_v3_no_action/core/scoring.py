"""
Scoring Module for V3 No-Action

Computes spatial need and visual disagreement for each cell.
All weights are loaded from config.py, never hard-coded.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from our_v3_no_action.core.adaptive_grid import (
    AdaptiveCell, get_leaf_cells, get_neighbor_cells,
    compute_cell_area, cell_center,
)
from our_v3_no_action.config import SPATIAL_WEIGHT, VISUAL_WEIGHT


def compute_spatial_need(
    cell: AdaptiveCell,
    acquired_positions: Dict[int, Tuple[float, float, float]],
) -> float:
    """
    Measure how sparsely sampled this cell currently is.

    Combines:
    - Cell area (larger cells have higher need)
    - Number of samples in this cell (fewer samples = higher need)
    - Minimum pairwise distance among samples in cell (larger gaps = higher need)

    Returns a non-negative score; higher means more spatial need.
    """
    area = compute_cell_area(cell)
    n_samples = cell.n_samples

    if n_samples == 0:
        # No samples yet: maximum spatial need proportional to area
        return area

    # Compute minimum pairwise distance among samples in this cell
    positions = cell.sample_positions
    if len(positions) == 1:
        # Single sample: need based on distance to cell boundaries
        cx, cy, _ = cell_center(cell)
        sx, sy = positions[0][0], positions[0][1]
        dist_to_center = np.sqrt((sx - cx)**2 + (sy - cy)**2)
        max_dist = np.sqrt(area) / 2.0
        return area * (dist_to_center / max_dist) if max_dist > 0 else area

    # Multiple samples: compute min pairwise distance
    min_pairwise_dist = float("inf")
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = np.sqrt(
                (positions[i][0] - positions[j][0])**2 +
                (positions[i][1] - positions[j][1])**2
            )
            min_pairwise_dist = min(min_pairwise_dist, d)

    # Higher area + larger gaps + fewer samples = higher need
    density_factor = 1.0 / (1.0 + n_samples)
    gap_factor = min_pairwise_dist if min_pairwise_dist != float("inf") else 0.0

    return area * density_factor * (1.0 + gap_factor)


def compute_visual_disagreement(
    cell: AdaptiveCell,
    all_cells: Dict[str, AdaptiveCell],
    acquired_embeddings: Dict[int, Dict],
) -> float:
    """
    Compute visual disagreement for this cell using only acquired episodes.

    If cell has only 1 sample: use distance to neighboring cell representatives.
    If cell has multiple samples: compute internal variance + neighbor disagreement.

    Returns a non-negative score; higher means more visual diversity/disagreement.
    """
    if cell.n_samples == 0:
        # No visual data: return a moderate default to encourage exploration
        return 1.0

    cell_embeddings = []
    for ep_idx in cell.sample_episode_indices:
        if ep_idx in acquired_embeddings:
            emb = acquired_embeddings[ep_idx]
            combined = np.concatenate([emb["phi_global"], emb["phi_wrist"]])
            cell_embeddings.append(combined)

    if not cell_embeddings:
        return 1.0

    if len(cell_embeddings) == 1:
        # Single sample: measure distance to neighboring cell representatives
        neighbors = get_neighbor_cells(cell, all_cells)
        if not neighbors:
            return 1.0

        neighbor_dists = []
        for neighbor in neighbors:
            if neighbor.n_samples == 0:
                continue
            neighbor_embs = []
            for ep_idx in neighbor.sample_episode_indices:
                if ep_idx in acquired_embeddings:
                    emb = acquired_embeddings[ep_idx]
                    neighbor_embs.append(
                        np.concatenate([emb["phi_global"], emb["phi_wrist"]])
                    )
            if neighbor_embs:
                neighbor_rep = np.mean(neighbor_embs, axis=0)
                dist = np.linalg.norm(cell_embeddings[0] - neighbor_rep)
                neighbor_dists.append(dist)

        if not neighbor_dists:
            return 1.0

        return float(np.mean(neighbor_dists))

    # Multiple samples: compute internal pairwise distance + neighbor disagreement
    n = len(cell_embeddings)
    internal_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(cell_embeddings[i] - cell_embeddings[j])
            internal_dists.append(d)

    internal_score = float(np.mean(internal_dists)) if internal_dists else 0.0

    # Neighbor disagreement
    neighbors = get_neighbor_cells(cell, all_cells)
    neighbor_dists = []
    cell_rep = np.mean(cell_embeddings, axis=0)

    for neighbor in neighbors:
        if neighbor.n_samples == 0:
            continue
        neighbor_embs = []
        for ep_idx in neighbor.sample_episode_indices:
            if ep_idx in acquired_embeddings:
                emb = acquired_embeddings[ep_idx]
                neighbor_embs.append(
                    np.concatenate([emb["phi_global"], emb["phi_wrist"]])
                )
        if neighbor_embs:
            neighbor_rep = np.mean(neighbor_embs, axis=0)
            dist = np.linalg.norm(cell_rep - neighbor_rep)
            neighbor_dists.append(dist)

    neighbor_score = float(np.mean(neighbor_dists)) if neighbor_dists else 0.0

    return internal_score + neighbor_score


def normalize_scores(values: List[float]) -> List[float]:
    """Min-max normalize a list of scores to [0, 1]."""
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax - vmin < 1e-12:
        return [0.5] * len(values)
    return [(v - vmin) / (vmax - vmin) for v in values]


def compute_cell_priority(
    cell: AdaptiveCell,
    all_cells: Dict[str, AdaptiveCell],
    acquired_positions: Dict[int, Tuple[float, float, float]],
    acquired_embeddings: Dict[int, Dict],
    spatial_weight: float = SPATIAL_WEIGHT,
    visual_weight: float = VISUAL_WEIGHT,
) -> Tuple[float, float, float]:
    """
    Compute the overall priority for a cell.

    priority = SPATIAL_WEIGHT * normalized_spatial_need + VISUAL_WEIGHT * normalized_visual_disagreement

    Returns:
        (spatial_component, visual_component, final_priority)
    """
    spatial_need = compute_spatial_need(cell, acquired_positions)
    visual_disagreement = compute_visual_disagreement(cell, all_cells, acquired_embeddings)

    return spatial_need, visual_disagreement, 0.0  # raw scores, normalized in planner