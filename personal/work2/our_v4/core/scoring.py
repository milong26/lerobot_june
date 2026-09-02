"""
Scoring Module for V4

Computes spatial need, visual disagreement, and action disagreement for each cell.
All weights are loaded from config.py, never hard-coded.

Spatial need formula:
    spatial_need = area / (1 + n_samples)

Action disagreement:
    - 0 samples: neutral default (1.0)
    - 1 sample: distance to neighbor cell representative action embeddings
    - 2+ samples: internal pairwise distance + neighbor disagreement
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from our_v4.core.adaptive_grid import (
    AdaptiveCell, get_leaf_cells, get_neighbor_cells,
    compute_cell_area, cell_center,
)
from our_v4.core.visual_embedding import build_weighted_visual_embedding
from our_v4.config import (
    SPATIAL_WEIGHT, VISUAL_WEIGHT, ACTION_WEIGHT,
    VISUAL_GLOBAL_WEIGHT, VISUAL_WRIST_WEIGHT,
)


def compute_spatial_need(
    cell: AdaptiveCell,
    acquired_positions: Dict[int, Tuple[float, float, float]],
) -> float:
    """
    Measure how sparsely sampled this cell currently is.

    Formula design:
        Base: spatial_need = area / (1 + n_samples)

        For cells with multiple samples (n_samples >= 2), an additional
        coverage gap factor is added to encourage filling spatial gaps:
            coverage_gap = maximin_radius / cell_radius
            spatial_need = area / (1 + n_samples) * (1 + coverage_gap)

    Properties:
        - Under the same area: 0-sample > 1-sample > multi-sample
        - Larger cells have higher base need than smaller cells
        - Multi-sample cells with large gaps get additional boost

    Returns a non-negative score; higher means more spatial need.
    """
    area = compute_cell_area(cell)
    n_samples = cell.n_samples
    cell_radius = np.sqrt(area) / 2.0

    if n_samples == 0:
        return area

    base_need = area / (1.0 + n_samples)

    if n_samples == 1:
        return base_need

    positions = cell.sample_positions
    if len(positions) < 2:
        return base_need

    n_candidates = 30
    rng = np.random.RandomState(42)
    candidates_x = rng.uniform(cell.x_min, cell.x_max, n_candidates)
    candidates_y = rng.uniform(cell.y_min, cell.y_max, n_candidates)

    maximin_radius = 0.0
    for cx, cy in zip(candidates_x, candidates_y):
        min_dist = min(
            np.sqrt((cx - px)**2 + (cy - py)**2)
            for px, py, _ in positions
        )
        maximin_radius = max(maximin_radius, min_dist)

    coverage_gap = maximin_radius / cell_radius if cell_radius > 1e-9 else 0.0

    return base_need * (1.0 + coverage_gap)


def _get_combined_embedding(emb: Dict) -> np.ndarray:
    """Get weighted combined visual embedding from an episode's embedding dict."""
    return build_weighted_visual_embedding(
        emb["phi_global"],
        emb["phi_wrist"],
        global_weight=VISUAL_GLOBAL_WEIGHT,
        wrist_weight=VISUAL_WRIST_WEIGHT,
    )


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
        return 1.0

    cell_embeddings = []
    for ep_idx in cell.sample_episode_indices:
        if ep_idx in acquired_embeddings:
            emb = acquired_embeddings[ep_idx]
            combined = _get_combined_embedding(emb)
            cell_embeddings.append(combined)

    if not cell_embeddings:
        return 1.0

    if len(cell_embeddings) == 1:
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
                    neighbor_embs.append(_get_combined_embedding(emb))
            if neighbor_embs:
                neighbor_rep = np.mean(neighbor_embs, axis=0)
                dist = np.linalg.norm(cell_embeddings[0] - neighbor_rep)
                neighbor_dists.append(dist)

        if not neighbor_dists:
            return 1.0

        return float(np.mean(neighbor_dists))

    n = len(cell_embeddings)
    internal_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(cell_embeddings[i] - cell_embeddings[j])
            internal_dists.append(d)

    internal_score = float(np.mean(internal_dists)) if internal_dists else 0.0

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
                neighbor_embs.append(_get_combined_embedding(emb))
        if neighbor_embs:
            neighbor_rep = np.mean(neighbor_embs, axis=0)
            dist = np.linalg.norm(cell_rep - neighbor_rep)
            neighbor_dists.append(dist)

    neighbor_score = float(np.mean(neighbor_dists)) if neighbor_dists else 0.0

    return internal_score + neighbor_score


def compute_action_disagreement(
    cell: AdaptiveCell,
    all_cells: Dict[str, AdaptiveCell],
    acquired_action_embeddings: Dict[int, np.ndarray],
) -> float:
    """
    Compute action disagreement for this cell using only acquired action embeddings.

    Rules:
    - 0 samples: neutral default (1.0), same as visual disagreement
    - 1 sample: distance to neighboring cell representative action embeddings
    - 2+ samples: internal pairwise distance + neighbor disagreement

    All action distances are based on L2-normalized embeddings.

    Returns a non-negative score; higher means more action diversity/disagreement.
    Empty cells get a neutral default, NOT infinity.
    """
    if cell.n_samples == 0:
        return 1.0

    cell_action_embs = []
    for ep_idx in cell.sample_episode_indices:
        if ep_idx in acquired_action_embeddings:
            cell_action_embs.append(acquired_action_embeddings[ep_idx])

    if not cell_action_embs:
        return 1.0

    if len(cell_action_embs) == 1:
        neighbors = get_neighbor_cells(cell, all_cells)
        if not neighbors:
            return 1.0

        neighbor_dists = []
        for neighbor in neighbors:
            if neighbor.n_samples == 0:
                continue
            neighbor_embs = []
            for ep_idx in neighbor.sample_episode_indices:
                if ep_idx in acquired_action_embeddings:
                    neighbor_embs.append(acquired_action_embeddings[ep_idx])
            if neighbor_embs:
                neighbor_rep = np.mean(neighbor_embs, axis=0)
                dist = np.linalg.norm(cell_action_embs[0] - neighbor_rep)
                neighbor_dists.append(dist)

        if not neighbor_dists:
            return 1.0

        return float(np.mean(neighbor_dists))

    n = len(cell_action_embs)
    internal_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(cell_action_embs[i] - cell_action_embs[j])
            internal_dists.append(d)

    internal_score = float(np.mean(internal_dists)) if internal_dists else 0.0

    neighbors = get_neighbor_cells(cell, all_cells)
    neighbor_dists = []
    cell_rep = np.mean(cell_action_embs, axis=0)

    for neighbor in neighbors:
        if neighbor.n_samples == 0:
            continue
        neighbor_embs = []
        for ep_idx in neighbor.sample_episode_indices:
            if ep_idx in acquired_action_embeddings:
                neighbor_embs.append(acquired_action_embeddings[ep_idx])
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
    acquired_action_embeddings: Dict[int, np.ndarray],
    spatial_weight: float = SPATIAL_WEIGHT,
    visual_weight: float = VISUAL_WEIGHT,
    action_weight: float = ACTION_WEIGHT,
) -> Tuple[float, float, float, float]:
    """
    Compute the overall priority for a cell with all three signals.

    Returns raw (unnormalized) spatial_need, visual_disagreement, action_disagreement,
    and the weighted sum. Normalization is done at the planner level across all leaf cells.

    Returns:
        (raw_spatial_need, raw_visual_disagreement, raw_action_disagreement, weighted_sum)
    """
    spatial_need = compute_spatial_need(cell, acquired_positions)
    visual_disagreement = compute_visual_disagreement(cell, all_cells, acquired_embeddings)
    action_disagreement = compute_action_disagreement(cell, all_cells, acquired_action_embeddings)
    weighted_sum = (spatial_weight * spatial_need +
                    visual_weight * visual_disagreement +
                    action_weight * action_disagreement)

    return spatial_need, visual_disagreement, action_disagreement, weighted_sum