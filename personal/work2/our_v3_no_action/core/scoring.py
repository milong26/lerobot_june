"""
Scoring Module for V3 No-Action

Computes spatial need and visual disagreement for each cell.
All weights are loaded from config.py, never hard-coded.

Spatial need formula:
    spatial_need = area / (1 + n_samples)
    
    This ensures:
    - 0 samples: spatial_need = area (maximum need)
    - 1 sample: spatial_need = area / 2
    - n samples: spatial_need = area / (1 + n)
    
    Under the same area: 0-sample > 1-sample > multi-sample.
    Larger cells naturally have higher spatial need than smaller cells.
    
    For cells with multiple samples, an additional coverage gap factor
    is added based on the maximin radius (largest empty circle within
    the cell relative to existing samples), encouraging filling gaps.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional

from our_v3_no_action.core.adaptive_grid import (
    AdaptiveCell, get_leaf_cells, get_neighbor_cells,
    compute_cell_area, cell_center,
)
from our_v3_no_action.core.visual_embedding import build_weighted_visual_embedding
from our_v3_no_action.config import (
    SPATIAL_WEIGHT, VISUAL_WEIGHT,
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
        
        The maximin_radius is the largest distance from any candidate point
        inside the cell to the nearest existing sample, normalized by the
        approximate cell radius (sqrt(area) / 2).

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
        # No samples yet: maximum spatial need proportional to area
        return area

    # Base spatial need: decreases with more samples
    base_need = area / (1.0 + n_samples)

    if n_samples == 1:
        # Single sample: use base need only
        # A large cell with 1 sample still has significant need for more exploration
        return base_need

    # Multiple samples: add coverage gap factor
    positions = cell.sample_positions
    if len(positions) < 2:
        return base_need

    # Compute maximin radius: pick candidate points in cell, find the one
    # farthest from all existing samples
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

    # Normalize by cell radius
    coverage_gap = maximin_radius / cell_radius if cell_radius > 1e-9 else 0.0

    return base_need * (1.0 + coverage_gap)


def _get_combined_embedding(emb: Dict) -> np.ndarray:
    """
    Get weighted combined visual embedding from an episode's embedding dict.
    Uses build_weighted_visual_embedding to respect VISUAL_GLOBAL_WEIGHT
    and VISUAL_WRIST_WEIGHT from config.
    """
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
        # No visual data: return a moderate default to encourage exploration
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
                    neighbor_embs.append(_get_combined_embedding(emb))
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
                neighbor_embs.append(_get_combined_embedding(emb))
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

    Returns raw (unnormalized) spatial_need, visual_disagreement, and
    the weighted sum. Normalization is done at the planner level across
    all leaf cells.

    Returns:
        (raw_spatial_need, raw_visual_disagreement, weighted_sum)
    """
    spatial_need = compute_spatial_need(cell, acquired_positions)
    visual_disagreement = compute_visual_disagreement(cell, all_cells, acquired_embeddings)
    weighted_sum = spatial_weight * spatial_need + visual_weight * visual_disagreement

    return spatial_need, visual_disagreement, weighted_sum