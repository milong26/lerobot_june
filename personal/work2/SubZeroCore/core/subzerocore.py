"""
SubZeroCore Core Algorithm

Implements the SubZeroCore facility-location based episode selection algorithm.
"""

import numpy as np
from typing import Dict, List, Optional

from SubZeroCore.config import COVERAGE_GAMMA, EPS


def validate_embedding_matrix(X: np.ndarray) -> None:
    """Check that X is a 2-D numpy array with at least 2 samples and all finite elements."""
    if not isinstance(X, np.ndarray):
        raise TypeError(f"X must be a numpy array, got {type(X)}")
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if X.shape[0] < 2:
        raise ValueError(f"X must have at least 2 samples, got {X.shape[0]}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values (NaN or Inf)")


def compute_cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """
    Compute full cosine similarity matrix assuming X is already L2 normalized.
    Uses X @ X.T and clips to [-1, 1].
    """
    sim = X @ X.T
    sim = np.clip(sim, -1.0, 1.0)
    return sim


def prepare_similarity_matrix(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Convert cosine similarity to facility-location compatible similarity form.
    SubZeroCore uses max(0, similarity) to ensure non-negative similarities
    for the facility-location objective.
    """
    return np.maximum(similarity_matrix, 0.0)


def compute_cosine_distance_matrix(similarity_matrix: np.ndarray) -> np.ndarray:
    """Compute 1 - similarity and clip to [0, 2]."""
    dist = 1.0 - similarity_matrix
    dist = np.clip(dist, 0.0, 2.0)
    return dist


def determine_k(
    candidate_pool_size: int,
    target_size: int,
    coverage_gamma: float = COVERAGE_GAMMA,
    k_override: Optional[int] = None,
) -> int:
    """
    Determine K for KNN radius computation.
    If k_override is provided, validate and return it.
    Otherwise compute K based on SubZeroCore coverage target rule:
    K = ceil(coverage_gamma * N / M), clamped to [1, N-1].
    """
    if k_override is not None:
        if k_override < 1 or k_override >= candidate_pool_size:
            raise ValueError(
                f"k_override must be in [1, {candidate_pool_size - 1}], got {k_override}"
            )
        return k_override

    n = candidate_pool_size
    m = target_size
    k = int(np.ceil(coverage_gamma * n / m))
    k = max(1, min(k, n - 1))
    return k


def compute_knn_radius(distance_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    For each sample, compute the distance to its K-th nearest neighbor (excluding self).
    Returns array of length N.
    """
    n = distance_matrix.shape[0]
    knn_radius = np.zeros(n)

    for i in range(n):
        dists = distance_matrix[i].copy()
        dists[i] = np.inf
        knn_radius[i] = np.sort(dists)[k - 1]

    return knn_radius


def compute_density_weights(knn_radius: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Compute density weights from KNN radius following SubZeroCore algorithm.
    density_weight_i = 1 / (knn_radius_i + eps)
    Handles near-zero std by returning uniform weights if all radii are nearly identical.
    """
    if np.std(knn_radius) < eps:
        return np.ones(len(knn_radius))

    weights = 1.0 / (knn_radius + eps)
    weights = np.clip(weights, 0.0, np.inf)
    return weights


def compute_weighted_similarity_matrix(
    similarity_matrix: np.ndarray,
    density_weights: np.ndarray,
) -> np.ndarray:
    """
    Compute weighted similarity matrix per SubZeroCore paper.
    weighted_sim[i, j] = density_weight[j] * similarity[i, j]
    """
    return similarity_matrix * density_weights[np.newaxis, :]


def compute_candidate_marginal_gains(
    current_coverage: np.ndarray,
    weighted_similarity: np.ndarray,
    selected_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute facility-location marginal gain for each unselected candidate.
    marginal_gain[j] = sum_i max(0, weighted_sim[i, j] - current_coverage[i])
    Already selected candidates get -inf gain.
    """
    n = weighted_similarity.shape[0]
    gains = np.zeros(n)

    for j in range(n):
        if selected_mask[j]:
            gains[j] = -np.inf
            continue
        gain = np.sum(np.maximum(0.0, weighted_similarity[:, j] - current_coverage))
        gains[j] = gain

    return gains


def select_best_candidate(
    marginal_gains: np.ndarray,
    episode_indices: List[int],
    seed: int,
) -> int:
    """
    Select the candidate with the highest marginal gain.
    Uses deterministic tie-breaking: among equal gains, pick the one with
    the smallest episode index for reproducibility.
    """
    max_gain = -np.inf
    best_row = -1
    best_ep_idx = float("inf")

    for j in range(len(marginal_gains)):
        g = marginal_gains[j]
        if g == -np.inf:
            continue
        ep_idx = episode_indices[j]
        if g > max_gain or (g == max_gain and ep_idx < best_ep_idx):
            max_gain = g
            best_row = j
            best_ep_idx = ep_idx

    return best_row


def greedy_facility_location(
    weighted_similarity: np.ndarray,
    target_size: int,
    episode_indices: List[int],
    seed: int,
) -> Dict:
    """
    Greedy facility-location selection.
    Maintains current_coverage and selected_mask, selects one episode per round
    until target_size is reached.
    """
    n = weighted_similarity.shape[0]
    selected_mask = np.zeros(n, dtype=bool)
    current_coverage = np.zeros(n)
    selected_row_indices = []
    selection_order = []
    marginal_gains_history = []

    for _ in range(target_size):
        gains = compute_candidate_marginal_gains(
            current_coverage, weighted_similarity, selected_mask
        )
        best_row = select_best_candidate(gains, episode_indices, seed)

        selected_row_indices.append(int(best_row))
        selection_order.append(int(best_row))
        marginal_gains_history.append(float(gains[best_row]))

        selected_mask[best_row] = True
        current_coverage = np.maximum(current_coverage, weighted_similarity[:, best_row])

    final_objective = float(np.sum(current_coverage))

    return {
        "selected_row_indices": selected_row_indices,
        "selection_order": selection_order,
        "marginal_gains": marginal_gains_history,
        "final_objective": final_objective,
    }


def map_rows_to_episode_indices(
    selected_row_indices: List[int],
    episode_indices: List[int],
) -> List[int]:
    """Convert internal row indices to real episode indices."""
    return [episode_indices[r] for r in selected_row_indices]


def run_subzerocore(
    X: np.ndarray,
    episode_indices: List[int],
    target_size: int,
    coverage_gamma: float = COVERAGE_GAMMA,
    k_override: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    """
    Run the full SubZeroCore selection pipeline.
    Returns a result dictionary with all intermediate and final results.
    """
    validate_embedding_matrix(X)

    similarity_matrix = compute_cosine_similarity_matrix(X)
    sim_prepared = prepare_similarity_matrix(similarity_matrix)
    distance_matrix = compute_cosine_distance_matrix(sim_prepared)

    n = X.shape[0]
    k = determine_k(n, target_size, coverage_gamma, k_override)

    knn_radius = compute_knn_radius(distance_matrix, k)
    density_weights = compute_density_weights(knn_radius)

    weighted_sim = compute_weighted_similarity_matrix(sim_prepared, density_weights)

    fl_result = greedy_facility_location(weighted_sim, target_size, episode_indices, seed)

    selected_episode_indices = map_rows_to_episode_indices(
        fl_result["selected_row_indices"], episode_indices
    )

    return {
        "selected_episode_indices": selected_episode_indices,
        "selected_row_indices": fl_result["selected_row_indices"],
        "k": k,
        "knn_radius": knn_radius,
        "density_weights": density_weights,
        "marginal_gains": fl_result["marginal_gains"],
        "final_objective": fl_result["final_objective"],
        "selection_order": fl_result["selection_order"],
        "candidate_pool_size": n,
        "target_size": target_size,
        "coverage_gamma": coverage_gamma,
        "embedding_dim": X.shape[1],
    }