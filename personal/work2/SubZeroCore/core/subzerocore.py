"""
SubZeroCore Core Algorithm

Implements the SubZeroCore facility-location based episode selection algorithm.

Algorithm pipeline (fixed):
1. Official numerical inversion to determine K
2. Raw cosine distance to compute K-th nearest neighbor distance r_i
3. Gaussian density score s_i = exp(-((r_i - mu)^2) / (2 * sigma^2))
4. Raw cosine similarity sim(x_i, x_j) = x_i^T x_j
5. Weighted similarity s_j * sim(x_i, x_j)
6. Greedy selection optimizing F(S) = sum_i max_{j in S} [s_j * sim(x_i, x_j)]
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
    Compute raw cosine similarity matrix assuming X is already L2 normalized.
    sim(i, j) = x_i^T x_j

    Only clips to [-1, 1] to correct floating point errors.
    """
    sim = X @ X.T
    sim = np.clip(sim, -1.0, 1.0)
    return sim


def compute_cosine_distance_matrix(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Compute cosine distance from raw cosine similarity.
    distance(i, j) = 1 - similarity(i, j)

    Clips to [0, 2] to correct floating point errors.
    This distance matrix is only used for KNN radius computation.
    """
    dist = 1.0 - similarity_matrix
    dist = np.clip(dist, 0.0, 2.0)
    return dist


def find_k_for_coverage(
    candidate_pool_size: int,
    target_size: int,
    coverage_gamma: float,
) -> int:
    """
    Find K using the official SubZeroCore numerical inversion method.

    N = candidate_pool_size, M = target_size, gamma = coverage_gamma.

    Algorithm:
        k = 0, coverage = 0.0, numerator = 1.0, denominator = 1.0
        while coverage < gamma and k < N - M - 1:
            k += 1
            numerator *= (N - M - k)
            denominator *= (N - k)
            coverage = 1.0 - numerator / denominator
        return k

    For N=500, M=112, gamma=0.6, this yields K=4.
    """
    n = candidate_pool_size
    m = target_size

    k = 0
    coverage = 0.0
    numerator = 1.0
    denominator = 1.0

    while coverage < coverage_gamma and k < n - m - 1:
        k += 1
        numerator *= (n - m - k)
        denominator *= (n - k)
        coverage = 1.0 - numerator / denominator

    return k


def determine_k(
    candidate_pool_size: int,
    target_size: int,
    coverage_gamma: float = COVERAGE_GAMMA,
    k_override: Optional[int] = None,
) -> int:
    """
    Determine K for KNN radius computation.

    If k_override is provided, validate 1 <= k_override < candidate_pool_size and return it.
    Otherwise, call find_k_for_coverage using the official numerical inversion.
    """
    if k_override is not None:
        if k_override < 1 or k_override >= candidate_pool_size:
            raise ValueError(
                f"k_override must be in [1, {candidate_pool_size - 1}], got {k_override}"
            )
        return k_override

    return find_k_for_coverage(candidate_pool_size, target_size, coverage_gamma)


def compute_knn_radius(distance_matrix: np.ndarray, k: int) -> np.ndarray:
    """
    For each sample i, compute the distance to its K-th nearest neighbor (excluding self).

    For each row i: copy distance_matrix[i], set dists[i] = inf, sort remaining,
    take sorted_distances[k-1].

    Returns knn_radius of shape [N], where knn_radius[i] = NND_K(x_i).
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
    Compute Gaussian density weights from KNN radius per SubZeroCore algorithm.

    Formula: s_i = exp(-((r_i - mu)^2) / (2 * sigma^2))

    where mu = mean(knn_radius), sigma = std(knn_radius).

    If sigma < eps, returns all-ones weights to avoid division by zero.

    Note: This is NOT 1/r density. The Gaussian weight is highest (close to 1)
    when r_i is close to the mean mu, and decreases as r_i deviates from mu.
    Episodes with radius near the population mean get weight ~1; episodes
    with very small or very large radius get lower weight.

    Returns non-negative finite weights with same shape as knn_radius.
    """
    mu = np.mean(knn_radius)
    sigma = np.std(knn_radius)

    if sigma < eps:
        return np.ones(len(knn_radius))

    weights = np.exp(-((knn_radius - mu) ** 2) / (2 * sigma ** 2))
    return weights


def compute_weighted_similarity_matrix(
    similarity_matrix: np.ndarray,
    density_weights: np.ndarray,
) -> np.ndarray:
    """
    Compute density-weighted similarity matrix per SubZeroCore.

    weighted_sim[i, j] = s_j * sim(x_i, x_j)

    The density weight is applied on the column (candidate) dimension.
    """
    return similarity_matrix * density_weights[np.newaxis, :]


def compute_candidate_marginal_gains(
    current_coverage: Optional[np.ndarray],
    weighted_similarity: np.ndarray,
    selected_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute facility-location marginal gain for each unselected candidate.

    If current_coverage is None (first round, S is empty):
        gain[j] = sum_i weighted_similarity[i, j]

    If current_coverage is not None (S is non-empty):
        gain[j] = sum_i (max(current_coverage[i], weighted_similarity[i, j]) - current_coverage[i])

    Already selected candidates get -inf gain.
    """
    n = weighted_similarity.shape[0]
    gains = np.zeros(n)

    if current_coverage is None:
        for j in range(n):
            if selected_mask[j]:
                gains[j] = -np.inf
                continue
            gains[j] = np.sum(weighted_similarity[:, j])
    else:
        for j in range(n):
            if selected_mask[j]:
                gains[j] = -np.inf
                continue
            candidate_coverage = np.maximum(current_coverage, weighted_similarity[:, j])
            gains[j] = np.sum(candidate_coverage) - np.sum(current_coverage)

    return gains


def select_best_candidate(
    marginal_gains: np.ndarray,
    episode_indices: List[int],
    seed: int,
) -> int:
    """
    Select the candidate with the highest marginal gain.

    Deterministic tie-breaking: among candidates with identical gain,
    pick the one with the smallest real episode index.

    Raises RuntimeError if no valid candidate is available.
    """
    max_gain = -np.inf
    best_row = -1
    best_ep_idx = float("inf")
    found = False

    for j in range(len(marginal_gains)):
        g = marginal_gains[j]
        if g == -np.inf:
            continue
        ep_idx = episode_indices[j]
        if g > max_gain or (g == max_gain and ep_idx < best_ep_idx):
            max_gain = g
            best_row = j
            best_ep_idx = ep_idx
            found = True

    if not found:
        raise RuntimeError(
            "No valid candidate available for selection. "
            "All candidates may already be selected or have invalid gains."
        )

    return best_row


def greedy_facility_location(
    weighted_similarity: np.ndarray,
    target_size: int,
    episode_indices: List[int],
    seed: int,
) -> Dict:
    """
    Greedy facility-location selection optimizing:
    F(S) = sum_i max_{j in S} [s_j * sim(x_i, x_j)]

    First round (S is empty): select candidate with largest singleton objective
        singleton_obj[j] = sum_i weighted_similarity[i, j]

    Subsequent rounds: select candidate with largest marginal gain
        marginal_gain[j] = F(S union {j}) - F(S)

    Returns:
        selected_row_indices: internal row indices of selected candidates
        selection_order_episode_indices: real episode indices in selection order
        marginal_gains: marginal gain at each selection step
        final_objective: final facility-location objective value F(S)
    """
    n = weighted_similarity.shape[0]
    selected_mask = np.zeros(n, dtype=bool)
    current_coverage = None
    current_objective = 0.0
    selected_row_indices = []
    selection_order_episode_indices = []
    marginal_gains_history = []

    for step in range(target_size):
        gains = compute_candidate_marginal_gains(
            current_coverage, weighted_similarity, selected_mask
        )
        best_row = select_best_candidate(gains, episode_indices, seed)

        assert not selected_mask[best_row], (
            f"Selected row {best_row} was already marked as selected"
        )

        selected_row_indices.append(int(best_row))
        selection_order_episode_indices.append(int(episode_indices[best_row]))
        marginal_gains_history.append(float(gains[best_row]))

        selected_mask[best_row] = True

        if current_coverage is None:
            current_coverage = weighted_similarity[:, best_row].copy()
            current_objective = float(np.sum(current_coverage))
        else:
            current_coverage = np.maximum(current_coverage, weighted_similarity[:, best_row])
            current_objective = float(np.sum(current_coverage))

    final_objective = current_objective

    return {
        "selected_row_indices": selected_row_indices,
        "selection_order_episode_indices": selection_order_episode_indices,
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

    Strict execution order:
    1. validate_embedding_matrix(X)
    2. N = X.shape[0]; check 1 <= target_size < N
    3. check 0 < coverage_gamma < 1
    4. check len(episode_indices) == N
    5. raw_similarity = compute_cosine_similarity_matrix(X)
    6. distance_matrix = compute_cosine_distance_matrix(raw_similarity)
    7. K = determine_k(N, target_size, coverage_gamma, k_override)
    8. check 1 <= K < N
    9. knn_radius = compute_knn_radius(distance_matrix, K)
    10. density_weights = compute_density_weights(knn_radius)
    11. weighted_similarity = compute_weighted_similarity_matrix(raw_similarity, density_weights)
    12. greedy_facility_location(weighted_similarity, target_size, episode_indices, seed)
    13. map_rows_to_episode_indices

    Returns a result dictionary with all intermediate and final results.
    """
    n = X.shape[0]

    validate_embedding_matrix(X)

    if target_size < 1:
        raise ValueError(f"target_size must be >= 1, got {target_size}")
    if target_size >= n:
        raise ValueError(
            f"target_size ({target_size}) must be < candidate_pool_size ({n}). "
            "SubZeroCore requires pruning from a full candidate pool."
        )
    if len(episode_indices) != n:
        raise ValueError(
            f"len(episode_indices)={len(episode_indices)} must match X.shape[0]={n}"
        )
    if not (0 < coverage_gamma < 1):
        raise ValueError(f"coverage_gamma must be in (0, 1), got {coverage_gamma}")

    raw_similarity = compute_cosine_similarity_matrix(X)
    distance_matrix = compute_cosine_distance_matrix(raw_similarity)

    k = determine_k(n, target_size, coverage_gamma, k_override)

    if k < 1 or k >= n:
        raise ValueError(
            f"Computed K={k} must satisfy 1 <= K < candidate_pool_size ({n})"
        )

    knn_radius = compute_knn_radius(distance_matrix, k)
    density_weights = compute_density_weights(knn_radius)

    weighted_sim = compute_weighted_similarity_matrix(raw_similarity, density_weights)

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
        "selection_order_episode_indices": fl_result["selection_order_episode_indices"],
        "similarity_matrix": raw_similarity,
        "weighted_similarity_matrix": weighted_sim,
        "candidate_pool_size": n,
        "target_size": target_size,
        "coverage_gamma": coverage_gamma,
        "embedding_dim": X.shape[1],
    }