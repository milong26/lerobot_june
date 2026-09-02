"""
Validation utilities for SubZeroCore selection results.
"""

import numpy as np
from typing import Dict, List, Optional


def validate_selected_indices(
    selected_episode_indices: List[int],
    candidate_episode_indices: List[int],
    target_size: int,
) -> List[str]:
    """
    Check selection count, uniqueness, and that all selected episodes
    belong to the candidate pool.
    """
    issues = []

    if len(selected_episode_indices) != target_size:
        issues.append(
            f"Expected {target_size} selected episodes, got {len(selected_episode_indices)}"
        )

    if len(selected_episode_indices) != len(set(selected_episode_indices)):
        issues.append("Duplicate episodes found in selected_episode_indices")

    candidate_set = set(candidate_episode_indices)
    invalid = [ep for ep in selected_episode_indices if ep not in candidate_set]
    if invalid:
        issues.append(f"Selected episodes not in candidate pool: {invalid}")

    return issues


def validate_knn_radius(knn_radius: np.ndarray) -> List[str]:
    """Check KNN radius length, finiteness, and non-negativity."""
    issues = []

    if len(knn_radius) == 0:
        issues.append("knn_radius is empty")
        return issues

    if not np.all(np.isfinite(knn_radius)):
        issues.append("knn_radius contains non-finite values")

    if np.any(knn_radius < 0):
        issues.append("knn_radius contains negative values")

    return issues


def validate_density_weights(density_weights: np.ndarray) -> List[str]:
    """Check density weights length, finiteness, and non-negativity."""
    issues = []

    if len(density_weights) == 0:
        issues.append("density_weights is empty")
        return issues

    if not np.all(np.isfinite(density_weights)):
        issues.append("density_weights contains non-finite values")

    if np.any(density_weights < 0):
        issues.append("density_weights contains negative values")

    return issues


def validate_similarity_matrix(similarity_matrix: np.ndarray) -> List[str]:
    """Check that similarity matrix is square and all finite."""
    issues = []

    if similarity_matrix.ndim != 2:
        issues.append(f"similarity_matrix must be 2-D, got {similarity_matrix.ndim}-D")
        return issues

    if similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        issues.append(
            f"similarity_matrix must be square, got shape {similarity_matrix.shape}"
        )

    if not np.all(np.isfinite(similarity_matrix)):
        issues.append("similarity_matrix contains non-finite values")

    return issues


def validate_k(k: int, candidate_pool_size: int) -> List[str]:
    """Check that 1 <= K < candidate_pool_size."""
    issues = []

    if k < 1:
        issues.append(f"K must be >= 1, got {k}")

    if k >= candidate_pool_size:
        issues.append(
            f"K must be < candidate_pool_size ({candidate_pool_size}), got {k}"
        )

    return issues


def validate_marginal_gains(marginal_gains: List[float], target_size: int) -> List[str]:
    """Check that marginal gains length equals target_size and all are finite."""
    issues = []

    if len(marginal_gains) != target_size:
        issues.append(
            f"Expected {target_size} marginal gains, got {len(marginal_gains)}"
        )

    for i, g in enumerate(marginal_gains):
        if not np.isfinite(g):
            issues.append(f"marginal_gains[{i}] is non-finite: {g}")
            break

    return issues


def validate_weighted_similarity_matrix(weighted_similarity: np.ndarray) -> List[str]:
    """Check that weighted similarity matrix is 2-D square and all finite."""
    issues = []

    if weighted_similarity.ndim != 2:
        issues.append(
            f"weighted_similarity_matrix must be 2-D, got {weighted_similarity.ndim}-D"
        )
        return issues

    if weighted_similarity.shape[0] != weighted_similarity.shape[1]:
        issues.append(
            f"weighted_similarity_matrix must be square, got shape {weighted_similarity.shape}"
        )

    if not np.all(np.isfinite(weighted_similarity)):
        issues.append("weighted_similarity_matrix contains non-finite values")

    return issues


def validate_embedding_matrix_result(X: np.ndarray) -> List[str]:
    """
    Check that X is 2-D, all finite, and each row has L2 norm close to 1.
    Allows reasonable floating point error (|norm - 1| < 1e-5).
    """
    issues = []

    if X.ndim != 2:
        issues.append(f"X must be 2-D, got {X.ndim}-D")
        return issues

    if not np.all(np.isfinite(X)):
        issues.append("X contains non-finite values")
        return issues

    norms = np.linalg.norm(X, axis=1)
    bad_norms = np.where(np.abs(norms - 1.0) > 1e-5)[0]
    if len(bad_norms) > 0:
        issues.append(
            f"{len(bad_norms)} rows have L2 norm not close to 1 "
            f"(max deviation: {np.max(np.abs(norms - 1.0)):.2e})"
        )

    return issues


def validate_selection_result(
    result: Dict,
    candidate_episode_indices: List[int],
    target_size: int,
    embedding_matrix: Optional[np.ndarray] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    weighted_similarity_matrix: Optional[np.ndarray] = None,
) -> Dict:
    """
    Run all validations on the selection result.

    Checks:
    - selected count matches target_size
    - selected episodes are unique
    - all selected episodes belong to candidate pool
    - target_size <= candidate pool size
    - K is valid (1 <= K < candidate_pool_size)
    - knn_radius is valid
    - density_weights is valid
    - similarity_matrix is valid (if provided)
    - weighted_similarity_matrix is valid (if provided)
    - marginal_gains is valid
    - embedding_matrix is valid (if provided)

    Returns {"valid": bool, "issues": list[str]}.
    """
    issues = []
    candidate_pool_size = len(candidate_episode_indices)

    issues.extend(
        validate_selected_indices(
            result["selected_episode_indices"],
            candidate_episode_indices,
            target_size,
        )
    )

    if target_size > candidate_pool_size:
        issues.append(
            f"target_size ({target_size}) must be <= candidate_pool_size ({candidate_pool_size})"
        )

    issues.extend(validate_k(result["k"], candidate_pool_size))
    issues.extend(validate_knn_radius(result["knn_radius"]))
    issues.extend(validate_density_weights(result["density_weights"]))
    issues.extend(validate_marginal_gains(result["marginal_gains"], target_size))

    if similarity_matrix is not None:
        issues.extend(validate_similarity_matrix(similarity_matrix))

    if weighted_similarity_matrix is not None:
        issues.extend(validate_weighted_similarity_matrix(weighted_similarity_matrix))

    if embedding_matrix is not None:
        issues.extend(validate_embedding_matrix_result(embedding_matrix))

    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }