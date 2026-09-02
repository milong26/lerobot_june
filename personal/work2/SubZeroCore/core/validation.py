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


def validate_knn_radius(knn_radius: np.ndarray, candidate_pool_size: int) -> List[str]:
    """Check KNN radius length equals candidate_pool_size, all finite, all >= 0."""
    issues = []

    if len(knn_radius) != candidate_pool_size:
        issues.append(
            f"knn_radius length {len(knn_radius)} must equal candidate_pool_size {candidate_pool_size}"
        )
        return issues

    if not np.all(np.isfinite(knn_radius)):
        issues.append("knn_radius contains non-finite values")

    if np.any(knn_radius < 0):
        issues.append("knn_radius contains negative values")

    return issues


def validate_density_weights(density_weights: np.ndarray, candidate_pool_size: int) -> List[str]:
    """
    Check density weights length equals candidate_pool_size, all finite,
    all >= 0 and <= 1 + 1e-8 (Gaussian formula theoretical range is (0, 1]).
    """
    issues = []

    if len(density_weights) != candidate_pool_size:
        issues.append(
            f"density_weights length {len(density_weights)} must equal candidate_pool_size {candidate_pool_size}"
        )
        return issues

    if not np.all(np.isfinite(density_weights)):
        issues.append("density_weights contains non-finite values")

    if np.any(density_weights < 0):
        issues.append("density_weights contains negative values")

    if np.any(density_weights > 1.0 + 1e-8):
        issues.append(
            f"density_weights exceeds theoretical max 1.0, max={np.max(density_weights):.6e}"
        )

    return issues


def validate_similarity_matrix(similarity_matrix: np.ndarray, candidate_pool_size: int) -> List[str]:
    """
    Check shape is (candidate_pool_size, candidate_pool_size), all finite,
    all values in [-1 - 1e-8, 1 + 1e-8]. Does NOT require non-negative.
    """
    issues = []

    if similarity_matrix.ndim != 2:
        issues.append(f"similarity_matrix must be 2-D, got {similarity_matrix.ndim}-D")
        return issues

    expected_shape = (candidate_pool_size, candidate_pool_size)
    if similarity_matrix.shape != expected_shape:
        issues.append(
            f"similarity_matrix shape {similarity_matrix.shape} must be {expected_shape}"
        )
        return issues

    if not np.all(np.isfinite(similarity_matrix)):
        issues.append("similarity_matrix contains non-finite values")

    if np.any(similarity_matrix < -1.0 - 1e-8) or np.any(similarity_matrix > 1.0 + 1e-8):
        issues.append(
            f"similarity_matrix values out of [-1, 1], "
            f"min={np.min(similarity_matrix):.6f}, max={np.max(similarity_matrix):.6f}"
        )

    return issues


def validate_weighted_similarity_matrix(weighted_similarity: np.ndarray, candidate_pool_size: int) -> List[str]:
    """
    Check shape is (N, N), all finite. Does NOT require non-negative
    because raw cosine can be negative.
    """
    issues = []

    if weighted_similarity.ndim != 2:
        issues.append(
            f"weighted_similarity_matrix must be 2-D, got {weighted_similarity.ndim}-D"
        )
        return issues

    expected_shape = (candidate_pool_size, candidate_pool_size)
    if weighted_similarity.shape != expected_shape:
        issues.append(
            f"weighted_similarity_matrix shape {weighted_similarity.shape} must be {expected_shape}"
        )
        return issues

    if not np.all(np.isfinite(weighted_similarity)):
        issues.append("weighted_similarity_matrix contains non-finite values")

    return issues


def validate_k(k: int, candidate_pool_size: int, target_size: int) -> List[str]:
    """Check 1 <= K < candidate_pool_size and target_size < candidate_pool_size."""
    issues = []

    if k < 1:
        issues.append(f"K must be >= 1, got {k}")

    if k >= candidate_pool_size:
        issues.append(
            f"K must be < candidate_pool_size ({candidate_pool_size}), got {k}"
        )

    if target_size >= candidate_pool_size:
        issues.append(
            f"target_size ({target_size}) must be < candidate_pool_size ({candidate_pool_size})"
        )

    return issues


def validate_marginal_gains(marginal_gains: List[float], target_size: int) -> List[str]:
    """Check length equals target_size and all are finite."""
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


def validate_final_objective(final_objective: float) -> List[str]:
    """Check that final_objective is finite."""
    issues = []

    if not np.isfinite(final_objective):
        issues.append(f"final_objective is non-finite: {final_objective}")

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
    - target_size < candidate_pool_size (SubZeroCore requires pruning)
    - K is valid (1 <= K < candidate_pool_size)
    - knn_radius is valid
    - density_weights is valid
    - similarity_matrix is valid (if provided)
    - weighted_similarity_matrix is valid (if provided)
    - marginal_gains is valid
    - final_objective is valid
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

    issues.extend(validate_k(result["k"], candidate_pool_size, target_size))
    issues.extend(validate_knn_radius(result["knn_radius"], candidate_pool_size))
    issues.extend(validate_density_weights(result["density_weights"], candidate_pool_size))
    issues.extend(validate_marginal_gains(result["marginal_gains"], target_size))
    issues.extend(validate_final_objective(result["final_objective"]))

    if similarity_matrix is not None:
        issues.extend(validate_similarity_matrix(similarity_matrix, candidate_pool_size))

    if weighted_similarity_matrix is not None:
        issues.extend(validate_weighted_similarity_matrix(weighted_similarity_matrix, candidate_pool_size))

    if embedding_matrix is not None:
        issues.extend(validate_embedding_matrix_result(embedding_matrix))

    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }