"""
Validation utilities for SubZeroCore selection results.
"""

import numpy as np
from typing import Dict, List


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


def validate_selection_result(
    result: Dict,
    candidate_episode_indices: List[int],
    target_size: int,
) -> Dict:
    """
    Run all validations on the selection result.
    Returns {"valid": bool, "issues": list[str]}.
    """
    issues = []

    issues.extend(
        validate_selected_indices(
            result["selected_episode_indices"],
            candidate_episode_indices,
            target_size,
        )
    )

    issues.extend(validate_knn_radius(result["knn_radius"]))
    issues.extend(validate_density_weights(result["density_weights"]))

    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }