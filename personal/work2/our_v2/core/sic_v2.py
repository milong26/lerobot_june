"""
V2 SIC (State-Action Information Coverage) Computation Module

Implements pairwise distance, marginal SIC gain, redundancy penalty,
and candidate scoring for the state-action manifold.
"""

import numpy as np
from typing import Dict, List, Tuple


def validate_embedding_dimension(embeddings: Dict[int, np.ndarray]) -> int:
    """
    Validate that all episode embeddings have consistent dimensions.

    Args:
        embeddings: dict of episode embeddings.

    Returns:
        Embedding dimension.

    Raises:
        ValueError: if dimensions are inconsistent.
    """
    if not embeddings:
        raise ValueError("No embeddings provided for validation.")

    indices = list(embeddings.keys())
    ref_dim = embeddings[indices[0]].shape[0]

    for idx in indices[1:]:
        dim = embeddings[idx].shape[0]
        if dim != ref_dim:
            raise ValueError(
                f"Inconsistent embedding dimensions: "
                f"episode {indices[0]} has dim={ref_dim}, "
                f"episode {idx} has dim={dim}. "
                f"All embeddings must have the same dimension."
            )

    return ref_dim


def compute_pairwise_distance(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
) -> float:
    """
    Compute L2 distance between two state-action embeddings.

    Args:
        emb_a: embedding vector A.
        emb_b: embedding vector B.

    Returns:
        L2 distance as float.
    """
    return float(np.linalg.norm(emb_a - emb_b))


def laplacian_kernel(distance: float, bandwidth: float) -> float:
    """
    Laplacian kernel: K = exp(-dist / bandwidth).

    Args:
        distance: L2 distance.
        bandwidth: kernel bandwidth parameter.

    Returns:
        Kernel value in [0, 1].
    """
    if bandwidth < 1e-8:
        return 1.0
    return float(np.exp(-distance / bandwidth))


def compute_bandwidth(embeddings: Dict[int, np.ndarray], k: int = 5) -> float:
    """
    Estimate kernel bandwidth using mean kNN distance.

    Args:
        embeddings: dict of episode embeddings.
        k: number of nearest neighbors.

    Returns:
        Estimated bandwidth (mean kNN distance).
    """
    indices = list(embeddings.keys())
    n = len(indices)
    if n < 2:
        return 1.0

    embs = np.array([embeddings[i] for i in indices])
    emb_dim = embs.shape[1]
    k_actual = min(k, n - 1)

    knn_dists = []
    for i in range(n):
        dists = np.linalg.norm(embs - embs[i], axis=1)
        dists[i] = np.inf
        knn_dists.append(np.sort(dists)[:k_actual].mean())

    bandwidth = float(np.mean(knn_dists))
    if bandwidth < 1e-8:
        bandwidth = 1.0

    print(f"  compute_bandwidth: n_episodes={n}, embedding_dim={emb_dim}, bandwidth={bandwidth:.4f}")

    return bandwidth


def compute_sic_total(
    selected_indices: List[int],
    embeddings: Dict[int, np.ndarray],
    bandwidth: float,
) -> float:
    """
    Compute total SIC score for a selected set.

    SIC = Σ_a σ(a) / (1 + σ(a)), where
    σ(a) = Σ_{c in selected} K(a, c)

    Args:
        selected_indices: list of selected episode indices.
        embeddings: episode embedding dict.
        bandwidth: kernel bandwidth.

    Returns:
        Total SIC score.
    """
    if len(selected_indices) < 1:
        return 0.0

    selected_embs = [embeddings[i] for i in selected_indices if i in embeddings]
    if not selected_embs:
        return 0.0

    n = len(selected_embs)
    sigma = np.zeros(n)

    for i in range(n):
        for j in range(n):
            dist = np.linalg.norm(selected_embs[i] - selected_embs[j])
            sigma[i] += laplacian_kernel(dist, bandwidth)

    saturation = sigma / (1.0 + sigma)
    return float(np.sum(saturation))


def compute_sic_gain(
    selected_indices: List[int],
    candidate_idx: int,
    embeddings: Dict[int, np.ndarray],
    bandwidth: float,
) -> float:
    """
    Compute marginal SIC gain of adding a candidate episode.

    gain = SIC(selected + candidate) - SIC(selected)

    Args:
        selected_indices: currently selected episode indices.
        candidate_idx: candidate episode index.
        embeddings: episode embedding dict.
        bandwidth: kernel bandwidth.

    Returns:
        Marginal SIC gain.
    """
    current_sic = compute_sic_total(selected_indices, embeddings, bandwidth)

    new_indices = selected_indices + [candidate_idx]
    new_sic = compute_sic_total(new_indices, embeddings, bandwidth)

    return new_sic - current_sic


def compute_redundancy_penalty(
    candidate_idx: int,
    selected_indices: List[int],
    embeddings: Dict[int, np.ndarray],
    k: int = 5,
) -> float:
    """
    Compute redundancy penalty using kNN distance to selected episodes.

    Lower kNN distance means higher redundancy.
    Penalty = 1 / (1 + mean_knn_distance)

    Args:
        candidate_idx: candidate episode index.
        selected_indices: currently selected episode indices.
        embeddings: episode embedding dict.
        k: number of nearest neighbors.

    Returns:
        Redundancy penalty in [0, 1].
    """
    if candidate_idx not in embeddings or not selected_indices:
        return 0.0

    candidate_emb = embeddings[candidate_idx]
    selected_embs = np.array([embeddings[i] for i in selected_indices if i in embeddings])

    if len(selected_embs) == 0:
        return 0.0

    dists = np.linalg.norm(selected_embs - candidate_emb, axis=1)
    k_actual = min(k, len(dists))
    mean_knn = np.sort(dists)[:k_actual].mean()

    penalty = 1.0 / (1.0 + mean_knn)
    return float(penalty)


def compute_candidate_score(
    selected_indices: List[int],
    candidate_idx: int,
    embeddings: Dict[int, np.ndarray],
    bandwidth: float,
    k: int = 5,
) -> float:
    """
    Compute comprehensive candidate score.

    score = marginal_gain / (1 + redundancy_penalty)

    Args:
        selected_indices: currently selected episode indices.
        candidate_idx: candidate episode index.
        embeddings: episode embedding dict.
        bandwidth: kernel bandwidth.
        k: kNN parameter for redundancy.

    Returns:
        Candidate score.
    """
    marginal_gain = compute_sic_gain(selected_indices, candidate_idx, embeddings, bandwidth)
    redundancy = compute_redundancy_penalty(candidate_idx, selected_indices, embeddings, k)

    score = marginal_gain / (1.0 + redundancy)
    return score