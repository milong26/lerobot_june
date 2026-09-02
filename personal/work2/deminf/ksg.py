"""
DemInf KSG Mutual Information Estimator

KSG-style per-sample information contribution estimator using kNN distances
in the joint latent space of state and action embeddings.

Mathematical formulation:
    For N samples (z_s_i, z_a_i):
    - d_s(i,j) = ||z_s_i - z_s_j||_2  (state latent Euclidean distance)
    - d_a(i,j) = ||z_a_i - z_a_j||_2  (action latent Euclidean distance)
    - d_joint(i,j) = max(d_s(i,j), d_a(i,j))  (max metric, per DemInf paper)

    For each sample i and each k in ks:
    - epsilon_i^k = k-th nearest neighbor distance in joint space (excluding self)
    - n_s(i,k) = #{j != i | d_s(i,j) < epsilon_i^k}
    - n_a(i,k) = #{j != i | d_a(i,j) < epsilon_i^k}

    Full KSG MI local contribution:
        i_hat_i(k) = psi(k) + psi(N) - psi(n_s(i,k) + 1) - psi(n_a(i,k) + 1)

    DemInf ranking mode (used for episode sorting):
        score_i(k) = -(psi(n_s(i,k) + 1) + psi(n_a(i,k) + 1))

    Final score: mean over all k in ks.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.special import digamma

import logging

logger = logging.getLogger("deminf")


def validate_ksg_inputs(
    z_s: np.ndarray,
    z_a: np.ndarray,
    ks: Tuple[int, ...],
) -> None:
    """
    Validate inputs for KSG estimation.

    Args:
        z_s: State latent embeddings [N, d_s].
        z_a: Action latent embeddings [N, d_a].
        ks: Tuple of k values.

    Raises:
        ValueError: If inputs are invalid.
    """
    if z_s.shape[0] != z_a.shape[0]:
        raise ValueError(
            f"z_s and z_a must have same number of samples: "
            f"z_s={z_s.shape[0]}, z_a={z_a.shape[0]}"
        )
    N = z_s.shape[0]
    max_k = max(ks)
    if N <= max_k + 1:
        raise ValueError(
            f"Number of samples N={N} must be greater than max(k)+1={max_k+1} "
            f"for KSG estimation with ks={ks}"
        )
    if not np.all(np.isfinite(z_s)):
        raise ValueError("z_s contains NaN or Inf")
    if not np.all(np.isfinite(z_a)):
        raise ValueError("z_a contains NaN or Inf")


def pairwise_l2_distances(
    query: torch.Tensor,
    reference: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """
    Compute pairwise L2 distances between query and reference points.

    Uses torch.cdist for efficient GPU computation with chunking to avoid OOM.

    Args:
        query: Query points [B, D].
        reference: Reference points [N, D].
        chunk_size: Chunk size for query points.

    Returns:
        Distance matrix [B, N].
    """
    B = query.shape[0]
    N = reference.shape[0]

    if B <= chunk_size:
        return torch.cdist(query, reference, p=2.0)

    # Chunked computation
    dists = []
    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        chunk_dists = torch.cdist(query[start:end], reference, p=2.0)
        dists.append(chunk_dists)
    return torch.cat(dists, dim=0)


def compute_knn_counts(
    z_s: torch.Tensor,
    z_a: torch.Tensor,
    ks: Tuple[int, ...],
    batch_size: int = 1024,
    mode: str = "deminf_rank",
) -> np.ndarray:
    """
    Compute KSG per-sample information contribution scores.

    Uses chunked pairwise distance computation on GPU for memory efficiency.

    Args:
        z_s: State latent embeddings [N, d_s].
        z_a: Action latent embeddings [N, d_a].
        ks: Tuple of k values for KSG estimator.
        batch_size: Query chunk size for distance computation.
        mode: 'deminf_rank' for ranking score, 'full_mi' for complete MI.

    Returns:
        Per-sample scores [N].
    """
    device = z_s.device
    N = z_s.shape[0]
    max_k = max(ks)

    # Auto-adjust batch_size if [B, N] would use too much memory
    # Estimate memory: 2 * B * N * 4 bytes (two distance matrices, float32)
    max_mem_bytes = 2 * 1024**3  # 2 GB limit
    bytes_per_query = 2 * N * 4  # two float32 values per reference point
    auto_batch = max(1, max_mem_bytes // max(bytes_per_query, 1))
    effective_batch = min(batch_size, auto_batch)

    if effective_batch < batch_size:
        logger.info(f"Auto-reduced KSG batch size from {batch_size} to {effective_batch} to avoid OOM")

    all_scores = np.zeros(N, dtype=np.float64)

    for k in ks:
        k_scores = np.zeros(N, dtype=np.float64)

        for start in range(0, N, effective_batch):
            end = min(start + effective_batch, N)
            B = end - start

            # Compute distances for this chunk
            query_s = z_s[start:end]  # [B, d_s]
            query_a = z_a[start:end]  # [B, d_a]

            dist_s = torch.cdist(query_s, z_s, p=2.0)  # [B, N]
            dist_a = torch.cdist(query_a, z_a, p=2.0)  # [B, N]

            # Joint distance: max metric
            dist_joint = torch.maximum(dist_s, dist_a)  # [B, N]

            # Set self-distances to +inf
            for b in range(B):
                global_idx = start + b
                dist_s[b, global_idx] = float("inf")
                dist_a[b, global_idx] = float("inf")
                dist_joint[b, global_idx] = float("inf")

            # Get k-th nearest neighbor distance in joint space
            # topk returns (values, indices), we only need values
            kth_values, _ = torch.topk(dist_joint, k=k, dim=1, largest=False)
            epsilon_k = kth_values[:, -1]  # [B], the k-th smallest distance

            # Add small epsilon to avoid exact equality issues
            epsilon_k = epsilon_k + 1e-10

            # Count marginal neighbors with STRICT < epsilon
            n_s = (dist_s < epsilon_k.unsqueeze(1)).sum(dim=1).float()  # [B]
            n_a = (dist_a < epsilon_k.unsqueeze(1)).sum(dim=1).float()  # [B]

            # Compute scores
            if mode == "deminf_rank":
                # DemInf ranking: score_i(k) = -(psi(n_s + 1) + psi(n_a + 1))
                # digamma argument must be >= 1
                n_s_clamped = torch.clamp(n_s, min=0.0)
                n_a_clamped = torch.clamp(n_a, min=0.0)
                score_chunk = -(
                    digamma(n_s_clamped.cpu().numpy() + 1.0)
                    + digamma(n_a_clamped.cpu().numpy() + 1.0)
                )
            else:
                # Full KSG MI: i_hat_i(k) = psi(k) + psi(N) - psi(n_s + 1) - psi(n_a + 1)
                n_s_clamped = torch.clamp(n_s, min=0.0)
                n_a_clamped = torch.clamp(n_a, min=0.0)
                score_chunk = (
                    digamma(np.array([k])) + digamma(np.array([N]))
                    - digamma(n_s_clamped.cpu().numpy() + 1.0)
                    - digamma(n_a_clamped.cpu().numpy() + 1.0)
                )

            k_scores[start:end] = score_chunk

        all_scores += k_scores

    # Average over all k values
    all_scores /= len(ks)

    return all_scores


def ksg_local_scores(
    z_s: np.ndarray,
    z_a: np.ndarray,
    ks: Tuple[int, ...] = (5, 6, 7),
    chunk_size: int = 1024,
    mode: str = "deminf_rank",
    backend: str = "chunked",
) -> np.ndarray:
    """
    Compute KSG local information contribution scores for all samples.

    Args:
        z_s: State latent embeddings [N, d_s].
        z_a: Action latent embeddings [N, d_a].
        ks: Tuple of k values. Default (5, 6, 7) per DemInf paper.
        chunk_size: Chunk size for distance computation.
        mode: 'deminf_rank' or 'full_mi'.
        backend: 'chunked' (memory-safe) or 'full' (for small data validation).

    Returns:
        Per-sample scores [N]. Higher score = stronger state-action dependency.
    """
    validate_ksg_inputs(z_s, z_a, ks)

    # Convert to torch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z_s_t = torch.from_numpy(z_s).float().to(device)
    z_a_t = torch.from_numpy(z_a).float().to(device)

    if backend == "full" and z_s.shape[0] <= 5000:
        # Full backend: compute all distances at once (for validation)
        N = z_s.shape[0]
        max_k = max(ks)

        dist_s = torch.cdist(z_s_t, z_s_t, p=2.0)  # [N, N]
        dist_a = torch.cdist(z_a_t, z_a_t, p=2.0)  # [N, N]
        dist_joint = torch.maximum(dist_s, dist_a)

        # Set self-distances to +inf
        dist_s.fill_diagonal_(float("inf"))
        dist_a.fill_diagonal_(float("inf"))
        dist_joint.fill_diagonal_(float("inf"))

        all_scores = np.zeros(N, dtype=np.float64)

        for k in ks:
            kth_values, _ = torch.topk(dist_joint, k=k, dim=1, largest=False)
            epsilon_k = kth_values[:, -1] + 1e-10  # [N]

            n_s = (dist_s < epsilon_k.unsqueeze(1)).sum(dim=1).float()
            n_a = (dist_a < epsilon_k.unsqueeze(1)).sum(dim=1).float()

            if mode == "deminf_rank":
                n_s_c = torch.clamp(n_s, min=0.0).cpu().numpy()
                n_a_c = torch.clamp(n_a, min=0.0).cpu().numpy()
                score_k = -(digamma(n_s_c + 1.0) + digamma(n_a_c + 1.0))
            else:
                n_s_c = torch.clamp(n_s, min=0.0).cpu().numpy()
                n_a_c = torch.clamp(n_a, min=0.0).cpu().numpy()
                score_k = (
                    digamma(np.full(N, k)) + digamma(np.full(N, N))
                    - digamma(n_s_c + 1.0) - digamma(n_a_c + 1.0)
                )

            all_scores += score_k

        all_scores /= len(ks)
        return all_scores
    else:
        # Chunked backend
        return compute_knn_counts(
            z_s_t, z_a_t, ks, batch_size=chunk_size, mode=mode
        )