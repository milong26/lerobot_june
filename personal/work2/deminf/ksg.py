"""
DemInf KSG Mutual Information Estimator

Official DemInf batch-local KSG estimator for ranking.

For a batch of B samples with latent embeddings (z_s, z_a):
    state_dist[i,j]  = ||z_s_i - z_s_j||_2
    action_dist[i,j] = ||z_a_i - z_a_j||_2
    joint_dist       = max(state_dist, action_dist)

    joint_knn_dists = sort(joint_dist, dim=-1)[:, ks]
    where ks = [5, 6, 7] (zero-based indices, includes self-distance=0 at index 0)

    For each k threshold epsilon = joint_knn_dists[:, k_idx]:
        obs_count  = sum(state_dist  < epsilon, axis=-1)
        action_count = sum(action_dist < epsilon, axis=-1)

    score_i = -mean_k [ digamma(obs_count_i_k) + digamma(action_count_i_k) ]

IMPORTANT:
- Self-distances are NOT set to inf; diagonal remains 0.
- No +1 is added to counts.
- No epsilon offset (no 1e-10).
- This is the ONLY formula used for selection_method="deminf".
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy.special import digamma

import logging

logger = logging.getLogger("deminf")


def deminf_ksg_batch_scores(
    z_state: torch.Tensor,
    z_action: torch.Tensor,
    ks: Tuple[int, ...] = (5, 6, 7),
) -> np.ndarray:
    """
    Compute official DemInf per-sample KSG scores for a single batch.

    This is the core estimator used in quality inference batches.

    Args:
        z_state: State latent embeddings [B, d_s].
        z_action: Action latent embeddings [B, d_a].
        ks: Tuple of k values. Default (5, 6, 7), zero-based indices.

    Returns:
        Per-sample scores [B]. Higher score = stronger state-action dependency.
    """
    B = z_state.shape[0]
    assert z_action.shape[0] == B, "z_state and z_action must have same batch size"

    state_dist = torch.cdist(z_state, z_state, p=2.0)   # [B, B]
    action_dist = torch.cdist(z_action, z_action, p=2.0)  # [B, B]
    joint_dist = torch.maximum(state_dist, action_dist)    # [B, B]

    sorted_joint = torch.sort(joint_dist, dim=-1).values  # [B, B]

    all_scores = np.zeros(B, dtype=np.float64)

    for k in ks:
        epsilon_k = sorted_joint[:, k]  # [B], zero-based index k

        obs_count = (state_dist < epsilon_k.unsqueeze(1)).sum(dim=-1).float()    # [B]
        action_count = (action_dist < epsilon_k.unsqueeze(1)).sum(dim=-1).float()  # [B]

        score_k = -(
            digamma(obs_count.cpu().numpy())
            + digamma(action_count.cpu().numpy())
        )
        all_scores += score_k

    all_scores /= len(ks)
    return all_scores


def official_ksg_reference_numpy(
    z_state: np.ndarray,
    z_action: np.ndarray,
    ks: Tuple[int, ...] = (5, 6, 7),
) -> np.ndarray:
    """
    Direct NumPy reference implementation of the official DemInf KSG estimator.

    Strictly follows the official pseudocode for validation.

    Args:
        z_state: State latent embeddings [B, d_s].
        z_action: Action latent embeddings [B, d_a].
        ks: Tuple of k values (zero-based indices).

    Returns:
        Per-sample scores [B].
    """
    B = z_state.shape[0]

    state_dist = np.linalg.norm(z_state[:, None, :] - z_state[None, :, :], axis=-1)
    action_dist = np.linalg.norm(z_action[:, None, :] - z_action[None, :, :], axis=-1)
    joint_dist = np.maximum(state_dist, action_dist)

    sorted_joint = np.sort(joint_dist, axis=-1)

    all_scores = np.zeros(B, dtype=np.float64)

    for k in ks:
        epsilon_k = sorted_joint[:, k]

        obs_count = np.sum(state_dist < epsilon_k[:, None], axis=-1).astype(np.float64)
        action_count = np.sum(action_dist < epsilon_k[:, None], axis=-1).astype(np.float64)

        score_k = -(digamma(obs_count) + digamma(action_count))
        all_scores += score_k

    all_scores /= len(ks)
    return all_scores


def validate_ksg_inputs(
    z_s: np.ndarray,
    z_a: np.ndarray,
    ks: Tuple[int, ...] = (5, 6, 7),
) -> None:
    """
    Validate inputs for KSG estimator.

    Args:
        z_s: State latent embeddings [N, d_s].
        z_a: Action latent embeddings [N, d_a].
        ks: Tuple of k values.

    Raises:
        ValueError: If inputs are invalid.
    """
    if z_s.shape[0] != z_a.shape[0]:
        raise ValueError(
            f"z_s and z_a must have the same number of samples, "
            f"got {z_s.shape[0]} and {z_a.shape[0]}"
        )

    N = z_s.shape[0]
    max_k = max(ks)
    min_samples = max_k + 2
    if N < min_samples:
        raise ValueError(
            f"Number of samples ({N}) must be greater than max(k)+1 ({min_samples}) "
            f"for ks={ks}"
        )

    if not np.all(np.isfinite(z_s)):
        raise ValueError("z_s contains NaN or Inf values")
    if not np.all(np.isfinite(z_a)):
        raise ValueError("z_a contains NaN or Inf values")


def ksg_local_scores(
    z_s: np.ndarray,
    z_a: np.ndarray,
    ks: Tuple[int, ...] = (5, 6, 7),
    chunk_size: int = 1024,
    mode: str = "deminf_rank",
    backend: str = "chunked",
) -> np.ndarray:
    """
    LEGACY / DIAGNOSTIC ONLY - NOT USED BY OFFICIAL QUALITY PIPELINE.

    This function processes the full dataset in chunks of size `chunk_size`,
    calling deminf_ksg_batch_scores for each chunk. It does NOT replicate
    the official repeat/shuffle batch-local KSG scoring, because simply
    chunking the full latent array sequentially is not equivalent to the
    official quality inference pipeline (repeat=4 shuffled 1024-sized batches
    with drop_remainder).

    The official quality pipeline uses:
        score_latents -> build_official_quality_batches -> score_quality_batches
        -> deminf_ksg_batch_scores

    Args:
        z_s: State latent embeddings [N, d_s].
        z_a: Action latent embeddings [N, d_a].
        ks: Tuple of k values. Default (5, 6, 7).
        chunk_size: Batch size for KSG distance computation.
        mode: 'deminf_rank' or 'full_mi'. Only 'deminf_rank' is official.
        backend: 'chunked' or 'full'. For small data validation only.

    Returns:
        Per-sample scores [N].
    """
    N = z_s.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z_s_t = torch.from_numpy(z_s).float().to(device)
    z_a_t = torch.from_numpy(z_a).float().to(device)

    if backend == "full" and N <= 5000:
        scores = deminf_ksg_batch_scores(z_s_t, z_a_t, ks)
    else:
        all_scores = np.zeros(N, dtype=np.float64)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            batch_s = z_s_t[start:end]
            batch_a = z_a_t[start:end]
            batch_scores = deminf_ksg_batch_scores(batch_s, batch_a, ks)
            all_scores[start:end] = batch_scores
        scores = all_scores

    return scores