"""Selection strategies: K-Center, FPS, SIC-Noise."""

import numpy as np
from tqdm import tqdm


def k_center_select(embeddings_b0, embeddings_candidates, n_select):
    """
    K-Center greedy selection in PCA embedding space.
    Start from B0 set, greedily add candidates that maximize min distance to selected set.

    Args:
        embeddings_b0: np.ndarray (n_b0, dim) - B0 demo embeddings
        embeddings_candidates: np.ndarray (n_candidates, dim) - candidate embeddings
        n_select: int - number of candidates to select

    Returns:
        selected_indices: list of indices into embeddings_candidates
    """
    selected = list(range(len(embeddings_b0)))  # start with B0 indices (virtual)
    all_embeddings = np.vstack([embeddings_b0, embeddings_candidates])
    n_candidates = len(embeddings_candidates)

    selected_candidate_indices = []

    for _ in tqdm(range(n_select), desc="K-Center selection"):
        best_idx = -1
        best_min_dist = -1

        for c_idx in range(n_candidates):
            if c_idx in selected_candidate_indices:
                continue
            global_idx = len(embeddings_b0) + c_idx

            min_dist = np.inf
            for s_idx in selected:
                dist = np.linalg.norm(all_embeddings[global_idx] - all_embeddings[s_idx])
                if dist < min_dist:
                    min_dist = dist

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = c_idx

        if best_idx >= 0:
            selected_candidate_indices.append(best_idx)
            selected.append(len(embeddings_b0) + best_idx)

    return selected_candidate_indices


def fps_select(embeddings_b0, embeddings_candidates, n_select):
    """
    Farthest Point Sampling in PCA embedding space.
    Start from B0, add candidates by FPS.

    Args:
        embeddings_b0: np.ndarray (n_b0, dim)
        embeddings_candidates: np.ndarray (n_candidates, dim)
        n_select: int

    Returns:
        selected_indices: list of indices into embeddings_candidates
    """
    selected = list(range(len(embeddings_b0)))
    all_embeddings = np.vstack([embeddings_b0, embeddings_candidates])
    n_candidates = len(embeddings_candidates)

    selected_candidate_indices = []

    for _ in tqdm(range(n_select), desc="FPS selection"):
        best_idx = -1
        best_min_dist = -1

        for c_idx in range(n_candidates):
            if c_idx in selected_candidate_indices:
                continue
            global_idx = len(embeddings_b0) + c_idx

            min_dist = np.inf
            for s_idx in selected:
                dist = np.linalg.norm(all_embeddings[global_idx] - all_embeddings[s_idx])
                if dist < min_dist:
                    min_dist = dist

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = c_idx

        if best_idx >= 0:
            selected_candidate_indices.append(best_idx)
            selected.append(len(embeddings_b0) + best_idx)

    return selected_candidate_indices


def compute_sic_scores_b0(embeddings_b0, alpha=0.05):
    """
    Compute SIC-like anchor support scores for B0 demos.
    Lower score = more under-covered.

    For each demo, compute its average similarity to all other demos.
    Demos with low average similarity are under-covered.

    Args:
        embeddings_b0: np.ndarray (n_b0, dim)
        alpha: float - SIC alpha parameter

    Returns:
        scores: np.ndarray (n_b0,) - SIC support score for each demo
    """
    n = len(embeddings_b0)
    scores = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(embeddings_b0[i] - embeddings_b0[j])
                scores[i] += np.exp(-dist / (0.5 + 1e-8))

    return scores


def sic_noise_select(embeddings_b0, n_undercovered=10, n_noise_per=5):
    """
    SIC-Noise strategy: identify under-covered positions and generate noise-augmented variants.

    Args:
        embeddings_b0: np.ndarray (n_b0, dim)
        n_undercovered: number of most under-covered positions to augment
        n_noise_per: number of noise-augmented demos per under-covered position

    Returns:
        undercovered_indices: list of indices of most under-covered demos
    """
    scores = compute_sic_scores_b0(embeddings_b0, alpha=0.05)
    undercovered_indices = np.argsort(scores)[:n_undercovered].tolist()
    return undercovered_indices, scores