"""SIC function, k-center, FPS implementations."""

import numpy as np
from tqdm import tqdm


def k_center_greedy(embeddings_selected, embeddings_candidates, n_select):
    """
    K-Center greedy selection.
    embeddings_selected: already selected embeddings (B0)
    embeddings_candidates: pool of candidates to select from
    Returns indices of selected candidates.
    """
    all_embeddings = np.vstack([embeddings_selected, embeddings_candidates])
    n_candidates = len(embeddings_candidates)
    n_selected_base = len(embeddings_selected)

    selected_indices = list(range(n_selected_base))
    chosen_candidate_indices = []

    for _ in tqdm(range(n_select), desc="K-Center"):
        best_idx = -1
        best_min_dist = -1

        for c in range(n_candidates):
            if c in chosen_candidate_indices:
                continue
            global_idx = n_selected_base + c

            min_dist = np.inf
            for s in selected_indices:
                d = np.linalg.norm(all_embeddings[global_idx] - all_embeddings[s])
                if d < min_dist:
                    min_dist = d

            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = c

        if best_idx >= 0:
            chosen_candidate_indices.append(best_idx)
            selected_indices.append(n_selected_base + best_idx)

    return chosen_candidate_indices


def fps_selection(embeddings_selected, embeddings_candidates, n_select):
    """
    Farthest Point Sampling.
    Same logic as k-center for this use case.
    """
    return k_center_greedy(embeddings_selected, embeddings_candidates, n_select)


def compute_sic_scores(embeddings, alpha=0.05):
    """
    Compute SIC support scores for each embedding.
    Lower score = more under-covered.
    """
    n = len(embeddings)
    scores = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                scores[i] += np.exp(-dist / (0.5 + 1e-8))

    return scores


def find_undercovered(embeddings, n_undercovered=10):
    """Find the most under-covered embeddings."""
    scores = compute_sic_scores(embeddings)
    undercovered = np.argsort(scores)[:n_undercovered].tolist()
    return undercovered, scores