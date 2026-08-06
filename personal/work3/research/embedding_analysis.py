"""SIC Framework - Embedding space analysis."""

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def compute_cross_config_distance_curve(
    wrist_embs: dict,
    config_map: dict,
    n_time_points: int = 50
) -> np.ndarray:
    """Compute mean pairwise L2 distance between embeddings of different configs."""
    distances = np.zeros(n_time_points)
    count = 0

    ep_indices = list(config_map.keys())

    for i in range(len(ep_indices)):
        for j in range(i+1, len(ep_indices)):
            ep_i, ep_j = ep_indices[i], ep_indices[j]
            if config_map[ep_i] == config_map[ep_j]:
                continue

            emb_i = wrist_embs[ep_i]
            emb_j = wrist_embs[ep_j]

            for t in range(n_time_points):
                ti = int(t / n_time_points * (len(emb_i) - 1))
                tj = int(t / n_time_points * (len(emb_j) - 1))
                distances[t] += np.linalg.norm(emb_i[ti] - emb_j[tj])
            count += 1

    return distances / (count + 1e-8)


def compute_pca_variance_curve(embeddings_dict: dict, max_components: int = 128) -> dict:
    """Compute cumulative explained variance ratio vs number of PCA components."""
    all_embs = np.vstack(list(embeddings_dict.values()))
    pca = PCA(n_components=min(max_components, all_embs.shape[1]))
    pca.fit(all_embs)

    return {
        'n_components': list(range(1, len(pca.explained_variance_ratio_) + 1)),
        'cumulative_variance': list(np.cumsum(pca.explained_variance_ratio_)),
        'per_component_variance': list(pca.explained_variance_ratio_)
    }


def compute_tsne_for_configs(
    wrist_embs: dict,
    config_map: dict,
    selected_configs: list = None,
    pca_dim: int = 50,
    tsne_perplexity: int = 30
) -> dict:
    """Compute t-SNE embedding of trajectory frames for selected configs."""
    if selected_configs is None:
        import random
        all_cfgs = list(set(config_map.values()))
        selected_configs = random.sample(all_cfgs, min(4, len(all_cfgs)))

    all_embs = []
    labels = []
    frame_positions = []

    for ep_idx, cfg in config_map.items():
        if cfg not in selected_configs:
            continue
        embs = wrist_embs[ep_idx]
        all_embs.append(embs)
        labels.extend([cfg] * len(embs))
        frame_positions.extend(np.linspace(0, 1, len(embs)).tolist())

    all_embs = np.vstack(all_embs)

    pca = PCA(n_components=min(pca_dim, all_embs.shape[1]))
    reduced = pca.fit_transform(all_embs)

    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=42)
    tsne_xy = tsne.fit_transform(reduced)

    return {
        'tsne_xy': tsne_xy,
        'config_labels': labels,
        'frame_positions': frame_positions,
        'selected_configs': selected_configs
    }