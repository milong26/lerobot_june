"""SIC Framework - Anchor reference system construction."""

import os
import numpy as np
from sklearn.decomposition import IncrementalPCA
import pickle


def fit_pca_both_views(global_embeddings_dict: dict, wrist_embeddings_dict: dict,
                        d_pca: int = 32) -> dict:
    """Fit independent PCA for global and wrist views."""
    def fit_single(emb_dict):
        pca = IncrementalPCA(n_components=d_pca)
        mu = np.mean(np.vstack(list(emb_dict.values())), axis=0)
        for emb in emb_dict.values():
            pca.partial_fit(emb - mu)
        return pca, mu

    pca_g, mu_g = fit_single(global_embeddings_dict)
    pca_w, mu_w = fit_single(wrist_embeddings_dict)
    return {'global': pca_g, 'wrist': pca_w, 'mu_global': mu_g, 'mu_wrist': mu_w}


def transform_embedding(embedding: np.ndarray, pca, mu: np.ndarray) -> np.ndarray:
    """Apply PCA transform: z = W^T (f - mu)"""
    return pca.transform((embedding - mu).reshape(1, -1)).flatten()


def compute_anchor_embeddings(
    global_embs: dict, wrist_embs: dict,
    config_map: dict,
    pca_dict: dict
) -> dict:
    """Build anchor embeddings for each (pos_id, rot_id) configuration."""
    anchors = {}
    for ep_idx, (pos_id, rot_id) in config_map.items():
        key = (pos_id, rot_id)

        g_emb = global_embs[ep_idx]
        phi_g = transform_embedding(g_emb[0], pca_dict['global'], pca_dict['mu_global'])

        w_emb = wrist_embs[ep_idx]
        N = len(w_emb)
        mid_start, mid_end = int(0.2 * N), int(0.8 * N)
        mid_embs = w_emb[mid_start:mid_end]
        phi_w_raw = mid_embs.mean(axis=0)
        phi_w = transform_embedding(phi_w_raw, pca_dict['wrist'], pca_dict['mu_wrist'])

        anchors[key] = {'phi_global': phi_g, 'phi_wrist': phi_w}

    return anchors


def compute_distance_scales(anchors: dict) -> dict:
    """Compute per-view distance scales as mean nearest-neighbor distance."""
    keys = list(anchors.keys())

    def mean_nn_dist(view):
        nn_dists = []
        for i, k in enumerate(keys):
            phi_i = anchors[k][f'phi_{view}']
            dists = [np.linalg.norm(phi_i - anchors[k2][f'phi_{view}'])
                     for j, k2 in enumerate(keys) if j != i]
            nn_dists.append(min(dists))
        return np.mean(nn_dists)

    return {
        'd_bar_global': mean_nn_dist('global'),
        'd_bar_wrist': mean_nn_dist('wrist')
    }


def build_anchor_reference(
    global_embs: dict, wrist_embs: dict,
    config_map: dict,
    d_pca: int = 32,
    save_path: str = None
) -> dict:
    """Build complete anchor reference system R."""
    pca_dict = fit_pca_both_views(global_embs, wrist_embs, d_pca)
    anchors = compute_anchor_embeddings(global_embs, wrist_embs, config_map, pca_dict)
    dist_scales = compute_distance_scales(anchors)

    R = {
        'anchors': anchors,
        'd_bar_global': dist_scales['d_bar_global'],
        'd_bar_wrist': dist_scales['d_bar_wrist'],
        'pca_dict': pca_dict
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(R, f)
        print(f"Anchor reference saved to {save_path}")

    return R