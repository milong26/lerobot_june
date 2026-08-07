"""Build anchor reference and compute SIC scores."""

import os
import pickle
import numpy as np
from sklearn.decomposition import IncrementalPCA


def fit_pca(embeddings_dict, d_pca=32):
    all_embs = np.vstack(list(embeddings_dict.values()))
    mu = np.mean(all_embs, axis=0)
    pca = IncrementalPCA(n_components=d_pca)
    pca.fit(all_embs - mu)
    return pca, mu


def transform_embedding(embedding, pca, mu):
    return pca.transform((embedding - mu).reshape(1, -1)).flatten()


def build_anchor_reference(embeddings_dict, d_pca=32, save_path=None):
    pca, mu = fit_pca(embeddings_dict, d_pca)

    anchors = {}
    for ep_idx, emb in embeddings_dict.items():
        mean_emb = emb.mean(axis=0)
        anchors[ep_idx] = {
            'phi': transform_embedding(mean_emb, pca, mu),
            'mean_emb': mean_emb,
        }

    keys = list(anchors.keys())
    nn_dists = []
    for i, k in enumerate(keys):
        phi_i = anchors[k]['phi']
        dists = [np.linalg.norm(phi_i - anchors[k2]['phi'])
                 for j, k2 in enumerate(keys) if j != i]
        nn_dists.append(min(dists))
    d_bar = np.mean(nn_dists)

    R = {
        'anchors': anchors,
        'd_bar': d_bar,
        'pca': pca,
        'mu': mu,
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(R, f)

    return R


def laplacian_kernel(phi_a, phi_pr, d_bar):
    return np.exp(-np.linalg.norm(phi_a - phi_pr) / (d_bar + 1e-8))


def tau_cumsum(T_pr, alpha):
    return alpha * np.log(T_pr + 1)


def saturation(sigma):
    return sigma / (1.0 + sigma)


def compute_anchor_support(anchor_key, collection_plan, anchors, d_bar, alpha):
    phi_a = anchors[anchor_key]['phi']
    sigma = 0.0

    for ep_idx, n_times in collection_plan.items():
        if ep_idx not in anchors:
            continue
        phi_pr = anchors[ep_idx]['phi']
        weight = tau_cumsum(n_times, alpha)
        sigma += weight * laplacian_kernel(phi_a, phi_pr, d_bar)

    return sigma


def compute_sic_score(collection_plan, anchor_ref, alpha=0.05):
    anchors = anchor_ref['anchors']
    d_bar = anchor_ref['d_bar']
    total_sic = 0.0
    anchor_supports = {}

    for anchor_key in anchors:
        support = compute_anchor_support(anchor_key, collection_plan, anchors, d_bar, alpha)
        total_sic += saturation(support)
        anchor_supports[anchor_key] = support

    return {
        'sic': total_sic,
        'anchor_supports': anchor_supports,
    }