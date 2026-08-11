#!/usr/bin/env python
"""
Adaptive Temporal Embedding Search + Episode Embedding Evaluation

Automatically searches for the most informative temporal windows in episodes
and evaluates different temporal pooling strategies.

Usage:
    python evaluate_episode_embeddings.py
"""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR
CACHE_DIR = BASE_DIR / "cache"
ALL_FRAMES_DIR = BASE_DIR.parent / "all_frames"
POOL_DIR = BASE_DIR.parent / "pool"
DATASET_DIR = BASE_DIR.parent.parent / "dataset"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SearchConfig:
    pca_dim: int = 32
    random_seed: int = 42
    test_size: float = 0.2
    min_window: float = 0.10
    max_window: float = 1.00
    window_step: float = 0.05
    max_pairs: int = 100000
    distance_matrix_episodes: int = 50
    scatter_pairs: int = 500
    n_clusters: int = 10


def normalize_episode_time(frame_indices: np.ndarray, total_frames: int) -> np.ndarray:
    """Normalize frame indices to [0, 1] range."""
    if total_frames <= 1:
        return np.zeros_like(frame_indices, dtype=float)
    return frame_indices / (total_frames - 1)


def generate_temporal_windows(
    min_window: float = 0.10,
    max_window: float = 1.00,
    window_step: float = 0.05,
    stride: float = 0.05,
) -> List[Tuple[float, float]]:
    """Generate candidate temporal windows via sliding window search."""
    windows = []
    window_lengths = np.arange(min_window, max_window + window_step, window_step)
    for wl in window_lengths:
        wl = round(wl, 4)
        if wl > 1.0:
            continue
        start = 0.0
        while start + wl <= 1.0 + 1e-9:
            start = round(start, 4)
            end = round(min(start + wl, 1.0), 4)
            if end > start:
                windows.append((start, end))
            start += stride
    return windows


def temporal_mean_pool(
    frame_embeddings: np.ndarray,
    start_pct: float = 0.0,
    end_pct: float = 1.0,
) -> np.ndarray:
    """Uniform mean pooling over a temporal window."""
    K = frame_embeddings.shape[0]
    s = max(0, int(start_pct * K))
    e = min(K, int(end_pct * K))
    if e <= s:
        s = 0
        e = K
    return frame_embeddings[s:e].mean(axis=0)


def _generate_weights(
    n: int,
    weighting: str = "uniform",
    center: float = 0.5,
    sigma: float = 0.2,
) -> np.ndarray:
    """Generate temporal weights for a window."""
    t = np.linspace(0, 1, n)
    if weighting == "uniform":
        return np.ones(n)
    elif weighting == "gaussian":
        return np.exp(-((t - center) ** 2) / (2 * sigma ** 2))
    elif weighting == "triangular":
        return 1.0 - np.abs(t - center) / max(center, 1 - center)
    elif weighting == "cosine":
        return 0.5 * (1 + np.cos(np.pi * (t - center) / max(center, 1 - center)))
    else:
        return np.ones(n)


def temporal_weighted_pool(
    frame_embeddings: np.ndarray,
    start_pct: float = 0.0,
    end_pct: float = 1.0,
    weighting: str = "gaussian",
    center: float = 0.5,
    sigma: float = 0.2,
) -> np.ndarray:
    """Weighted mean pooling over a temporal window."""
    K = frame_embeddings.shape[0]
    s = max(0, int(start_pct * K))
    e = min(K, int(end_pct * K))
    if e <= s:
        s = 0
        e = K
    frames = frame_embeddings[s:e]
    n = len(frames)
    weights = _generate_weights(n, weighting, center, sigma)
    weights = weights / (weights.sum() + 1e-12)
    return (weights[:, None] * frames).sum(axis=0)


def temporal_change_score_pool(
    frame_embeddings: np.ndarray,
    start_pct: float = 0.0,
    end_pct: float = 1.0,
    smoothing_window: int = 5,
    percentile_clip: float = 95,
) -> np.ndarray:
    """Pooling based on temporal change scores (delta_t)."""
    K = frame_embeddings.shape[0]
    s = max(0, int(start_pct * K))
    e = min(K, int(end_pct * K))
    if e <= s:
        s = 0
        e = K
    frames = frame_embeddings[s:e]
    n = len(frames)
    if n <= 1:
        return frames.mean(axis=0)

    deltas = np.linalg.norm(frames[1:] - frames[:-1], axis=1)
    deltas = np.concatenate([[deltas[0]], deltas])

    if smoothing_window > 1 and n > smoothing_window:
        kernel = np.ones(smoothing_window) / smoothing_window
        deltas = np.convolve(deltas, kernel, mode="same")

    threshold = np.percentile(deltas, percentile_clip)
    weights = np.clip(deltas, 0, threshold)
    weights = weights / (weights.sum() + 1e-12)
    return (weights[:, None] * frames).sum(axis=0)


def _make_candidate_key(
    method: str,
    start: float,
    end: float,
    weighting: str = "uniform",
    center: float = 0.5,
    sigma: float = 0.2,
) -> str:
    """Create a unique cache key for a candidate."""
    raw = f"{method}_{start:.4f}_{end:.4f}_{weighting}_{center:.4f}_{sigma:.4f}"
    return hashlib.md5(raw.encode()).hexdigest()


def _load_cached_embedding(key: str) -> Optional[np.ndarray]:
    """Load cached episode embeddings."""
    path = CACHE_DIR / f"{key}.npy"
    if path.exists():
        return np.load(path)
    return None


def _save_cached_embedding(key: str, embeddings: np.ndarray):
    """Save episode embeddings to cache."""
    path = CACHE_DIR / f"{key}.npy"
    np.save(path, embeddings)


def _load_cached_metrics(key: str) -> Optional[dict]:
    """Load cached evaluation metrics."""
    path = CACHE_DIR / f"{key}_metrics.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _save_cached_metrics(key: str, metrics: dict):
    """Save evaluation metrics to cache."""
    path = CACHE_DIR / f"{key}_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f)


def _extract_episode_embeddings_raw(
    all_frame_embeddings: List[np.ndarray],
    method: str,
    start_pct: float = 0.0,
    end_pct: float = 1.0,
    weighting: str = "uniform",
    center: float = 0.5,
    sigma: float = 0.2,
) -> np.ndarray:
    """Extract raw episode embeddings without PCA."""
    episode_embs = []
    for frames in all_frame_embeddings:
        frames = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
        if method == "uniform_full":
            emb = frames.mean(axis=0)
        elif method == "temporal_window":
            emb = temporal_mean_pool(frames, start_pct, end_pct)
        elif method == "temporal_weighted_window":
            emb = temporal_weighted_pool(frames, start_pct, end_pct, weighting, center, sigma)
        elif method == "temporal_attention_like":
            emb = temporal_change_score_pool(frames, start_pct, end_pct)
        elif method == "temporal_multi_window":
            mid = (start_pct + end_pct) / 2
            half_w = (end_pct - start_pct) / 4
            emb1 = temporal_mean_pool(frames, mid - half_w, mid)
            emb2 = temporal_mean_pool(frames, mid, mid + half_w)
            emb = 0.5 * emb1 + 0.5 * emb2
        else:
            emb = frames.mean(axis=0)
        episode_embs.append(emb)

    episode_embs = np.array(episode_embs)
    episode_embs = np.nan_to_num(episode_embs, nan=0.0, posinf=0.0, neginf=0.0)
    return episode_embs


def extract_episode_embeddings(
    all_frame_embeddings: List[np.ndarray],
    method: str,
    start_pct: float = 0.0,
    end_pct: float = 1.0,
    weighting: str = "uniform",
    center: float = 0.5,
    sigma: float = 0.2,
    pca_dim: int = 32,
    pca_model=None,
    use_cache: bool = True,
) -> Tuple[np.ndarray, PCA]:
    """
    Extract episode-level embeddings from frame-level embeddings.

    Returns:
        (N, pca_dim) array, PCA model
    """
    episode_embs = _extract_episode_embeddings_raw(
        all_frame_embeddings, method, start_pct, end_pct, weighting, center, sigma
    )

    if pca_model is not None:
        return pca_model.transform(episode_embs), pca_model

    n_samples = episode_embs.shape[0]
    actual_dim = min(pca_dim, n_samples, episode_embs.shape[1])
    if actual_dim < 1:
        actual_dim = 1

    pca = PCA(n_components=actual_dim, random_state=RANDOM_SEED)
    episode_embs_pca = pca.fit_transform(episode_embs)

    return episode_embs_pca, pca


def compute_position_correlation(
    episode_embs: np.ndarray,
    obj_positions: np.ndarray,
    max_pairs: int = 100000,
) -> float:
    """
    Compute Spearman correlation between embedding distances and physical distances.
    """
    N = len(episode_embs)
    N_obj = len(obj_positions)
    if N != N_obj:
        raise ValueError(f"Shape mismatch: episode_embs has {N} entries but obj_positions has {N_obj}")
    if N < 2:
        return 0.0

    obj_positions = np.nan_to_num(obj_positions, nan=0.0)

    if N <= 500:
        emb_dists = cdist(episode_embs, episode_embs, metric="euclidean")
        phys_dists = cdist(obj_positions, obj_positions, metric="euclidean")
        rows, cols = np.triu_indices(N, k=1)
        assert rows.max() < N, f"Row index {rows.max()} >= N={N}"
        assert cols.max() < N, f"Col index {cols.max()} >= N={N}"
        emb_dists_flat = emb_dists[rows, cols]
        phys_dists_flat = phys_dists[rows, cols]
    else:
        n_pairs = min(max_pairs, N * (N - 1) // 2)
        rng = np.random.RandomState(RANDOM_SEED)
        i_idx = rng.randint(0, N, size=n_pairs)
        j_idx = rng.randint(0, N, size=n_pairs)
        while True:
            dup = i_idx == j_idx
            if not dup.any():
                break
            j_idx[dup] = rng.randint(0, N, size=dup.sum())

        emb_dists_flat = np.linalg.norm(episode_embs[i_idx] - episode_embs[j_idx], axis=1)
        phys_dists_flat = np.linalg.norm(obj_positions[i_idx] - obj_positions[j_idx], axis=1)

    valid = phys_dists_flat > 1e-10
    if valid.sum() < 3:
        return 0.0

    corr, _ = spearmanr(emb_dists_flat[valid], phys_dists_flat[valid])
    if np.isnan(corr):
        return 0.0
    return float(corr)


def compute_d_bar_ratio(
    episode_embs: np.ndarray,
    max_pairs: int = 100000,
) -> float:
    """
    Compute d_bar_ratio = mean_nearest_neighbor_distance / (0.1 * embedding_space_range).
    """
    N = len(episode_embs)
    if N < 2:
        return 0.0

    if N <= 500:
        dists = cdist(episode_embs, episode_embs, metric="euclidean")
        np.fill_diagonal(dists, np.inf)
        nn_dists = dists.min(axis=1)
        max_dist = dists[np.triu_indices(N, k=1)].max()
    else:
        n_pairs = min(max_pairs, N * (N - 1) // 2)
        rng = np.random.RandomState(RANDOM_SEED)
        i_idx = rng.randint(0, N, size=n_pairs)
        j_idx = rng.randint(0, N, size=n_pairs)
        while True:
            dup = i_idx == j_idx
            if not dup.any():
                break
            j_idx[dup] = rng.randint(0, N, size=dup.sum())

        pairwise = np.linalg.norm(episode_embs[i_idx] - episode_embs[j_idx], axis=1)
        max_dist = pairwise.max()

        nn_dists = np.full(N, np.inf)
        for i in range(N):
            mask_i = i_idx == i
            mask_j = j_idx == i
            d_i = pairwise[mask_i]
            d_j = pairwise[mask_j]
            all_d = np.concatenate([d_i, d_j])
            if len(all_d) > 0:
                nn_dists[i] = all_d.min()

        nn_dists = np.where(np.isinf(nn_dists), 0.0, nn_dists)

    mean_nn = nn_dists.mean()
    space_range = max_dist if max_dist > 1e-10 else 1.0

    ratio = mean_nn / (0.1 * space_range)
    ratio = min(max(ratio, 0.0), 1.0)
    return float(ratio)


def compute_cluster_separation(
    episode_embs: np.ndarray,
    n_clusters: int = 10,
) -> float:
    """
    Compute cluster separation = mean_inter_cluster_dist / mean_intra_cluster_dist.
    """
    N = len(episode_embs)
    actual_k = min(n_clusters, N)
    if actual_k < 2:
        return 0.0

    try:
        kmeans = KMeans(n_clusters=actual_k, random_state=RANDOM_SEED, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(episode_embs)
    except Exception:
        return 0.0

    centers = kmeans.cluster_centers_
    if actual_k >= 2:
        inter_dists = cdist(centers, centers, metric="euclidean")
        np.fill_diagonal(inter_dists, np.inf)
        mean_inter = inter_dists.min(axis=1).mean()
    else:
        mean_inter = 0.0

    intra_dists = []
    for c in range(actual_k):
        mask = labels == c
        if mask.sum() < 2:
            continue
        cluster_pts = episode_embs[mask]
        c_dists = cdist(cluster_pts, cluster_pts, metric="euclidean")
        intra_dists.append(c_dists[np.triu_indices(len(cluster_pts), k=1)].mean())

    if len(intra_dists) == 0:
        return 0.0

    mean_intra = np.mean(intra_dists)
    if mean_intra < 1e-10:
        return 0.0

    return float(mean_inter / mean_intra)


def evaluate_episode_embedding(
    method_name: str,
    episode_embs: np.ndarray,
    obj_positions: np.ndarray,
    config: SearchConfig,
) -> dict:
    """
    Evaluate episode embeddings with three metrics.
    """
    episode_embs = np.nan_to_num(episode_embs, nan=0.0, posinf=0.0, neginf=0.0)
    assert len(episode_embs) == len(obj_positions), \
        f"Shape mismatch in evaluate: embs={len(episode_embs)}, obj={len(obj_positions)}"

    pos_corr = compute_position_correlation(
        episode_embs, obj_positions, config.max_pairs
    )
    d_bar = compute_d_bar_ratio(episode_embs, config.max_pairs)
    cluster_sep = compute_cluster_separation(episode_embs, config.n_clusters)

    return {
        "method_name": method_name,
        "position_correlation": pos_corr,
        "d_bar_ratio": d_bar,
        "cluster_separation": cluster_sep,
    }


def _normalize_scores(scores: List[float]) -> List[float]:
    """Min-max normalization of scores."""
    if len(scores) == 0:
        return []
    mn = min(scores)
    mx = max(scores)
    if mx - mn < 1e-10:
        return [0.5] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


def compute_overall_score(
    pos_corr: float,
    d_bar: float,
    cluster_sep: float,
) -> float:
    """Compute weighted overall score."""
    return 0.60 * pos_corr + 0.20 * d_bar + 0.20 * cluster_sep


def search_best_temporal_strategy(
    all_frame_embeddings: List[np.ndarray],
    obj_positions: np.ndarray,
    config: SearchConfig,
    search_indices: List[int],
    eval_indices: List[int],
    force_recompute: bool = False,
) -> dict:
    """
    Search for the best temporal pooling strategy.
    """
    # Check if complete search results already exist
    results_file = OUTPUT_DIR / "temporal_search_results.csv"
    if not force_recompute and results_file.exists():
        print(f"\nFound existing search results: {results_file}")
        print("Loading cached results instead of re-computing...")
        df = pd.read_csv(results_file)
        results = df.to_dict("records")
        print(f"Loaded {len(results)} candidates from cache")
        return results

    search_frames = [all_frame_embeddings[i] for i in search_indices]
    search_obj = obj_positions[search_indices]
    eval_frames = [all_frame_embeddings[i] for i in eval_indices]
    eval_obj = obj_positions[eval_indices]

    candidates = []

    # 1. uniform_full baseline
    candidates.append({
        "method": "uniform_full",
        "start_pct": 0.0,
        "end_pct": 1.0,
        "weighting": "uniform",
        "center": 0.5,
        "sigma": 0.2,
    })

    # 2. temporal_window candidates (sliding window)
    windows = generate_temporal_windows(
        config.min_window, config.max_window, config.window_step, config.window_step
    )
    for start, end in windows:
        candidates.append({
            "method": "temporal_window",
            "start_pct": start,
            "end_pct": end,
            "weighting": "uniform",
            "center": 0.5,
            "sigma": 0.2,
        })

    # 3. temporal_weighted_window candidates
    weightings = ["gaussian", "triangular", "cosine"]
    for start, end in windows:
        center = (start + end) / 2
        sigma = (end - start) / 4
        for w in weightings:
            candidates.append({
                "method": "temporal_weighted_window",
                "start_pct": start,
                "end_pct": end,
                "weighting": w,
                "center": center,
                "sigma": sigma,
            })

    # 4. temporal_attention_like candidates
    for start, end in windows:
        candidates.append({
            "method": "temporal_attention_like",
            "start_pct": start,
            "end_pct": end,
            "weighting": "uniform",
            "center": 0.5,
            "sigma": 0.2,
        })

    # 5. temporal_multi_window candidates
    for start, end in windows:
        candidates.append({
            "method": "temporal_multi_window",
            "start_pct": start,
            "end_pct": end,
            "weighting": "uniform",
            "center": 0.5,
            "sigma": 0.2,
        })

    print(f"\nTotal candidates to evaluate: {len(candidates)}")

    results = []
    for i, cand in enumerate(tqdm(candidates, desc="Evaluating candidates")):
        key = _make_candidate_key(
            cand["method"], cand["start_pct"], cand["end_pct"],
            cand["weighting"], cand["center"], cand["sigma"]
        )

        cached_metrics = _load_cached_metrics(key)
        if cached_metrics is not None:
            results.append(cached_metrics)
            continue

        try:
            search_embs, pca_model = extract_episode_embeddings(
                search_frames,
                cand["method"],
                cand["start_pct"],
                cand["end_pct"],
                cand["weighting"],
                cand["center"],
                cand["sigma"],
                config.pca_dim,
            )

            eval_embs, _ = extract_episode_embeddings(
                eval_frames,
                cand["method"],
                cand["start_pct"],
                cand["end_pct"],
                cand["weighting"],
                cand["center"],
                cand["sigma"],
                config.pca_dim,
                pca_model=pca_model,
            )

            eval_metrics = evaluate_episode_embedding(
                cand["method"],
                eval_embs,
                eval_obj,
                config,
            )

            eval_metrics["overall_score"] = compute_overall_score(
                eval_metrics["position_correlation"],
                eval_metrics["d_bar_ratio"],
                eval_metrics["cluster_separation"],
            )
            eval_metrics["start_pct"] = cand["start_pct"]
            eval_metrics["end_pct"] = cand["end_pct"]
            eval_metrics["window_length_pct"] = cand["end_pct"] - cand["start_pct"]
            eval_metrics["weighting"] = cand["weighting"]
            eval_metrics["cache_key"] = key

            _save_cached_metrics(key, eval_metrics)
            results.append(eval_metrics)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"\n[ERROR] Candidate {i}: {cand['method']} ({cand['start_pct']}-{cand['end_pct']})")
            print(f"  Error: {e}")
            if i < 3:
                print(tb)
            results.append({
                "method_name": cand["method"],
                "position_correlation": 0.0,
                "d_bar_ratio": 0.0,
                "cluster_separation": 0.0,
                "overall_score": 0.0,
                "start_pct": cand["start_pct"],
                "end_pct": cand["end_pct"],
                "window_length_pct": cand["end_pct"] - cand["start_pct"],
                "weighting": cand["weighting"],
                "error": str(e),
                "cache_key": key,
            })

    results.sort(key=lambda x: x["position_correlation"], reverse=True)
    return results


def plot_embedding_comparison(
    all_frame_embeddings: List[np.ndarray],
    obj_positions: np.ndarray,
    top_candidates: List[dict],
    config: SearchConfig,
):
    """Generate embedding comparison visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_methods = min(5, len(top_candidates))
    methods_to_plot = top_candidates[:n_methods]

    eval_indices = list(range(len(all_frame_embeddings)))
    eval_frames = [all_frame_embeddings[i] for i in eval_indices]

    all_embs_2d = []
    for cand in methods_to_plot:
        method = cand.get("method", cand.get("method_name", "uniform_full"))
        embs, _ = extract_episode_embeddings(
            eval_frames,
            method,
            cand["start_pct"],
            cand["end_pct"],
            cand.get("weighting", "uniform"),
            cand.get("center", 0.5),
            cand.get("sigma", 0.2),
            2,
        )
        all_embs_2d.append(embs)

    n_samples = min(50, len(eval_indices))
    rng = np.random.RandomState(RANDOM_SEED)
    sample_idx = rng.choice(len(eval_indices), size=n_samples, replace=False)

    fig, axes = plt.subplots(3, n_methods, figsize=(5 * n_methods, 15))
    if n_methods == 1:
        axes = axes.reshape(3, 1)

    obj_x = obj_positions[:, 0]
    cmap = plt.cm.viridis

    for j, cand in enumerate(methods_to_plot):
        embs_2d = all_embs_2d[j]
        method = cand.get("method", cand.get("method_name", "uniform_full"))
        method_title = f"{method}\n{cand['start_pct']*100:.0f}%-{cand['end_pct']*100:.0f}%"

        # Row 1: PCA 2D scatter
        ax = axes[0, j]
        colors = cmap((obj_x - obj_x.min()) / (obj_x.max() - obj_x.min() + 1e-10))
        ax.scatter(embs_2d[:, 0], embs_2d[:, 1], c=colors, s=10, alpha=0.7)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(method_title, fontsize=9)
        ax.set_aspect("equal", adjustable="box")

        # Row 2: Distance matrix heatmap
        ax = axes[1, j]
        sub_embs = embs_2d[sample_idx]
        dist_matrix = cdist(sub_embs, sub_embs, metric="euclidean")
        im = ax.imshow(dist_matrix, cmap="viridis", aspect="auto")
        ax.set_xlabel("Episode Index")
        ax.set_ylabel("Episode Index")
        ax.set_title(f"Distance Matrix\n{method_title}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Row 3: Physical vs embedding distance scatter
        ax = axes[2, j]
        N = len(embs_2d)
        n_pairs = min(config.scatter_pairs, N * (N - 1) // 2)
        rng2 = np.random.RandomState(RANDOM_SEED)
        i_idx = rng2.randint(0, N, size=n_pairs)
        j_idx = rng2.randint(0, N, size=n_pairs)
        while True:
            dup = i_idx == j_idx
            if not dup.any():
                break
            j_idx[dup] = rng2.randint(0, N, size=dup.sum())

        emb_dists = np.linalg.norm(embs_2d[i_idx] - embs_2d[j_idx], axis=1)
        phys_dists = np.linalg.norm(obj_positions[i_idx] - obj_positions[j_idx], axis=1)

        ax.scatter(phys_dists, emb_dists, s=2, alpha=0.3, color="steelblue")
        corr, _ = spearmanr(phys_dists, emb_dists)
        ax.set_xlabel("Physical Distance")
        ax.set_ylabel("Embedding Distance")
        ax.set_title(f"Spearman r={corr:.3f}\n{method_title}", fontsize=9)

    plt.tight_layout()
    png_path = OUTPUT_DIR / "embedding_comparison.png"
    svg_path = OUTPUT_DIR / "embedding_comparison.svg"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved embedding comparison to {png_path}")


def plot_temporal_importance(
    all_frame_embeddings: List[np.ndarray],
    obj_positions: np.ndarray,
    all_results: List[dict],
    config: SearchConfig,
):
    """Generate temporal importance curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_bins = np.arange(0, 101, 1)
    importance = np.zeros(101)
    count = np.zeros(101)

    for r in all_results:
        if "error" in r:
            continue
        score = r.get("position_correlation", 0.0)
        start = int(r["start_pct"] * 100)
        end = int(r["end_pct"] * 100)
        start = max(0, min(100, start))
        end = max(0, min(100, end))
        if end > start:
            importance[start:end + 1] += score
            count[start:end + 1] += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_importance = np.where(count > 0, importance / count, 0)

    from scipy.ndimage import gaussian_filter1d
    avg_importance = gaussian_filter1d(avg_importance, sigma=3)

    best_result = None
    for r in all_results:
        if "error" in r:
            continue
        if best_result is None or r["position_correlation"] > best_result["position_correlation"]:
            best_result = r

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    x = np.arange(101)
    ax.plot(x, avg_importance, linewidth=2, color="steelblue", label="Temporal Importance")
    ax.fill_between(x, avg_importance, alpha=0.3, color="steelblue")

    if best_result is not None:
        bs = best_result["start_pct"] * 100
        be = best_result["end_pct"] * 100
        ax.axvspan(bs, be, alpha=0.2, color="red", label=f"Best Window: {bs:.0f}%-{be:.0f}%")
        ax.axvline(bs, color="red", linestyle="--", alpha=0.7)
        ax.axvline(be, color="red", linestyle="--", alpha=0.7)

    ax.set_xlabel("Normalized Episode Time (%)")
    ax.set_ylabel("Temporal Importance")
    ax.set_title("Temporal Importance Curve\n(Higher = More Informative for Position Correlation)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    png_path = OUTPUT_DIR / "temporal_importance.png"
    svg_path = OUTPUT_DIR / "temporal_importance.svg"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved temporal importance to {png_path}")


def save_evaluation_results(
    all_results: List[dict],
    best_result: dict,
    full_ep_result: dict,
):
    """Save evaluation results to CSV."""
    df = pd.DataFrame(all_results)
    df = df.sort_values("position_correlation", ascending=False)
    df["recommended"] = ""
    if best_result is not None:
        best_key = best_result.get("cache_key", "")
        mask = df["cache_key"] == best_key
        df.loc[mask, "recommended"] = "★"

    csv_path = OUTPUT_DIR / "evaluation.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved evaluation results to {csv_path}")

    search_df = pd.DataFrame(all_results)
    search_df = search_df.sort_values("position_correlation", ascending=False)
    if "rank" not in search_df.columns:
        search_df.insert(0, "rank", range(1, len(search_df) + 1))
    else:
        search_df = search_df.drop(columns=["rank"])
        search_df.insert(0, "rank", range(1, len(search_df) + 1))
    search_csv = OUTPUT_DIR / "temporal_search_results.csv"
    search_df.to_csv(search_csv, index=False)
    print(f"Saved temporal search results to {search_csv}")


def print_summary(
    best_result: dict,
    full_ep_result: dict,
    n_episodes: int,
    frame_dim: int,
    pca_dim: int,
):
    """Print final summary."""
    method = best_result.get("method_name", "N/A")
    start_pct = best_result.get("start_pct", 0.0) * 100
    end_pct = best_result.get("end_pct", 1.0) * 100
    wl_pct = best_result.get("window_length_pct", 1.0) * 100
    weighting = best_result.get("weighting", "N/A")
    pos_corr = best_result.get("position_correlation", 0.0)
    d_bar = best_result.get("d_bar_ratio", 0.0)
    cluster_sep = best_result.get("cluster_separation", 0.0)
    overall = best_result.get("overall_score", 0.0)

    full_corr = full_ep_result.get("position_correlation", 0.0)
    improvement = ((pos_corr - full_corr) / (abs(full_corr) + 1e-10)) * 100

    print("\n" + "=" * 60)
    print("ADAPTIVE TEMPORAL EMBEDDING SEARCH")
    print("=" * 60)
    print(f"\nDataset:")
    print(f"MetaWorld PickPlace-v3")
    print(f"\nEpisodes:")
    print(f"{n_episodes}")
    print(f"\nFrame embedding dimension:")
    print(f"{frame_dim}")
    print(f"\nFinal episode embedding dimension:")
    print(f"{pca_dim}")

    print("\n" + "-" * 60)
    print("BEST STRATEGY")
    print("-" * 60)
    print(f"\nMethod:")
    print(f"{method}")
    print(f"\nTemporal window:")
    print(f"{start_pct:.0f}% - {end_pct:.0f}%")
    print(f"\nWindow length:")
    print(f"{wl_pct:.0f}%")
    print(f"\nWeighting:")
    print(f"{weighting}")

    print("\n" + "-" * 60)
    print("METRICS")
    print("-" * 60)
    print(f"\nPosition correlation:")
    print(f"{pos_corr:.4f}")
    print(f"\nd_bar ratio:")
    print(f"{d_bar:.4f}")
    print(f"\nCluster separation:")
    print(f"{cluster_sep:.4f}")
    print(f"\nOverall score:")
    print(f"{overall:.4f}")

    print("\n" + "-" * 60)
    print("BASELINE COMPARISON")
    print("-" * 60)
    print(f"\nFull episode:")
    print(f"{full_corr:.4f}")
    print(f"\nBest temporal strategy:")
    print(f"{pos_corr:.4f}")
    print(f"\nImprovement:")
    print(f"{improvement:+.2f}%")

    print("\n" + "-" * 60)
    print("RECOMMENDATION")
    print("-" * 60)
    print(f"\nUse:")
    print(f"{method}")
    print(f"\nEffective temporal region:")
    print(f"{start_pct:.0f}% - {end_pct:.0f}%")
    print(f"\nReason:")
    print(f"Highest position correlation on held-out episodes.")
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Adaptive Temporal Embedding Search")
    parser.add_argument("--force", action="store_true", help="Force re-computation")
    args = parser.parse_args()

    config = SearchConfig()

    config_path = OUTPUT_DIR / "search_config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(config), f, indent=2)
    print(f"Saved search config to {config_path}")

    print("Loading episode metadata...")
    meta_path = POOL_DIR / "episode_metadata.csv"
    df_meta = pd.read_csv(meta_path)
    episode_indices = df_meta["episode_index"].tolist()
    obj_positions = df_meta[["obj_x", "obj_y", "obj_z"]].values

    print(f"Total episodes: {len(episode_indices)}")

    print("Loading frame embeddings...")
    all_frame_embeddings = []
    for ep_idx in episode_indices:
        ep_path = ALL_FRAMES_DIR / f"episode_{ep_idx:04d}.npy"
        if not ep_path.exists():
            print(f"  WARNING: {ep_path} not found, skipping")
            continue
        emb = np.load(ep_path)
        emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
        all_frame_embeddings.append(emb)

    n_episodes = len(all_frame_embeddings)
    frame_dim = all_frame_embeddings[0].shape[1] if len(all_frame_embeddings) > 0 else 1024
    print(f"Loaded {n_episodes} episodes, frame dim = {frame_dim}")

    print("\nSplitting into search/eval sets...")
    search_indices, eval_indices = train_test_split(
        list(range(n_episodes)),
        test_size=config.test_size,
        random_state=config.random_seed,
    )
    print(f"Search set: {len(search_indices)} episodes")
    print(f"Eval set: {len(eval_indices)} episodes")

    print("\n" + "=" * 60)
    print("SEARCHING FOR BEST TEMPORAL STRATEGY")
    print("=" * 60)

    all_results = search_best_temporal_strategy(
        all_frame_embeddings,
        obj_positions,
        config,
        search_indices,
        eval_indices,
        force_recompute=args.force,
    )

    valid_results = [r for r in all_results if "error" not in r]
    if len(valid_results) == 0:
        print("ERROR: No valid results found!")
        return

    best_result = valid_results[0]

    full_ep_result = None
    for r in valid_results:
        if r["method_name"] == "uniform_full":
            full_ep_result = r
            break

    if full_ep_result is None:
        full_embs, _ = extract_episode_embeddings(
            [all_frame_embeddings[i] for i in eval_indices],
            "uniform_full",
            0.0, 1.0,
            pca_dim=config.pca_dim,
        )
        full_ep_result = evaluate_episode_embedding(
            "uniform_full",
            full_embs,
            obj_positions[eval_indices],
            config,
        )
        full_ep_result["overall_score"] = compute_overall_score(
            full_ep_result["position_correlation"],
            full_ep_result["d_bar_ratio"],
            full_ep_result["cluster_separation"],
        )
        full_ep_result["start_pct"] = 0.0
        full_ep_result["end_pct"] = 1.0
        full_ep_result["window_length_pct"] = 1.0
        full_ep_result["weighting"] = "uniform"

    print("\n" + "=" * 60)
    print("TOP 20 TEMPORAL STRATEGIES")
    print("=" * 60)
    print(f"{'Rank':<5} {'Method':<28} {'Start%':<7} {'End%':<7} {'Len%':<7} {'Weight':<12} {'PosCorr':<8} {'d_bar':<7} {'Cluster':<8} {'Overall':<8}")
    print("-" * 110)
    for i, r in enumerate(valid_results[:20]):
        print(f"{i+1:<5} {r['method_name']:<28} {r['start_pct']*100:<7.0f} {r['end_pct']*100:<7.0f} {r['window_length_pct']*100:<7.0f} {r['weighting']:<12} {r['position_correlation']:<8.4f} {r['d_bar_ratio']:<7.4f} {r['cluster_separation']:<8.4f} {r['overall_score']:<8.4f}")

    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    plot_embedding_comparison(
        all_frame_embeddings,
        obj_positions,
        valid_results[:5],
        config,
    )
    plot_temporal_importance(
        all_frame_embeddings,
        obj_positions,
        valid_results,
        config,
    )

    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    save_evaluation_results(valid_results, best_result, full_ep_result)

    print("\nSaving final episode embeddings (ALL 500 episodes)...")
    all_indices = list(range(len(all_frame_embeddings)))
    all_frames = [all_frame_embeddings[i] for i in all_indices]
    final_embs, pca_model = extract_episode_embeddings(
        all_frames,
        best_result["method_name"],
        best_result["start_pct"],
        best_result["end_pct"],
        best_result["weighting"],
        best_result.get("center", 0.5),
        best_result.get("sigma", 0.2),
        config.pca_dim,
    )
    np.save(OUTPUT_DIR / "episode_embeddings.npy", final_embs)

    with open(OUTPUT_DIR / "pca_model.pkl", "wb") as f:
        pickle.dump(pca_model, f)

    meta_out = []
    for i, ep_idx in enumerate(all_indices):
        meta_out.append({
            "episode_id": ep_idx,
            "object_x": obj_positions[ep_idx, 0],
            "object_y": obj_positions[ep_idx, 1],
            "object_z": obj_positions[ep_idx, 2],
            "success": df_meta.iloc[ep_idx].get("success", True),
            "selected_method": best_result["method_name"],
            "window_start": best_result["start_pct"],
            "window_end": best_result["end_pct"],
        })
    meta_df = pd.DataFrame(meta_out)
    meta_df.to_csv(OUTPUT_DIR / "episode_embeddings_metadata.csv", index=False)

    print_summary(
        best_result,
        full_ep_result,
        n_episodes,
        frame_dim,
        config.pca_dim,
    )


if __name__ == "__main__":
    main()