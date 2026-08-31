#!/usr/bin/env python
"""
Dataset Embedding Analysis Tool

验证 SmolVLM episode embedding 是否能够区分不同 MetaWorld task states，
以及 embedding distance 是否具有任务状态意义。

Usage:
    python analyze_dataset_embedding.py \
        --dataset-root /data/zhonglinye/jun/lerobot/personal/work2/dataset_view/pick_place_corner3 \
        --embeddings-dir <existing_embedding_dir> \
        --random-subset <optional> \
        --uniform-subset <optional> \
        --ours-subset <optional> \
        --output-dir personal/work2/see_dataset_after_eval/analysis_results/corner3 \
        --n-bootstrap 1000 \
        --seed 42
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.stats import spearmanr
from scipy.spatial.distance import pairwise_distances
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent))

from sic_v2 import FixedAnchorSIC, check_embeddings_valid
from analysis_utils import compute_fixed_universe_sic


def load_embeddings(embeddings_dir: Path) -> Dict[int, Dict]:
    """Load embedding cache"""
    embeddings = {}
    for f in embeddings_dir.glob("*.npy"):
        try:
            data = np.load(f, allow_pickle=True).item()
            ep_idx = data.get("episode_index")
            if ep_idx is not None:
                embeddings[ep_idx] = {
                    "phi_global": data["phi_global"],
                    "phi_wrist": data["phi_wrist"]
                }
        except Exception as e:
            print(f"  Skip invalid file: {f.name} ({e})")
    return embeddings


def load_episode_metadata(dataset_root: Path) -> Dict[int, Dict]:
    """Load episode_initial_states.json"""
    metadata_file = dataset_root / "episode_initial_states.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file) as f:
        data = json.load(f)
    
    episodes = {}
    for ep in data["episodes"]:
        ep_idx = ep["episode_index"]
        episodes[ep_idx] = {
            "obj_init_pos": np.array(ep["obj_init_pos"]),
            "goal_pos": np.array(ep.get("goal_pos", ep.get("goal_pose", [0, 0, 0]))),
        }
    
    return episodes


def align_embeddings_with_metadata(
    embeddings: Dict[int, Dict],
    metadata: Dict[int, Dict]
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Align embeddings with metadata.
    
    Returns:
        episode_indices, phi_globals, phi_wrists, obj_init_positions, goal_positions
    """
    embedding_episodes = set(embeddings.keys())
    metadata_episodes = set(metadata.keys())
    
    intersection = embedding_episodes & metadata_episodes
    missing_embeddings = metadata_episodes - embedding_episodes
    missing_metadata = embedding_episodes - metadata_episodes
    
    print(f"\n{'='*60}")
    print(f"Episode Alignment Report")
    print(f"{'='*60}")
    print(f"  Metadata episodes: {len(metadata_episodes)}")
    print(f"  Embedding episodes: {len(embedding_episodes)}")
    print(f"  Intersection: {len(intersection)}")
    
    if missing_embeddings:
        print(f"  WARNING: {len(missing_embeddings)} episodes missing embeddings")
        if len(missing_embeddings) <= 20:
            print(f"    Missing: {sorted(missing_embeddings)}")
        else:
            print(f"    First 10: {sorted(missing_embeddings)[:10]}")
    
    if missing_metadata:
        print(f"  WARNING: {len(missing_metadata)} episodes missing metadata")
        if len(missing_metadata) <= 20:
            print(f"    Missing: {sorted(missing_metadata)}")
        else:
            print(f"    First 10: {sorted(missing_metadata)[:10]}")
    
    if not intersection:
        raise ValueError("No intersection between embeddings and metadata!")
    
    episode_indices = sorted(intersection)
    phi_globals = np.array([embeddings[ep]["phi_global"] for ep in episode_indices])
    phi_wrists = np.array([embeddings[ep]["phi_wrist"] for ep in episode_indices])
    obj_init_positions = np.array([metadata[ep]["obj_init_pos"] for ep in episode_indices])
    goal_positions = np.array([metadata[ep]["goal_pos"] for ep in episode_indices])
    
    return episode_indices, phi_globals, phi_wrists, obj_init_positions, goal_positions


def compute_combined_embeddings(
    phi_globals: np.ndarray,
    phi_wrists: np.ndarray,
    lambda_wrist: float = 1.0
) -> np.ndarray:
    """
    Compute combined embeddings:
    combined = concat(g, lambda_wrist * w)
    where g and w are L2 normalized.
    """
    norms_g = np.linalg.norm(phi_globals, axis=1, keepdims=True)
    norms_w = np.linalg.norm(phi_wrists, axis=1, keepdims=True)
    
    g_norm = phi_globals / (norms_g + 1e-10)
    w_norm = phi_wrists / (norms_w + 1e-10)
    
    combined = np.concatenate([g_norm, lambda_wrist * w_norm], axis=1)
    return combined


def compute_effective_rank(phi: np.ndarray) -> float:
    """
    Compute effective rank of embedding matrix.
    
    effective_rank = exp(-sum(p_i * log(p_i + eps)))
    where p_i = lambda_i / sum(lambda)
    """
    centered = phi - phi.mean(axis=0)
    cov = centered.T @ centered / (len(centered) - 1)
    
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)
    
    total = eigenvalues.sum()
    if total < 1e-10:
        return 0.0
    
    p = eigenvalues / total
    p = p[p > 1e-15]
    
    entropy = -np.sum(p * np.log(p + 1e-15))
    effective_rank = np.exp(entropy)
    
    return float(effective_rank)


def compute_embedding_statistics(phi: np.ndarray, name: str) -> Dict:
    """Compute comprehensive embedding statistics"""
    norms = np.linalg.norm(phi, axis=1)
    
    dist_matrix = pairwise_distances(phi)
    upper_tri = dist_matrix[np.triu_indices(len(phi), k=1)]
    
    stats = {
        "name": name,
        "dimension": phi.shape[1],
        "n_episodes": len(phi),
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "norm_min": float(np.min(norms)),
        "norm_max": float(np.max(norms)),
        "pairwise_dist_mean": float(np.mean(upper_tri)),
        "pairwise_dist_std": float(np.std(upper_tri)),
        "pairwise_dist_p05": float(np.percentile(upper_tri, 5)),
        "pairwise_dist_p25": float(np.percentile(upper_tri, 25)),
        "pairwise_dist_p50": float(np.percentile(upper_tri, 50)),
        "pairwise_dist_p75": float(np.percentile(upper_tri, 75)),
        "pairwise_dist_p95": float(np.percentile(upper_tri, 95)),
        "pairwise_dist_max": float(np.max(upper_tri)),
        "effective_rank": compute_effective_rank(phi),
    }
    
    return stats


def compute_spearman_correlation(
    physical_distances: np.ndarray,
    embedding_distances: np.ndarray,
    n_samples: int = 50000,
    seed: int = 42
) -> Tuple[float, float, int]:
    """
    Compute Spearman correlation between physical and embedding distances.
    Sample pairs if too many.
    """
    rng = np.random.RandomState(seed)
    
    n = len(physical_distances)
    indices = np.triu_indices(n, k=1)
    n_pairs = len(indices[0])
    
    if n_pairs > n_samples:
        sample_idx = rng.choice(n_pairs, n_samples, replace=False)
        phys_dist = physical_distances[indices[0][sample_idx], indices[1][sample_idx]]
        emb_dist = embedding_distances[indices[0][sample_idx], indices[1][sample_idx]]
    else:
        phys_dist = physical_distances[indices]
        emb_dist = embedding_distances[indices]
    
    rho, p_value = spearmanr(phys_dist, emb_dist)
    
    return float(rho), float(p_value), len(phys_dist)


def permutation_test_spearman(
    physical_distances: np.ndarray,
    embedding_distances: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
    n_samples: int = 50000
) -> Dict:
    """
    Permutation test for Spearman correlation.
    """
    rng = np.random.RandomState(seed)
    
    n = len(physical_distances)
    indices = np.triu_indices(n, k=1)
    n_pairs = len(indices[0])
    
    if n_pairs > n_samples:
        sample_idx = rng.choice(n_pairs, n_samples, replace=False)
        phys_full = physical_distances[indices[0][sample_idx], indices[1][sample_idx]]
        emb_full = embedding_distances[indices[0][sample_idx], indices[1][sample_idx]]
    else:
        phys_full = physical_distances[indices]
        emb_full = embedding_distances[indices]
    
    observed_rho, _ = spearmanr(phys_full, emb_full)
    
    null_rhos = []
    
    for perm_i in range(n_permutations):
        shuffled_indices = rng.permutation(n)
        shuffled_phys = physical_distances[shuffled_indices][:, shuffled_indices]
        
        if n_pairs > n_samples:
            phys_shuffled = shuffled_phys[indices[0][sample_idx], indices[1][sample_idx]]
        else:
            phys_shuffled = shuffled_phys[indices]
        
        rho, _ = spearmanr(phys_shuffled, emb_full)
        null_rhos.append(rho)
        
        if (perm_i + 1) % 200 == 0:
            print(f"    Permutation {perm_i + 1}/{n_permutations}")
    
    null_rhos = np.array(null_rhos)
    
    percentile = float(np.sum(null_rhos >= observed_rho) / len(null_rhos))
    p_value = min(percentile, 1 - percentile) * 2
    
    return {
        "observed_rho": float(observed_rho),
        "null_mean": float(np.mean(null_rhos)),
        "null_std": float(np.std(null_rhos)),
        "percentile": percentile,
        "permutation_p_value": float(p_value),
        "n_permutations": n_permutations
    }


def position_probe(
    phi: np.ndarray,
    positions: np.ndarray,
    n_shuffles: int = 100,
    seed: int = 42
) -> Dict:
    """
    Position probing with Ridge and KNN regression.
    """
    rng = np.random.RandomState(seed)
    
    results = {
        "ridge": {"R2_x": 0.0, "R2_y": 0.0, "MAE_x": 0.0, "MAE_y": 0.0},
        "knn": {"R2_x": 0.0, "R2_y": 0.0, "MAE_x": 0.0, "MAE_y": 0.0},
        "shuffled_ridge": {"R2_x": [], "R2_y": [], "MAE_x": [], "MAE_y": []},
        "shuffled_knn": {"R2_x": [], "R2_y": [], "MAE_x": [], "MAE_y": []}
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    
    for dim in range(2):
        y = positions[:, dim]
        dim_name = "x" if dim == 0 else "y"
        
        ridge_preds = np.zeros(len(y))
        knn_preds = np.zeros(len(y))
        
        for train_idx, test_idx in kf.split(phi):
            X_train, X_test = phi[train_idx], phi[test_idx]
            y_train = y[train_idx]
            
            ridge_model = Ridge(alpha=1.0).fit(X_train, y_train)
            ridge_preds[test_idx] = ridge_model.predict(X_test)
            
            knn_model = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)
            knn_preds[test_idx] = knn_model.predict(X_test)
        
        results["ridge"][f"R2_{dim_name}"] = float(r2_score(y, ridge_preds))
        results["ridge"][f"MAE_{dim_name}"] = float(mean_absolute_error(y, ridge_preds))
        results["knn"][f"R2_{dim_name}"] = float(r2_score(y, knn_preds))
        results["knn"][f"MAE_{dim_name}"] = float(mean_absolute_error(y, knn_preds))
        
        for shuffle_i in range(n_shuffles):
            y_shuffled = y[rng.permutation(len(y))]
            
            ridge_shuf_preds = np.zeros(len(y))
            knn_shuf_preds = np.zeros(len(y))
            
            for train_idx, test_idx in kf.split(phi):
                X_train, X_test = phi[train_idx], phi[test_idx]
                y_train_shuf = y_shuffled[train_idx]
                
                ridge_model = Ridge(alpha=1.0).fit(X_train, y_train_shuf)
                ridge_shuf_preds[test_idx] = ridge_model.predict(X_test)
                
                knn_model = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train_shuf)
                knn_shuf_preds[test_idx] = knn_model.predict(X_test)
            
            results["shuffled_ridge"][f"R2_{dim_name}"].append(float(r2_score(y, ridge_shuf_preds)))
            results["shuffled_ridge"][f"MAE_{dim_name}"].append(float(mean_absolute_error(y, ridge_shuf_preds)))
            results["shuffled_knn"][f"R2_{dim_name}"].append(float(r2_score(y, knn_shuf_preds)))
            results["shuffled_knn"][f"MAE_{dim_name}"].append(float(mean_absolute_error(y, knn_shuf_preds)))
            
            if (shuffle_i + 1) % 20 == 0:
                print(f"    Shuffle {shuffle_i + 1}/{n_shuffles} for {dim_name}")
    
    for key in ["shuffled_ridge", "shuffled_knn"]:
        for metric in ["R2_x", "R2_y", "MAE_x", "MAE_y"]:
            arr = results[key][metric]
            results[key][metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr))
            }
    
    return results


def neighbor_overlap_analysis(
    phi: np.ndarray,
    positions: np.ndarray,
    ks: List[int] = [5, 10, 20],
    seed: int = 42
) -> Dict:
    """
    Analyze neighborhood preservation between physical and embedding space.
    """
    rng = np.random.RandomState(seed)
    
    n = len(phi)
    phys_dist = pairwise_distances(positions[:, :2])
    emb_dist = pairwise_distances(phi)
    
    results = {}
    
    for k in ks:
        phys_neighbors = np.argsort(phys_dist, axis=1)[:, 1:k+1]
        emb_neighbors = np.argsort(emb_dist, axis=1)[:, 1:k+1]
        
        overlaps = []
        for i in range(n):
            phys_set = set(phys_neighbors[i])
            emb_set = set(emb_neighbors[i])
            intersection = phys_set & emb_set
            union = phys_set | emb_set
            overlaps.append(len(intersection) / len(union) if union else 0)
        
        results[f"neighbor_overlap@{k}"] = float(np.mean(overlaps))
    
    phi_random = phi[rng.permutation(n)]
    emb_dist_random = pairwise_distances(phi_random)
    
    for k in ks:
        emb_neighbors_random = np.argsort(emb_dist_random, axis=1)[:, 1:k+1]
        phys_neighbors_fixed = np.argsort(phys_dist, axis=1)[:, 1:k+1]
        
        overlaps = []
        for i in range(n):
            phys_set = set(phys_neighbors_fixed[i])
            emb_set = set(emb_neighbors_random[i])
            intersection = phys_set & emb_set
            union = phys_set | emb_set
            overlaps.append(len(intersection) / len(union) if union else 0)
        
        results[f"random_neighbor_overlap@{k}"] = float(np.mean(overlaps))
    
    return results


def grid_classifiability(
    phi: np.ndarray,
    positions: np.ndarray,
    grid_sizes: List[Tuple[int, int]] = [(7, 4), (14, 8)],
    n_shuffles: int = 100,
    seed: int = 42
) -> Dict:
    """
    Grid-based classification analysis.
    """
    rng = np.random.RandomState(seed)
    
    results = {}
    
    for grid_x, grid_y in grid_sizes:
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        x_bins = np.linspace(x_min, x_max, grid_x + 1)
        y_bins = np.linspace(y_min, y_max, grid_y + 1)
        
        x_labels = np.digitize(positions[:, 0], x_bins) - 1
        y_labels = np.digitize(positions[:, 1], y_bins) - 1
        
        x_labels = np.clip(x_labels, 0, grid_x - 1)
        y_labels = np.clip(y_labels, 0, grid_y - 1)
        
        grid_ids = x_labels * grid_y + y_labels
        
        cell_counts = np.bincount(grid_ids, minlength=grid_x * grid_y)
        empty_cells = int(np.sum(cell_counts == 0))
        sparse_cells = int(np.sum((cell_counts > 0) & (cell_counts < 3)))
        
        if empty_cells > 0:
            print(f"  WARNING: {grid_x}x{grid_y} grid has {empty_cells} empty cells")
        
        knn = KNeighborsClassifier(n_neighbors=5)
        
        try:
            accuracies = cross_val_score(knn, phi, grid_ids, cv=5, scoring='accuracy')
            balanced_accuracies = cross_val_score(knn, phi, grid_ids, cv=5, scoring='balanced_accuracy')
        except Exception as e:
            print(f"  WARNING: Grid classification failed for {grid_x}x{grid_y}: {e}")
            accuracies = np.array([0.0])
            balanced_accuracies = np.array([0.0])
        
        shuffled_accs = []
        for _ in range(n_shuffles):
            grid_ids_shuffled = grid_ids[rng.permutation(len(grid_ids))]
            try:
                acc = np.mean(cross_val_score(knn, phi, grid_ids_shuffled, cv=5, scoring='accuracy'))
            except:
                acc = 0.0
            shuffled_accs.append(acc)
        
        results[f"{grid_x}x{grid_y}"] = {
            "accuracy": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "balanced_accuracy": float(np.mean(balanced_accuracies)),
            "balanced_accuracy_std": float(np.std(balanced_accuracies)),
            "shuffled_accuracy_mean": float(np.mean(shuffled_accs)),
            "shuffled_accuracy_std": float(np.std(shuffled_accs)),
            "empty_cells": empty_cells,
            "sparse_cells": sparse_cells,
            "total_cells": grid_x * grid_y
        }
    
    return results


def generate_visualizations(
    phi_globals: np.ndarray,
    phi_wrists: np.ndarray,
    phi_combined: np.ndarray,
    positions: np.ndarray,
    output_dir: Path
):
    """Generate all required visualizations"""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Generating PCA visualizations...")
    
    pca_global = PCA(n_components=2).fit_transform(phi_globals)
    pca_wrist = PCA(n_components=2).fit_transform(phi_wrists)
    pca_combined = PCA(n_components=2).fit_transform(phi_combined)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for ax, pca_result, name in [
        (axes[0, 0], pca_global, "Global"),
        (axes[0, 1], pca_wrist, "Wrist"),
        (axes[0, 2], pca_combined, "Combined"),
    ]:
        sc = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=positions[:, 0], cmap='viridis', s=10, alpha=0.7)
        ax.set_title(f"{name} PCA - colored by obj_x")
        plt.colorbar(sc, ax=ax)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    
    for ax, pca_result, name in [
        (axes[1, 0], pca_global, "Global"),
        (axes[1, 1], pca_wrist, "Wrist"),
        (axes[1, 2], pca_combined, "Combined"),
    ]:
        sc = ax.scatter(pca_result[:, 0], pca_result[:, 1], c=positions[:, 1], cmap='plasma', s=10, alpha=0.7)
        ax.set_title(f"{name} PCA - colored by obj_y")
        plt.colorbar(sc, ax=ax)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    
    plt.tight_layout()
    plt.savefig(figures_dir / "pca_positions.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Generating distance correlation plots...")
    
    phys_dist = pairwise_distances(positions[:, :2])
    global_dist = pairwise_distances(phi_globals)
    wrist_dist = pairwise_distances(phi_wrists)
    combined_dist = pairwise_distances(phi_combined)
    
    n = len(phi_globals)
    indices = np.triu_indices(n, k=1)
    phys_flat = phys_dist[indices]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, emb_dist, name in [
        (axes[0], global_dist[indices], "Global"),
        (axes[1], wrist_dist[indices], "Wrist"),
        (axes[2], combined_dist[indices], "Combined"),
    ]:
        ax.hexbin(phys_flat, emb_dist, gridsize=50, cmap='Blues', mincnt=1)
        ax.set_xlabel("Physical XY Distance")
        ax.set_ylabel(f"{name} Embedding Distance")
        ax.set_title(f"Physical vs {name} Embedding Distance")
    
    plt.tight_layout()
    plt.savefig(figures_dir / "physical_vs_embedding_distance.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Generating distance histograms...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, emb_dist, name in [
        (axes[0], global_dist[indices], "Global"),
        (axes[1], wrist_dist[indices], "Wrist"),
        (axes[2], combined_dist[indices], "Combined"),
    ]:
        ax.hist(emb_dist, bins=100, alpha=0.7, edgecolor='black')
        ax.set_xlabel("Embedding Distance")
        ax.set_ylabel("Count")
        ax.set_title(f"{name} Pairwise Distance Histogram")
    
    plt.tight_layout()
    plt.savefig(figures_dir / "pairwise_distance_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Figures saved to: {figures_dir}")


def match_subset_to_indices(
    subset_episodes: List[int],
    all_episode_indices: List[int]
) -> List[int]:
    """Match subset episodes to indices in the aligned dataset"""
    idx_map = {ep: i for i, ep in enumerate(all_episode_indices)}
    matched = []
    missing = []
    
    for ep in subset_episodes:
        if ep in idx_map:
            matched.append(idx_map[ep])
        else:
            missing.append(ep)
    
    if missing:
        print(f"  WARNING: {len(missing)} subset episodes not in aligned dataset")
        if len(missing) <= 10:
            print(f"    Missing: {missing}")
    
    return matched


def compute_subset_coverage(
    all_phi: np.ndarray,
    subset_indices: List[int],
    all_positions: np.ndarray,
    name: str
) -> Dict:
    """
    Compute coverage metrics for a subset.
    """
    if not subset_indices:
        return None
    
    subset_set = set(subset_indices)
    unselected = [i for i in range(len(all_phi)) if i not in subset_set]
    
    if not unselected:
        return None
    
    dist_matrix = pairwise_distances(all_phi)
    
    nearest_dists = []
    for ui in unselected:
        min_dist = min(dist_matrix[ui, si] for si in subset_indices)
        nearest_dists.append(min_dist)
    
    nearest_dists = np.array(nearest_dists)
    
    phys_dist_matrix = pairwise_distances(all_positions[:, :2])
    phys_nearest = []
    for ui in unselected:
        min_dist = min(phys_dist_matrix[ui, si] for si in subset_indices)
        phys_nearest.append(min_dist)
    
    phys_nearest = np.array(phys_nearest)
    
    coverage = {
        "method": name,
        "mean_nearest_distance": float(np.mean(nearest_dists)),
        "median_nearest_distance": float(np.median(nearest_dists)),
        "p90_nearest_distance": float(np.percentile(nearest_dists, 90)),
        "p95_nearest_distance": float(np.percentile(nearest_dists, 95)),
        "max_nearest_distance": float(np.max(nearest_dists)),
        "physical_mean_nearest": float(np.mean(phys_nearest)),
        "physical_p95": float(np.percentile(phys_nearest, 95)),
        "physical_max_radius": float(np.max(phys_nearest)),
        "subset_size": len(subset_indices),
        "unselected_size": len(unselected),
    }
    
    return coverage


def compute_subset_redundancy(
    subset_phi: np.ndarray,
    full_dataset_phi: np.ndarray,
    name: str
) -> Dict:
    """
    Compute redundancy metrics for a subset.
    """
    if len(subset_phi) < 2:
        return None
    
    dist_matrix = pairwise_distances(subset_phi)
    np.fill_diagonal(dist_matrix, np.inf)
    
    nearest_dists = np.min(dist_matrix, axis=1)
    
    full_dist = pairwise_distances(full_dataset_phi)
    upper_tri = full_dist[np.triu_indices(len(full_dataset_phi), k=1)]
    p05_threshold = float(np.percentile(upper_tri, 5))
    
    redundancy_fraction = float(np.mean(nearest_dists < p05_threshold))
    
    redundancy = {
        "method": name,
        "mean_nearest": float(np.mean(nearest_dists)),
        "median_nearest": float(np.median(nearest_dists)),
        "p10_nearest": float(np.percentile(nearest_dists, 10)),
        "p50_nearest": float(np.median(nearest_dists)),
        "p90_nearest": float(np.percentile(nearest_dists, 90)),
        "redundancy_threshold_p05": p05_threshold,
        "redundancy_fraction": redundancy_fraction,
    }
    
    return redundancy


def random_bootstrap_analysis(
    all_phi: np.ndarray,
    subset_indices: List[int],
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict:
    """
    Random bootstrap baseline analysis.
    """
    rng = np.random.RandomState(seed)
    n_total = len(all_phi)
    subset_size = len(subset_indices)
    
    if subset_size == 0 or subset_size >= n_total:
        return None
    
    bootstrap_metrics = {
        "mean_nearest": [],
        "p95_nearest": [],
        "max_radius": [],
    }
    
    dist_matrix = pairwise_distances(all_phi)
    
    print(f"  Running {n_bootstrap} bootstrap iterations...")
    for boot_i in range(n_bootstrap):
        sample_indices = rng.choice(n_total, subset_size, replace=False)
        
        unselected = [i for i in range(n_total) if i not in set(sample_indices)]
        
        nearest_dists = []
        for ui in unselected:
            min_dist = min(dist_matrix[ui, si] for si in sample_indices)
            nearest_dists.append(min_dist)
        
        nearest_dists = np.array(nearest_dists)
        
        bootstrap_metrics["mean_nearest"].append(float(np.mean(nearest_dists)))
        bootstrap_metrics["p95_nearest"].append(float(np.percentile(nearest_dists, 95)))
        bootstrap_metrics["max_radius"].append(float(np.max(nearest_dists)))
        
        if (boot_i + 1) % 200 == 0:
            print(f"    Bootstrap {boot_i + 1}/{n_bootstrap}")
    
    subset_set = set(subset_indices)
    unselected = [i for i in range(n_total) if i not in subset_set]
    
    subset_nearest_unselected = []
    for ui in unselected:
        min_dist = min(dist_matrix[ui, si] for si in subset_indices)
        subset_nearest_unselected.append(min_dist)
    
    subset_nearest_unselected = np.array(subset_nearest_unselected)
    
    results = {}
    for metric in ["mean_nearest", "p95_nearest", "max_radius"]:
        bootstrap_vals = np.array(bootstrap_metrics[metric])
        
        if metric == "max_radius":
            observed = float(np.max(subset_nearest_unselected))
        elif metric == "mean_nearest":
            observed = float(np.mean(subset_nearest_unselected))
        else:
            observed = float(np.percentile(subset_nearest_unselected, 95))
        
        better_fraction = float(np.sum(bootstrap_vals > observed) / len(bootstrap_vals))
        percentile = float(np.sum(bootstrap_vals <= observed) / len(bootstrap_vals))
        
        results[metric] = {
            "observed": observed,
            "bootstrap_mean": float(np.mean(bootstrap_vals)),
            "bootstrap_std": float(np.std(bootstrap_vals)),
            "better_than_random_fraction": better_fraction,
            "percentile": percentile,
        }
    
    return results


def compute_workspace_coverage(
    all_positions: np.ndarray,
    subset_indices: List[int],
    name: str,
    grid_sizes: List[Tuple[int, int]] = [(7, 4), (14, 8)]
) -> Dict:
    """
    Compute workspace coverage for a subset.
    """
    if not subset_indices:
        return None
    
    results = {"method": name}
    
    phys_dist_matrix = pairwise_distances(all_positions[:, :2])
    
    subset_set = set(subset_indices)
    unselected = [i for i in range(len(all_positions)) if i not in subset_set]
    
    if not unselected:
        return None
    
    nearest_dists = []
    for ui in unselected:
        min_dist = min(phys_dist_matrix[ui, si] for si in subset_indices)
        nearest_dists.append(min_dist)
    
    nearest_dists = np.array(nearest_dists)
    
    results["physical_mean_nearest"] = float(np.mean(nearest_dists))
    results["physical_p95"] = float(np.percentile(nearest_dists, 95))
    results["physical_max_radius"] = float(np.max(nearest_dists))
    
    for grid_x, grid_y in grid_sizes:
        x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
        y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()
        
        x_bins = np.linspace(x_min, x_max, grid_x + 1)
        y_bins = np.linspace(y_min, y_max, grid_y + 1)
        
        x_labels = np.clip(np.digitize(all_positions[:, 0], x_bins) - 1, 0, grid_x - 1)
        y_labels = np.clip(np.digitize(all_positions[:, 1], y_bins) - 1, 0, grid_y - 1)
        
        grid_ids = x_labels * grid_y + y_labels
        
        covered_cells = len(set(grid_ids[subset_indices]))
        total_cells = grid_x * grid_y
        
        results[f"grid_{grid_x}x{grid_y}_covered"] = covered_cells
        results[f"grid_{grid_x}x{grid_y}_total"] = total_cells
        results[f"grid_{grid_x}x{grid_y}_ratio"] = covered_cells / total_cells
    
    return results


def compute_fixed_universe_sic_for_subset(
    subset_indices: List[int],
    all_episode_indices: List[int],
    phi_globals: np.ndarray,
    phi_wrists: np.ndarray,
    name: str,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0
) -> Dict:
    """
    Compute fixed-universe SIC for a subset.
    """
    if not subset_indices:
        return None
    
    subset_episodes = [all_episode_indices[i] for i in subset_indices]
    
    sic_result = compute_fixed_universe_sic(
        selected_episodes=subset_episodes,
        all_episode_indices=all_episode_indices,
        phi_globals=phi_globals,
        phi_wrists=phi_wrists,
        alpha=alpha,
        lambda_wrist=lambda_wrist
    )
    
    return {
        "method": name,
        "fixed_universe_sic": sic_result["fixed_universe_sic"],
        "normalized_sic": sic_result["normalized_sic"],
        "reference_anchor_count": sic_result["reference_anchor_count"],
        "dbar_global": sic_result["dbar_global"],
        "dbar_wrist": sic_result["dbar_wrist"],
    }


def generate_report(
    analysis_results: Dict,
    output_dir: Path
):
    """Generate analysis_report.md"""
    report_path = output_dir / "analysis_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Dataset Embedding Analysis Report\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Total episodes: {analysis_results['n_episodes']}\n")
        f.write(f"- Global dimension: {analysis_results['global_stats']['dimension']}\n")
        f.write(f"- Wrist dimension: {analysis_results['wrist_stats']['dimension']}\n")
        f.write(f"- Combined dimension: {analysis_results['combined_stats']['dimension']}\n\n")
        
        f.write("## Embedding Quality\n\n")
        f.write(f"- Valid: {analysis_results['validation']['valid']}\n")
        f.write(f"- Exact duplicates (global): {analysis_results['validation']['stats']['n_exact_dup_global']}\n")
        f.write(f"- Exact duplicates (wrist): {analysis_results['validation']['stats']['n_exact_dup_wrist']}\n")
        f.write(f"- Near duplicates (global): {analysis_results['validation']['stats']['n_near_dup_global']}\n")
        f.write(f"- Near duplicates (wrist): {analysis_results['validation']['stats']['n_near_dup_wrist']}\n")
        f.write(f"- Zero norm (global): {analysis_results['validation']['stats']['zero_norm_global']}\n")
        f.write(f"- Zero norm (wrist): {analysis_results['validation']['stats']['zero_norm_wrist']}\n\n")
        
        if analysis_results['validation']['errors']:
            f.write("### Errors\n\n")
            for err in analysis_results['validation']['errors']:
                f.write(f"- {err}\n")
            f.write("\n")
        
        if analysis_results['validation']['warnings']:
            f.write("### Warnings\n\n")
            for warn in analysis_results['validation']['warnings']:
                f.write(f"- {warn}\n")
            f.write("\n")
        
        f.write("## Table 1: Representation Quality\n\n")
        f.write("| Representation | Spearman XY | p-value | R2 x | R2 y | NN overlap@10 | Effective Rank |\n")
        f.write("|---------------|-------------|---------|------|------|---------------|----------------|\n")
        
        for rep in ["global", "wrist", "combined"]:
            stats = analysis_results[f"{rep}_stats"]
            spearman = analysis_results["spearman"][rep]
            probe = analysis_results["probe"][rep]
            overlap = analysis_results["neighbor_overlap"][rep]
            
            f.write(f"| {rep} | {spearman['rho']:.4f} | {spearman['p_value']:.2e} | "
                   f"{probe['ridge']['R2_x']:.4f} | {probe['ridge']['R2_y']:.4f} | "
                   f"{overlap['neighbor_overlap@10']:.4f} | {stats['effective_rank']:.2f} |\n")
        
        f.write("\n## Table 2: Subset Coverage Comparison\n\n")
        f.write("| Method | Mean Cover Dist | P95 | Max Radius | Redundancy | Fixed SIC | Random Better Fraction |\n")
        f.write("|--------|----------------|-----|------------|------------|-----------|----------------------|\n")
        
        if "subset_comparison" in analysis_results:
            for method_data in analysis_results["subset_comparison"]:
                mean_cover = method_data.get('coverage', {}).get('mean_nearest_distance', 'N/A')
                p95 = method_data.get('coverage', {}).get('p95_nearest_distance', 'N/A')
                max_radius = method_data.get('coverage', {}).get('max_nearest_distance', 'N/A')
                redundancy = method_data.get('redundancy', {}).get('redundancy_fraction', 'N/A')
                fixed_sic = method_data.get('fixed_sic', {}).get('normalized_sic', 'N/A')
                better_frac = method_data.get('bootstrap', {}).get('mean_nearest', {}).get('better_than_random_fraction', 'N/A')
                
                mean_cover_str = f"{mean_cover:.4f}" if isinstance(mean_cover, float) else str(mean_cover)
                p95_str = f"{p95:.4f}" if isinstance(p95, float) else str(p95)
                max_radius_str = f"{max_radius:.4f}" if isinstance(max_radius, float) else str(max_radius)
                redundancy_str = f"{redundancy:.4f}" if isinstance(redundancy, float) else str(redundancy)
                fixed_sic_str = f"{fixed_sic:.4f}" if isinstance(fixed_sic, float) else str(fixed_sic)
                better_frac_str = f"{better_frac:.4f}" if isinstance(better_frac, float) else str(better_frac)
                
                f.write(f"| {method_data['method']} | {mean_cover_str} | "
                       f"{p95_str} | "
                       f"{max_radius_str} | "
                       f"{redundancy_str} | "
                       f"{fixed_sic_str} | "
                       f"{better_frac_str} |\n")
        
        f.write("\n## Hypothesis Evaluation\n\n")
        
        f.write("### H1: Embedding 是否具有可区分性？\n\n")
        eff_rank_global = analysis_results["global_stats"]["effective_rank"]
        eff_rank_wrist = analysis_results["wrist_stats"]["effective_rank"]
        eff_rank_combined = analysis_results["combined_stats"]["effective_rank"]
        
        dim_global = analysis_results["global_stats"]["dimension"]
        
        if eff_rank_global > dim_global * 0.1 and eff_rank_global > 10:
            h1_status = "SUPPORTED"
        elif eff_rank_global > dim_global * 0.05:
            h1_status = "WEAK"
        else:
            h1_status = "NOT SUPPORTED"
        
        f.write(f"Status: **{h1_status}**\n\n")
        f.write(f"Evidence:\n")
        f.write(f"- Global effective rank: {eff_rank_global:.2f} / {dim_global}\n")
        f.write(f"- Wrist effective rank: {eff_rank_wrist:.2f} / {analysis_results['wrist_stats']['dimension']}\n")
        f.write(f"- Combined effective rank: {eff_rank_combined:.2f} / {analysis_results['combined_stats']['dimension']}\n\n")
        
        f.write("### H2: Embedding distance 是否具有 task-state geometry？\n\n")
        spearman_global = analysis_results["spearman"]["global"]
        perm_test = analysis_results["permutation_test"]
        
        if spearman_global["rho"] > 0.3 and perm_test["permutation_p_value"] < 0.05:
            h2_status = "SUPPORTED"
        elif spearman_global["rho"] > 0.1 and perm_test["permutation_p_value"] < 0.1:
            h2_status = "WEAK"
        else:
            h2_status = "NOT SUPPORTED"
        
        f.write(f"Status: **{h2_status}**\n\n")
        f.write(f"Evidence:\n")
        f.write(f"- Global Spearman rho: {spearman_global['rho']:.4f} (p={spearman_global['p_value']:.2e})\n")
        f.write(f"- Permutation test p-value: {perm_test['permutation_p_value']:.4f}\n")
        f.write(f"- Null distribution: mean={perm_test['null_mean']:.4f}, std={perm_test['null_std']:.4f}\n\n")
        
        f.write("### H3: Ours subset 是否显著优于 random coverage？\n\n")
        if "subset_comparison" in analysis_results:
            ours_data = None
            for method_data in analysis_results["subset_comparison"]:
                if method_data["method"] == "Ours" and "bootstrap" in method_data:
                    ours_data = method_data
                    break
            
            if ours_data:
                bootstrap = ours_data["bootstrap"]
                better_fraction = bootstrap.get("mean_nearest", {}).get("better_than_random_fraction", 0.5)
                
                if better_fraction >= 0.95:
                    h3_status = "SUPPORTED"
                elif better_fraction >= 0.4:
                    h3_status = "WEAK"
                else:
                    h3_status = "NOT SUPPORTED"
                
                f.write(f"Status: **{h3_status}**\n\n")
                f.write(f"Evidence:\n")
                f.write(f"- Better than random fraction: {better_fraction:.4f}\n")
                f.write(f"- Observed mean nearest: {bootstrap['mean_nearest']['observed']:.4f}\n")
                f.write(f"- Random mean: {bootstrap['mean_nearest']['bootstrap_mean']:.4f}\n")
            else:
                f.write("Status: **NOT EVALUATED** (no Ours subset bootstrap results)\n\n")
        else:
            f.write("Status: **NOT EVALUATED** (no subsets provided)\n\n")
        
        f.write("\n## Phase representation warning\n\n")
        f.write("当前 embedding 主要验证 initial/pre-grasp representation，\n")
        f.write("不能证明 transport/place phase representation 充分。\n\n")
        f.write("建议后续工作：\n")
        f.write("1. 提取不同 phase 的 embedding 并分别分析\n")
        f.write("2. 验证 embedding 是否编码 transport/place phase 信息\n")
        f.write("3. 分析 phase representation 与 task success 的关系\n\n")
        
        if "spearman" in analysis_results:
            spearman_global_rho = analysis_results["spearman"]["global"]["rho"]
            if spearman_global_rho > 0.5:
                f.write("\n## Warning: Ours may mainly encode initial object position\n\n")
                f.write("Evidence suggests: Ours global representation may mainly encode initial object position.\n\n")
                f.write("观察到 global embedding distance 与 XY distance 高度相关 (rho={:.4f})。\n".format(spearman_global_rho))
                f.write("如果 Ours 与 Uniform coverage 很接近，\n")
                f.write("则 Ours 可能主要是在重新实现 position-based sampling。\n")
    
    print(f"  Report saved to: {report_path}")


def load_subset(subset_path: str) -> List[int]:
    """Load subset from JSON file"""
    with open(subset_path) as f:
        data = json.load(f)
    
    if "selected_episode_indices" in data:
        return data["selected_episode_indices"]
    elif "episodes" in data:
        return data["episodes"]
    else:
        raise ValueError(f"Unknown subset format in {subset_path}")


def main():
    parser = argparse.ArgumentParser(description="Dataset Embedding Analysis")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--embeddings-dir", type=str, required=True)
    parser.add_argument("--random-subset", type=str, default=None)
    parser.add_argument("--uniform-subset", type=str, default=None)
    parser.add_argument("--ours-subset", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-wrist", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    embeddings_dir = Path(args.embeddings_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Dataset Embedding Analysis")
    print(f"{'='*60}")
    
    print(f"\n[1/12] Loading embeddings...")
    embeddings = load_embeddings(embeddings_dir)
    print(f"  Loaded {len(embeddings)} embeddings")
    
    print(f"\n[2/12] Loading episode metadata...")
    metadata = load_episode_metadata(dataset_root)
    print(f"  Loaded {len(metadata)} episodes metadata")
    
    print(f"\n[3/12] Aligning embeddings with metadata...")
    episode_indices, phi_globals, phi_wrists, obj_init_positions, goal_positions = \
        align_embeddings_with_metadata(embeddings, metadata)
    print(f"  Aligned {len(episode_indices)} episodes")
    
    phi_combined = compute_combined_embeddings(phi_globals, phi_wrists, args.lambda_wrist)
    
    print(f"\n[4/12] Checking embedding validity...")
    validation = check_embeddings_valid(phi_globals, phi_wrists)
    print(f"  Valid: {validation['valid']}")
    if validation['errors']:
        print(f"  Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")
    
    print(f"\n[5/12] Computing embedding statistics...")
    global_stats = compute_embedding_statistics(phi_globals, "global")
    wrist_stats = compute_embedding_statistics(phi_wrists, "wrist")
    combined_stats = compute_embedding_statistics(phi_combined, "combined")
    
    print(f"  Global effective rank: {global_stats['effective_rank']:.2f}")
    print(f"  Wrist effective rank: {wrist_stats['effective_rank']:.2f}")
    print(f"  Combined effective rank: {combined_stats['effective_rank']:.2f}")
    
    print(f"\n[6/12] Computing Spearman correlation...")
    phys_dist = pairwise_distances(obj_init_positions[:, :2])
    global_dist = pairwise_distances(phi_globals)
    wrist_dist = pairwise_distances(phi_wrists)
    combined_dist = pairwise_distances(phi_combined)
    
    spearman_results = {}
    for name, emb_dist in [("global", global_dist), ("wrist", wrist_dist), ("combined", combined_dist)]:
        rho, p_value, n_pairs = compute_spearman_correlation(phys_dist, emb_dist, seed=args.seed)
        spearman_results[name] = {"rho": rho, "p_value": p_value, "n_pairs": n_pairs}
        print(f"  {name}: rho={rho:.4f}, p={p_value:.2e}")
    
    print(f"\n[7/12] Running permutation test...")
    perm_test = permutation_test_spearman(phys_dist, global_dist, n_permutations=1000, seed=args.seed)
    print(f"  Observed rho: {perm_test['observed_rho']:.4f}")
    print(f"  Permutation p-value: {perm_test['permutation_p_value']:.4f}")
    
    print(f"\n[8/12] Running position probe...")
    probe_results = {}
    for name, phi in [("global", phi_globals), ("wrist", phi_wrists), ("combined", phi_combined)]:
        print(f"  Probing {name}...")
        probe = position_probe(phi, obj_init_positions[:, :2], seed=args.seed)
        probe_results[name] = probe
        print(f"    {name} Ridge R2_x: {probe['ridge']['R2_x']:.4f}, R2_y: {probe['ridge']['R2_y']:.4f}")
    
    print(f"\n[9/12] Analyzing neighbor overlap...")
    overlap_results = {}
    for name, phi in [("global", phi_globals), ("wrist", phi_wrists), ("combined", phi_combined)]:
        overlap = neighbor_overlap_analysis(phi, obj_init_positions[:, :2], seed=args.seed)
        overlap_results[name] = overlap
        print(f"  {name} overlap@10: {overlap['neighbor_overlap@10']:.4f}")
    
    print(f"\n[10/12] Analyzing grid classifiability...")
    grid_results = {}
    for name, phi in [("global", phi_globals), ("wrist", phi_wrists), ("combined", phi_combined)]:
        grid = grid_classifiability(phi, obj_init_positions[:, :2], seed=args.seed)
        grid_results[name] = grid
        print(f"  {name} 7x4 accuracy: {grid.get('7x4', {}).get('accuracy', 'N/A')}")
    
    print(f"\n[11/12] Generating visualizations...")
    generate_visualizations(phi_globals, phi_wrists, phi_combined, obj_init_positions, output_dir)
    
    analysis_results = {
        "n_episodes": len(episode_indices),
        "validation": validation,
        "global_stats": global_stats,
        "wrist_stats": wrist_stats,
        "combined_stats": combined_stats,
        "spearman": spearman_results,
        "permutation_test": perm_test,
        "probe": probe_results,
        "neighbor_overlap": overlap_results,
        "grid_classifiability": grid_results,
    }
    
    print(f"\n[12/12] Computing subset comparison...")
    subset_comparison = []
    
    subsets_to_analyze = []
    if args.random_subset:
        subsets_to_analyze.append(("Random", args.random_subset))
    if args.uniform_subset:
        subsets_to_analyze.append(("Uniform", args.uniform_subset))
    if args.ours_subset:
        subsets_to_analyze.append(("Ours", args.ours_subset))
    
    for method_name, subset_path in subsets_to_analyze:
        print(f"\n  Analyzing {method_name} subset...")
        try:
            subset_episodes = load_subset(subset_path)
            subset_indices = match_subset_to_indices(subset_episodes, episode_indices)
            
            if not subset_indices:
                print(f"    WARNING: No matching episodes for {method_name}")
                continue
            
            subset_phi_g = phi_globals[subset_indices]
            subset_phi_w = phi_wrists[subset_indices]
            subset_phi_c = phi_combined[subset_indices]
            subset_positions = obj_init_positions[subset_indices]
            
            coverage_g = compute_subset_coverage(phi_globals, subset_indices, obj_init_positions, f"{method_name}_global")
            coverage_w = compute_subset_coverage(phi_wrists, subset_indices, obj_init_positions, f"{method_name}_wrist")
            coverage_c = compute_subset_coverage(phi_combined, subset_indices, obj_init_positions, f"{method_name}_combined")
            
            redundancy_g = compute_subset_redundancy(subset_phi_g, phi_globals, f"{method_name}_global")
            redundancy_w = compute_subset_redundancy(subset_phi_w, phi_wrists, f"{method_name}_wrist")
            redundancy_c = compute_subset_redundancy(subset_phi_c, phi_combined, f"{method_name}_combined")
            
            sic_g = compute_fixed_universe_sic_for_subset(
                subset_indices, episode_indices, phi_globals, phi_wrists,
                f"{method_name}_global", args.alpha, args.lambda_wrist
            )
            
            workspace_cov = compute_workspace_coverage(obj_init_positions, subset_indices, method_name)
            
            bootstrap = random_bootstrap_analysis(
                phi_globals, subset_indices,
                n_bootstrap=args.n_bootstrap, seed=args.seed
            )
            
            method_results = {
                "method": method_name,
                "subset_size": len(subset_indices),
                "coverage_global": coverage_g,
                "coverage_wrist": coverage_w,
                "coverage_combined": coverage_c,
                "redundancy_global": redundancy_g,
                "redundancy_wrist": redundancy_w,
                "redundancy_combined": redundancy_c,
                "fixed_sic": sic_g,
                "workspace_coverage": workspace_cov,
                "bootstrap": bootstrap,
            }
            
            subset_comparison.append(method_results)
            
            print(f"    {method_name} coverage (global): {coverage_g['mean_nearest_distance']:.4f}" if coverage_g else "    No coverage")
            print(f"    {method_name} redundancy (global): {redundancy_g['redundancy_fraction']:.4f}" if redundancy_g else "    No redundancy")
            print(f"    {method_name} fixed SIC: {sic_g['normalized_sic']:.4f}" if sic_g else "    No SIC")
            if bootstrap:
                print(f"    {method_name} better than random: {bootstrap['mean_nearest']['better_than_random_fraction']:.4f}")
            
        except Exception as e:
            print(f"    ERROR analyzing {method_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if subset_comparison:
        analysis_results["subset_comparison"] = subset_comparison
    
    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    print(f"\n  Summary saved to: {summary_path}")
    
    metrics_path = output_dir / "embedding_metrics.csv"
    with open(metrics_path, 'w') as f:
        f.write("representation,dimension,effective_rank,norm_mean,norm_std,pairwise_dist_mean,spearman_rho,spearman_p_value\n")
        for rep, stats in [("global", global_stats), ("wrist", wrist_stats), ("combined", combined_stats)]:
            spearman = spearman_results[rep]
            f.write(f"{rep},{stats['dimension']},{stats['effective_rank']:.4f},{stats['norm_mean']:.4f},{stats['norm_std']:.4f},{stats['pairwise_dist_mean']:.4f},{spearman['rho']:.4f},{spearman['p_value']:.2e}\n")
    print(f"  Metrics saved to: {metrics_path}")
    
    if subset_comparison:
        coverage_path = output_dir / "coverage_comparison.csv"
        with open(coverage_path, 'w') as f:
            f.write("method,subset_size,mean_cover_dist_global,p95_global,max_radius_global,redundancy_global,fixed_sic_global,better_than_random\n")
            for method_data in subset_comparison:
                cov = method_data.get('coverage_global', {})
                red = method_data.get('redundancy_global', {})
                sic = method_data.get('fixed_sic', {})
                boot = method_data.get('bootstrap', {})
                
                mean_cover = cov.get('mean_nearest_distance', 0)
                p95 = cov.get('p95_nearest_distance', 0)
                max_radius = cov.get('max_nearest_distance', 0)
                redundancy = red.get('redundancy_fraction', 0)
                fixed_sic = sic.get('normalized_sic', 0)
                better_frac = boot.get('mean_nearest', {}).get('better_than_random_fraction', 0)
                
                f.write(f"{method_data['method']},{method_data['subset_size']},{mean_cover:.4f},{p95:.4f},{max_radius:.4f},{redundancy:.4f},{fixed_sic:.4f},{better_frac:.4f}\n")
        print(f"  Coverage comparison saved to: {coverage_path}")
    
    generate_report(analysis_results, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()