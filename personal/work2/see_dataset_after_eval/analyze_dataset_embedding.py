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
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent))

from sic_v2 import FixedAnchorSIC, check_embeddings_valid, compute_dbar_from_embeddings, build_kernel_matrices
from analysis_utils import compute_fixed_universe_sic


def load_embeddings(embeddings_dir: Path) -> Tuple[Dict[int, Dict], Dict]:
    """Load embedding cache with duplicate detection"""
    embeddings = {}
    duplicate_episode_indices = []
    invalid_files = []
    file_count = 0

    for f in sorted(embeddings_dir.glob("*.npy")):
        try:
            data = np.load(f, allow_pickle=True).item()
            ep_idx = data.get("episode_index")
            if ep_idx is not None:
                file_count += 1
                if ep_idx in embeddings:
                    duplicate_episode_indices.append(ep_idx)
                    print(f"  WARNING: Duplicate episode_index {ep_idx} in {f.name}")
                embeddings[ep_idx] = {
                    "phi_global": data["phi_global"],
                    "phi_wrist": data["phi_wrist"]
                }
        except Exception as e:
            invalid_files.append(f.name)
            print(f"  Skip invalid file: {f.name} ({e})")

    load_info = {
        "file_count": file_count,
        "duplicate_episode_indices": duplicate_episode_indices,
        "invalid_files": invalid_files,
    }
    return embeddings, load_info


def load_episode_metadata(dataset_root: Path) -> Tuple[Dict[int, Dict], Dict]:
    """Load episode_initial_states.json with duplicate detection"""
    metadata_file = dataset_root / "episode_initial_states.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_file) as f:
        data = json.load(f)

    episodes = {}
    duplicate_indices = []
    for ep in data["episodes"]:
        ep_idx = ep["episode_index"]
        if ep_idx in episodes:
            duplicate_indices.append(ep_idx)
            print(f"  WARNING: Duplicate episode_index {ep_idx} in metadata")
        episodes[ep_idx] = {
            "obj_init_pos": np.array(ep["obj_init_pos"]),
            "goal_pos": np.array(ep.get("goal_pos", ep.get("goal_pose", [0, 0, 0]))),
        }

    meta_info = {"duplicate_episode_indices": duplicate_indices}
    return episodes, meta_info


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

    dup_ep = set(embeddings.keys()) & set(metadata.keys())
    dup_in_intersection = [ep for ep in intersection if ep in dup_ep]

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

    observed_percentile = float(np.mean(null_rhos <= observed_rho))
    upper_tail_fraction = float(np.mean(null_rhos >= observed_rho))
    permutation_p_value = float((np.sum(np.abs(null_rhos) >= np.abs(observed_rho)) + 1) / (len(null_rhos) + 1))

    return {
        "observed_rho": float(observed_rho),
        "null_mean": float(np.mean(null_rhos)),
        "null_std": float(np.std(null_rhos)),
        "observed_percentile": observed_percentile,
        "upper_tail_fraction": upper_tail_fraction,
        "permutation_p_value": permutation_p_value,
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
    Shuffled baseline uses permuted labels with same CV splits.
    """
    rng = np.random.RandomState(seed)

    results = {
        "ridge": {"R2_x": 0.0, "R2_y": 0.0, "MAE_x": 0.0, "MAE_y": 0.0},
        "knn": {"R2_x": 0.0, "R2_y": 0.0, "MAE_x": 0.0, "MAE_y": 0.0},
        "shuffled_ridge": {"R2_x": [], "R2_y": [], "MAE_x": [], "MAE_y": []},
        "shuffled_knn": {"R2_x": [], "R2_y": [], "MAE_x": [], "MAE_y": []}
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    n = len(phi)

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
            perm = rng.permutation(n)
            y_perm = y[perm]

            ridge_shuf_preds = np.zeros(n)
            knn_shuf_preds = np.zeros(n)

            for train_idx, test_idx in kf.split(phi):
                X_train, X_test = phi[train_idx], phi[test_idx]
                y_train_perm = y_perm[train_idx]

                ridge_model = Ridge(alpha=1.0).fit(X_train, y_train_perm)
                ridge_shuf_preds[test_idx] = ridge_model.predict(X_test)

                knn_model = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train_perm)
                knn_shuf_preds[test_idx] = knn_model.predict(X_test)

            y_perm_test = y_perm
            results["shuffled_ridge"][f"R2_{dim_name}"].append(float(r2_score(y_perm_test, ridge_shuf_preds)))
            results["shuffled_ridge"][f"MAE_{dim_name}"].append(float(mean_absolute_error(y_perm_test, ridge_shuf_preds)))
            results["shuffled_knn"][f"R2_{dim_name}"].append(float(r2_score(y_perm_test, knn_shuf_preds)))
            results["shuffled_knn"][f"MAE_{dim_name}"].append(float(mean_absolute_error(y_perm_test, knn_shuf_preds)))

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
    Grid-based classification analysis with sparse-cell handling.
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
        min_class_count = int(np.min(cell_counts[cell_counts > 0])) if np.any(cell_counts > 0) else 0

        if empty_cells > 0:
            print(f"  WARNING: {grid_x}x{grid_y} grid has {empty_cells} empty cells")

        knn = KNeighborsClassifier(n_neighbors=5)

        if min_class_count < 2:
            results[f"{grid_x}x{grid_y}"] = {
                "accuracy": None,
                "accuracy_std": None,
                "balanced_accuracy": None,
                "balanced_accuracy_std": None,
                "shuffled_accuracy_mean": None,
                "shuffled_accuracy_std": None,
                "empty_cells": empty_cells,
                "sparse_cells": sparse_cells,
                "total_cells": grid_x * grid_y,
                "status": "insufficient_samples",
                "min_class_count": min_class_count,
            }
            continue

        n_splits = min(5, min_class_count)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        try:
            accuracies = cross_val_score(knn, phi, grid_ids, cv=cv, scoring='accuracy')
            balanced_accuracies = cross_val_score(knn, phi, grid_ids, cv=cv, scoring='balanced_accuracy')
        except Exception as e:
            print(f"  WARNING: Grid classification failed for {grid_x}x{grid_y}: {e}")
            results[f"{grid_x}x{grid_y}"] = {
                "accuracy": None,
                "accuracy_std": None,
                "balanced_accuracy": None,
                "balanced_accuracy_std": None,
                "shuffled_accuracy_mean": None,
                "shuffled_accuracy_std": None,
                "empty_cells": empty_cells,
                "sparse_cells": sparse_cells,
                "total_cells": grid_x * grid_y,
                "status": "cv_failed",
                "error": str(e),
            }
            continue

        shuffled_accs = []
        for _ in range(n_shuffles):
            grid_ids_shuffled = grid_ids[rng.permutation(len(grid_ids))]
            try:
                acc = np.mean(cross_val_score(knn, phi, grid_ids_shuffled, cv=cv, scoring='accuracy'))
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
            "total_cells": grid_x * grid_y,
            "status": "ok",
            "n_splits": n_splits,
            "min_class_count": min_class_count,
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
    Returns both all-reference and unselected-only metrics.
    """
    if not subset_indices:
        return None

    n_total = len(all_phi)
    subset_set = set(subset_indices)
    unselected = [i for i in range(n_total) if i not in subset_set]

    if not unselected:
        return None

    dist_matrix = pairwise_distances(all_phi)

    nearest_all = dist_matrix[:, subset_indices].min(axis=1)

    nearest_unselected = nearest_all[unselected]

    phys_dist_matrix = pairwise_distances(all_positions[:, :2])
    phys_nearest_all = phys_dist_matrix[:, subset_indices].min(axis=1)
    phys_nearest_unselected = phys_nearest_all[unselected]

    coverage = {
        "method": name,
        "all_mean_nearest_distance": float(np.mean(nearest_all)),
        "all_p95_nearest_distance": float(np.percentile(nearest_all, 95)),
        "all_max_nearest_distance": float(np.max(nearest_all)),
        "unselected_mean_nearest_distance": float(np.mean(nearest_unselected)),
        "unselected_p95_nearest_distance": float(np.percentile(nearest_unselected, 95)),
        "unselected_max_nearest_distance": float(np.max(nearest_unselected)),
        "physical_all_mean_nearest": float(np.mean(phys_nearest_all)),
        "physical_all_p95": float(np.percentile(phys_nearest_all, 95)),
        "physical_all_max_radius": float(np.max(phys_nearest_all)),
        "physical_unselected_mean_nearest": float(np.mean(phys_nearest_unselected)),
        "physical_unselected_p95": float(np.percentile(phys_nearest_unselected, 95)),
        "physical_unselected_max_radius": float(np.max(phys_nearest_unselected)),
        "subset_size": len(subset_indices),
        "unselected_size": len(unselected),
    }

    return coverage


def compute_subset_redundancy(
    subset_phi: np.ndarray,
    full_dataset_phi: np.ndarray,
    name: str,
    full_p05_threshold: float = None
) -> Dict:
    """
    Compute redundancy metrics for a subset.
    """
    if len(subset_phi) < 2:
        return None

    dist_matrix = pairwise_distances(subset_phi)
    np.fill_diagonal(dist_matrix, np.inf)

    nearest_dists = np.min(dist_matrix, axis=1)

    if full_p05_threshold is None:
        full_dist = pairwise_distances(full_dataset_phi)
        upper_tri = full_dist[np.triu_indices(len(full_dataset_phi), k=1)]
        p05_threshold = float(np.percentile(upper_tri, 5))
    else:
        p05_threshold = full_p05_threshold

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
    seed: int = 42,
    full_p05_threshold: float = None,
    K_global: np.ndarray = None,
    K_wrist: np.ndarray = None,
    dbar_global: float = None,
    dbar_wrist: float = None,
    n_episodes_total: int = None,
) -> Dict:
    """
    Random bootstrap baseline analysis with coverage, redundancy, and fixed SIC.
    Vectorized for performance.
    """
    rng = np.random.RandomState(seed)
    n_total = len(all_phi)
    subset_size = len(subset_indices)

    if subset_size == 0 or subset_size >= n_total:
        return None

    dist_matrix = pairwise_distances(all_phi)

    full_upper_tri = dist_matrix[np.triu_indices(n_total, k=1)]
    if full_p05_threshold is None:
        full_p05_threshold = float(np.percentile(full_upper_tri, 5))

    bootstrap_metrics = {
        "mean_nearest": [],
        "p95_nearest": [],
        "max_radius": [],
        "redundancy_fraction": [],
    }

    if K_global is not None and K_wrist is not None and dbar_global is not None and dbar_wrist is not None:
        compute_sic = True
        bootstrap_metrics["normalized_fixed_sic"] = []
    else:
        compute_sic = False

    print(f"  Running {n_bootstrap} bootstrap iterations...")
    for boot_i in range(n_bootstrap):
        sample_indices = rng.choice(n_total, subset_size, replace=False)

        nearest_dists = dist_matrix[:, sample_indices].min(axis=1)

        unselected_mask = np.ones(n_total, dtype=bool)
        unselected_mask[sample_indices] = False
        nearest_unselected = nearest_dists[unselected_mask]

        bootstrap_metrics["mean_nearest"].append(float(np.mean(nearest_unselected)))
        bootstrap_metrics["p95_nearest"].append(float(np.percentile(nearest_unselected, 95)))
        bootstrap_metrics["max_radius"].append(float(np.max(nearest_unselected)))

        sub_matrix = dist_matrix[np.ix_(sample_indices, sample_indices)]
        np.fill_diagonal(sub_matrix, np.inf)
        nn_dists = sub_matrix.min(axis=1)
        redundancy_frac = float(np.mean(nn_dists < full_p05_threshold))
        bootstrap_metrics["redundancy_fraction"].append(redundancy_frac)

        if compute_sic:
            sample_episodes = list(range(n_episodes_total))
            selected_episodes = [sample_episodes[i] for i in sample_indices]
            all_episode_indices = list(range(n_episodes_total))

            from analysis_utils import compute_fixed_universe_sic_from_indices
            sic_result = compute_fixed_universe_sic_from_indices(
                selected_episodes=selected_episodes,
                all_episode_indices=all_episode_indices,
                K_global=K_global,
                K_wrist=K_wrist,
                dbar_global=dbar_global,
                dbar_wrist=dbar_wrist,
                alpha=1.0,
                lambda_wrist=1.0,
            )
            bootstrap_metrics["normalized_fixed_sic"].append(sic_result["normalized_sic"])

        if (boot_i + 1) % 200 == 0:
            print(f"    Bootstrap {boot_i + 1}/{n_bootstrap}")

    subset_set = set(subset_indices)
    unselected = [i for i in range(n_total) if i not in subset_set]
    subset_nearest_unselected = dist_matrix[:, subset_indices].min(axis=1)[unselected]

    sub_matrix_obs = dist_matrix[np.ix_(subset_indices, subset_indices)]
    np.fill_diagonal(sub_matrix_obs, np.inf)
    nn_dists_obs = sub_matrix_obs.min(axis=1)
    observed_redundancy = float(np.mean(nn_dists_obs < full_p05_threshold))

    results = {}
    for metric in ["mean_nearest", "p95_nearest", "max_radius", "redundancy_fraction"]:
        bootstrap_vals = np.array(bootstrap_metrics[metric])

        if metric == "max_radius":
            observed = float(np.max(subset_nearest_unselected))
        elif metric == "mean_nearest":
            observed = float(np.mean(subset_nearest_unselected))
        elif metric == "p95_nearest":
            observed = float(np.percentile(subset_nearest_unselected, 95))
        elif metric == "redundancy_fraction":
            observed = observed_redundancy

        if metric in ["mean_nearest", "p95_nearest", "max_radius", "redundancy_fraction"]:
            lower_better = metric in ["mean_nearest", "p95_nearest", "max_radius", "redundancy_fraction"]
            if lower_better:
                better_fraction = float(np.sum(bootstrap_vals > observed) / len(bootstrap_vals))
            else:
                better_fraction = float(np.sum(bootstrap_vals < observed) / len(bootstrap_vals))

        percentile = float(np.sum(bootstrap_vals <= observed) / len(bootstrap_vals))

        results[metric] = {
            "observed": observed,
            "bootstrap_mean": float(np.mean(bootstrap_vals)),
            "bootstrap_std": float(np.std(bootstrap_vals)),
            "better_than_random_fraction": better_fraction,
            "percentile": percentile,
        }

    if compute_sic:
        bootstrap_sic = np.array(bootstrap_metrics["normalized_fixed_sic"])

        if K_global is not None:
            all_episode_indices = list(range(n_episodes_total))
            selected_episodes = list(subset_indices)
            from analysis_utils import compute_fixed_universe_sic_from_indices
            sic_obs = compute_fixed_universe_sic_from_indices(
                selected_episodes=selected_episodes,
                all_episode_indices=all_episode_indices,
                K_global=K_global,
                K_wrist=K_wrist,
                dbar_global=dbar_global,
                dbar_wrist=dbar_wrist,
                alpha=1.0,
                lambda_wrist=1.0,
            )
            observed_sic = sic_obs["normalized_sic"]
        else:
            observed_sic = 0.0

        better_fraction_sic = float(np.sum(bootstrap_sic < observed_sic) / len(bootstrap_sic))

        results["normalized_fixed_sic"] = {
            "observed": observed_sic,
            "bootstrap_mean": float(np.mean(bootstrap_sic)),
            "bootstrap_std": float(np.std(bootstrap_sic)),
            "better_than_random_fraction": better_fraction_sic,
            "percentile": float(np.sum(bootstrap_sic <= observed_sic) / len(bootstrap_sic)),
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
    Returns both all-reference and unselected-only metrics.
    """
    if not subset_indices:
        return None

    results = {"method": name}

    phys_dist_matrix = pairwise_distances(all_positions[:, :2])

    n_total = len(all_positions)
    subset_set = set(subset_indices)
    unselected = [i for i in range(n_total) if i not in subset_set]

    if not unselected:
        return None

    nearest_all = phys_dist_matrix[:, subset_indices].min(axis=1)
    nearest_unselected = nearest_all[unselected]

    results["physical_all_mean_nearest"] = float(np.mean(nearest_all))
    results["physical_all_p95"] = float(np.percentile(nearest_all, 95))
    results["physical_all_max_radius"] = float(np.max(nearest_all))
    results["physical_unselected_mean_nearest"] = float(np.mean(nearest_unselected))
    results["physical_unselected_p95"] = float(np.percentile(nearest_unselected, 95))
    results["physical_unselected_max_radius"] = float(np.max(nearest_unselected))

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


def evaluate_h1(analysis_results: Dict) -> Dict:
    """
    Evaluate H1: Embedding 是否具有可区分性？
    Based on: validity/collapse, position probe vs shuffled, neighbor overlap vs random.
    """
    evidence = {}

    for rep in ["global", "wrist", "combined"]:
        rep_evidence = {"probe": False, "neighbor": False, "validity": False}

        probe = analysis_results.get("probe", {}).get(rep, {})
        if probe:
            ridge_r2_x = probe.get("ridge", {}).get("R2_x", 0)
            shuffled_r2_x_mean = probe.get("shuffled_ridge", {}).get("R2_x", {}).get("mean", 0)
            if ridge_r2_x > shuffled_r2_x_mean + 0.05:
                rep_evidence["probe"] = True

        overlap = analysis_results.get("neighbor_overlap", {}).get(rep, {})
        if overlap:
            real_overlap = overlap.get("neighbor_overlap@10", 0)
            random_overlap = overlap.get("random_neighbor_overlap@10", 0)
            if real_overlap > random_overlap + 0.001:
                rep_evidence["neighbor"] = True

        validation = analysis_results.get("validation", {})
        if validation.get("valid", False):
            rep_evidence["validity"] = True

        evidence[rep] = rep_evidence

    n_strong = 0
    for rep in ["global", "combined"]:
        rep_ev = evidence.get(rep, {})
        if sum(rep_ev.values()) >= 2:
            n_strong += 1

    if n_strong >= 2:
        status = "SUPPORTED"
    elif n_strong >= 1:
        status = "WEAK"
    else:
        status = "NOT SUPPORTED"

    return {
        "hypothesis": "H1",
        "status": status,
        "evidence": evidence,
        "description": "Embedding 是否具有可区分性？"
    }


def evaluate_h2(analysis_results: Dict) -> Dict:
    """
    Evaluate H2: Embedding distance 是否具有 task-state geometry？
    Based on: Spearman + permutation test, position probe vs shuffled, neighbor overlap vs random.
    """
    spearman_results = analysis_results.get("spearman", {})
    perm_tests = analysis_results.get("permutation_tests", {})
    probe_results = analysis_results.get("probe", {})
    overlap_results = analysis_results.get("neighbor_overlap", {})

    evidence = {}
    for rep in ["global", "wrist", "combined"]:
        rep_evidence = {"spearman_sig": False, "probe_sig": False, "neighbor_sig": False}

        spearman = spearman_results.get(rep, {})
        perm = perm_tests.get(rep, {})
        if perm.get("permutation_p_value", 1.0) < 0.05:
            rep_evidence["spearman_sig"] = True

        probe = probe_results.get(rep, {})
        if probe:
            ridge_r2_x = probe.get("ridge", {}).get("R2_x", 0)
            shuffled_r2_x_mean = probe.get("shuffled_ridge", {}).get("R2_x", {}).get("mean", 0)
            if ridge_r2_x > shuffled_r2_x_mean + 0.05:
                rep_evidence["probe_sig"] = True

        overlap = overlap_results.get(rep, {})
        if overlap:
            real_overlap = overlap.get("neighbor_overlap@10", 0)
            random_overlap = overlap.get("random_neighbor_overlap@10", 0)
            if real_overlap > random_overlap + 0.001:
                rep_evidence["neighbor_sig"] = True

        evidence[rep] = rep_evidence

    n_sig_spearman = sum(1 for rep in ["global", "wrist", "combined"] if evidence.get(rep, {}).get("spearman_sig", False))
    n_strong = sum(1 for rep in ["global", "combined"] if sum(evidence.get(rep, {}).values()) >= 2)

    if n_sig_spearman >= 2 and n_strong >= 1:
        status = "SUPPORTED"
    elif n_sig_spearman >= 1 or n_strong >= 1:
        status = "WEAK"
    else:
        status = "NOT SUPPORTED"

    return {
        "hypothesis": "H2",
        "status": status,
        "evidence": evidence,
        "n_sig_spearman": n_sig_spearman,
        "description": "Embedding distance 是否具有 task-state geometry？"
    }


def evaluate_h3(analysis_results: Dict) -> Dict:
    """
    Evaluate H3: Ours subset 是否显著优于 random coverage？
    Based on: bootstrap better_than_random_fraction for multiple metrics.
    """
    subset_comparison = analysis_results.get("subset_comparison", {})
    ours_data = subset_comparison.get("Ours", {})

    if not ours_data:
        return {
            "hypothesis": "H3",
            "status": "NOT EVALUATED",
            "evidence": {},
            "description": "Ours subset 是否显著优于 random coverage？"
        }

    core_metrics = []
    for rep in ["global", "wrist", "combined"]:
        bootstrap = ours_data.get(f"bootstrap_{rep}", {})
        if bootstrap:
            for metric in ["mean_nearest", "p95_nearest", "max_radius", "redundancy_fraction"]:
                metric_data = bootstrap.get(metric, {})
                btr = metric_data.get("better_than_random_fraction", None)
                if btr is not None:
                    core_metrics.append({
                        "representation": rep,
                        "metric": metric,
                        "better_than_random_fraction": btr,
                    })

            sic_data = bootstrap.get("normalized_fixed_sic", {})
            sic_btr = sic_data.get("better_than_random_fraction", None)
            if sic_btr is not None:
                core_metrics.append({
                    "representation": rep,
                    "metric": "normalized_fixed_sic",
                    "better_than_random_fraction": sic_btr,
                })

    if not core_metrics:
        return {
            "hypothesis": "H3",
            "status": "NOT EVALUATED",
            "evidence": {},
            "description": "Ours subset 是否显著优于 random coverage？"
        }

    n_strong = sum(1 for m in core_metrics if m["better_than_random_fraction"] >= 0.95)
    n_weak = sum(1 for m in core_metrics if 0.8 <= m["better_than_random_fraction"] < 0.95)
    n_random = sum(1 for m in core_metrics if 0.4 <= m["better_than_random_fraction"] <= 0.6)
    n_poor = sum(1 for m in core_metrics if m["better_than_random_fraction"] < 0.1)

    if n_strong >= 3 and n_poor == 0:
        status = "SUPPORTED"
    elif n_weak >= 2 and n_strong < 3:
        status = "WEAK"
    elif n_random > len(core_metrics) * 0.5 or n_strong < 2:
        status = "NOT SUPPORTED"
    else:
        status = "WEAK"

    return {
        "hypothesis": "H3",
        "status": status,
        "evidence": {
            "core_metrics": core_metrics,
            "n_strong": n_strong,
            "n_weak": n_weak,
            "n_random": n_random,
            "n_poor": n_poor,
        },
        "description": "Ours subset 是否显著优于 random coverage？"
    }


def evaluate_hypotheses(analysis_results: Dict) -> Dict:
    """Run all hypothesis evaluations"""
    h1 = evaluate_h1(analysis_results)
    h2 = evaluate_h2(analysis_results)
    h3 = evaluate_h3(analysis_results)

    return {
        "H1": h1,
        "H2": h2,
        "H3": h3,
    }


def generate_report(
    analysis_results: Dict,
    output_dir: Path
):
    """Generate analysis_report.md"""
    report_path = output_dir / "analysis_report.md"

    hypothesis_eval = analysis_results.get("hypothesis_evaluation", {})

    with open(report_path, 'w') as f:
        f.write("# Dataset Embedding Analysis Report\n\n")

        f.write("## Summary\n\n")
        f.write(f"- Total episodes: {analysis_results['n_episodes']}\n")
        f.write(f"- Global dimension: {analysis_results['global_stats']['dimension']}\n")
        f.write(f"- Wrist dimension: {analysis_results['wrist_stats']['dimension']}\n")
        f.write(f"- Combined dimension: {analysis_results['combined_stats']['dimension']}\n\n")

        f.write("## Embedding Quality\n\n")
        validation = analysis_results.get('validation', {})
        f.write(f"- Valid: {validation.get('valid', 'N/A')}\n")
        stats = validation.get('stats', {})
        f.write(f"- Exact duplicates (global): {stats.get('n_exact_dup_global', 'N/A')}\n")
        f.write(f"- Exact duplicates (wrist): {stats.get('n_exact_dup_wrist', 'N/A')}\n")
        f.write(f"- Near duplicates (global): {stats.get('n_near_dup_global', 'N/A')}\n")
        f.write(f"- Near duplicates (wrist): {stats.get('n_near_dup_wrist', 'N/A')}\n")
        f.write(f"- Zero norm (global): {stats.get('zero_norm_global', 'N/A')}\n")
        f.write(f"- Zero norm (wrist): {stats.get('zero_norm_wrist', 'N/A')}\n\n")

        if validation.get('errors'):
            f.write("### Errors\n\n")
            for err in validation['errors']:
                f.write(f"- {err}\n")
            f.write("\n")

        if validation.get('warnings'):
            f.write("### Warnings\n\n")
            for warn in validation['warnings']:
                f.write(f"- {warn}\n")
            f.write("\n")

        f.write("## Table 1: Representation Quality\n\n")
        f.write("| Representation | Spearman XY | p-value | R2 x | R2 y | NN overlap@10 | Effective Rank |\n")
        f.write("|---------------|-------------|---------|------|------|---------------|----------------|\n")

        for rep in ["global", "wrist", "combined"]:
            stats = analysis_results.get(f"{rep}_stats", {})
            spearman = analysis_results.get("spearman", {}).get(rep, {})
            probe = analysis_results.get("probe", {}).get(rep, {})
            overlap = analysis_results.get("neighbor_overlap", {}).get(rep, {})

            spearman_rho = spearman.get('rho', 'N/A')
            spearman_p = spearman.get('p_value', 'N/A')
            r2_x = probe.get('ridge', {}).get('R2_x', 'N/A')
            r2_y = probe.get('ridge', {}).get('R2_y', 'N/A')
            nn_overlap = overlap.get('neighbor_overlap@10', 'N/A')
            eff_rank = stats.get('effective_rank', 'N/A')

            spearman_rho_str = f"{spearman_rho:.4f}" if isinstance(spearman_rho, float) else str(spearman_rho)
            spearman_p_str = f"{spearman_p:.2e}" if isinstance(spearman_p, float) else str(spearman_p)
            r2_x_str = f"{r2_x:.4f}" if isinstance(r2_x, float) else str(r2_x)
            r2_y_str = f"{r2_y:.4f}" if isinstance(r2_y, float) else str(r2_y)
            nn_overlap_str = f"{nn_overlap:.4f}" if isinstance(nn_overlap, float) else str(nn_overlap)
            eff_rank_str = f"{eff_rank:.2f}" if isinstance(eff_rank, float) else str(eff_rank)

            f.write(f"| {rep} | {spearman_rho_str} | {spearman_p_str} | "
                   f"{r2_x_str} | {r2_y_str} | "
                   f"{nn_overlap_str} | {eff_rank_str} |\n")

        f.write("\n## Table 2: Subset Coverage Comparison\n\n")
        f.write("| Method | Global Mean Cover | Global P95 | Global Max Radius | Wrist Mean Cover | Combined Mean Cover | Global Redundancy | Wrist Redundancy | Combined Redundancy | Fixed SIC | Global Random Better Fraction | Wrist Random Better Fraction | Combined Random Better Fraction |\n")
        f.write("|--------|------------------|------------|-------------------|-----------------|---------------------|-------------------|------------------|---------------------|-----------|------------------------------|-----------------------------|--------------------------------|\n")

        subset_comparison = analysis_results.get("subset_comparison", {})
        for method_name, method_data in subset_comparison.items():
            cov_g = method_data.get('coverage_global', {})
            cov_w = method_data.get('coverage_wrist', {})
            cov_c = method_data.get('coverage_combined', {})
            red_g = method_data.get('redundancy_global', {})
            red_w = method_data.get('redundancy_wrist', {})
            red_c = method_data.get('redundancy_combined', {})
            sic = method_data.get('fixed_sic', {})
            boot_g = method_data.get('bootstrap_global', {})
            boot_w = method_data.get('bootstrap_wrist', {})
            boot_c = method_data.get('bootstrap_combined', {})

            def fmt(val):
                return f"{val:.4f}" if isinstance(val, (int, float)) and val is not None else "N/A"

            mean_cover_g = cov_g.get('unselected_mean_nearest_distance', cov_g.get('mean_nearest_distance')) if cov_g else None
            p95_g = cov_g.get('unselected_p95_nearest_distance', cov_g.get('p95_nearest_distance')) if cov_g else None
            max_radius_g = cov_g.get('unselected_max_nearest_distance', cov_g.get('max_nearest_distance')) if cov_g else None
            mean_cover_w = cov_w.get('unselected_mean_nearest_distance', cov_w.get('mean_nearest_distance')) if cov_w else None
            mean_cover_c = cov_c.get('unselected_mean_nearest_distance', cov_c.get('mean_nearest_distance')) if cov_c else None
            redundancy_g = red_g.get('redundancy_fraction') if red_g else None
            redundancy_w = red_w.get('redundancy_fraction') if red_w else None
            redundancy_c = red_c.get('redundancy_fraction') if red_c else None
            fixed_sic = sic.get('normalized_sic') if sic else None
            btr_g = boot_g.get('mean_nearest', {}).get('better_than_random_fraction') if boot_g else None
            btr_w = boot_w.get('mean_nearest', {}).get('better_than_random_fraction') if boot_w else None
            btr_c = boot_c.get('mean_nearest', {}).get('better_than_random_fraction') if boot_c else None

            f.write(f"| {method_name} | {fmt(mean_cover_g)} | "
                   f"{fmt(p95_g)} | "
                   f"{fmt(max_radius_g)} | "
                   f"{fmt(mean_cover_w)} | "
                   f"{fmt(mean_cover_c)} | "
                   f"{fmt(redundancy_g)} | "
                   f"{fmt(redundancy_w)} | "
                   f"{fmt(redundancy_c)} | "
                   f"{fmt(fixed_sic)} | "
                   f"{fmt(btr_g)} | "
                   f"{fmt(btr_w)} | "
                   f"{fmt(btr_c)} |\n")

        f.write("\n## Hypothesis Evaluation\n\n")

        h1 = hypothesis_eval.get("H1", {})
        h2 = hypothesis_eval.get("H2", {})
        h3 = hypothesis_eval.get("H3", {})

        f.write(f"### H1: {h1.get('description', 'Embedding 是否具有可区分性？')}\n\n")
        f.write(f"Status: **{h1.get('status', 'NOT EVALUATED')}**\n\n")
        h1_evidence = h1.get('evidence', {})
        for rep in ["global", "wrist", "combined"]:
            rep_ev = h1_evidence.get(rep, {})
            f.write(f"- {rep}: probe={rep_ev.get('probe', False)}, neighbor={rep_ev.get('neighbor', False)}, validity={rep_ev.get('validity', False)}\n")
        f.write("\n")

        f.write(f"### H2: {h2.get('description', 'Embedding distance 是否具有 task-state geometry？')}\n\n")
        f.write(f"Status: **{h2.get('status', 'NOT EVALUATED')}**\n\n")
        f.write(f"- Significant Spearman reps: {h2.get('n_sig_spearman', 0)}\n")
        h2_evidence = h2.get('evidence', {})
        for rep in ["global", "wrist", "combined"]:
            rep_ev = h2_evidence.get(rep, {})
            f.write(f"- {rep}: spearman_sig={rep_ev.get('spearman_sig', False)}, probe_sig={rep_ev.get('probe_sig', False)}, neighbor_sig={rep_ev.get('neighbor_sig', False)}\n")
        f.write("\n")

        f.write(f"### H3: {h3.get('description', 'Ours subset 是否显著优于 random coverage？')}\n\n")
        f.write(f"Status: **{h3.get('status', 'NOT EVALUATED')}**\n\n")
        h3_evidence = h3.get('evidence', {})
        core_metrics = h3_evidence.get('core_metrics', [])
        if core_metrics:
            f.write("Core metrics better-than-random fractions:\n")
            for m in core_metrics:
                f.write(f"- {m['representation']} {m['metric']}: {m['better_than_random_fraction']:.4f}\n")
            f.write(f"\n- n_strong (>=0.95): {h3_evidence.get('n_strong', 0)}\n")
            f.write(f"- n_weak (>=0.8): {h3_evidence.get('n_weak', 0)}\n")
            f.write(f"- n_random (0.4-0.6): {h3_evidence.get('n_random', 0)}\n")
            f.write(f"- n_poor (<0.1): {h3_evidence.get('n_poor', 0)}\n")
        else:
            f.write("No Ours subset bootstrap results available.\n")
        f.write("\n")

        f.write("\n## Phase representation warning\n\n")
        f.write("当前 embedding 主要验证 initial/pre-grasp representation，\n")
        f.write("不能证明 transport/place phase representation 充分。\n\n")
        f.write("建议后续工作：\n")
        f.write("1. 提取不同 phase 的 embedding 并分别分析\n")
        f.write("2. 验证 embedding 是否编码 transport/place phase 信息\n")
        f.write("3. 分析 phase representation 与 task success 的关系\n\n")

        ours_uniform = analysis_results.get("ours_uniform_comparison", {})
        if ours_uniform:
            f.write("\n## Ours vs Uniform\n\n")
            f.write(f"- Episode overlap: {ours_uniform.get('episode_overlap_count', 'N/A')} / {ours_uniform.get('episode_overlap_ratio', 'N/A')}\n")
            f.write(f"- Workspace coverage delta: {ours_uniform.get('workspace_coverage_delta', 'N/A')}\n")
            f.write(f"- Global embedding mean cover delta: {ours_uniform.get('global_mean_cover_delta', 'N/A')}\n")
            f.write(f"- Fixed SIC delta: {ours_uniform.get('fixed_sic_delta', 'N/A')}\n\n")

            if ours_uniform.get("warning_position_encoding", False):
                f.write("**Warning:** Evidence suggests Ours may mainly reproduce initial-position coverage.\n\n")

        f.write("\n## WARNING: Training-time camera mismatch\n\n")
        f.write("train_and_eval_v2.sh 中的 training-time env eval camera 仍写死为 `corner`，\n")
        f.write("因此 corner2 / corner3 的内置 16-episode eval 不能作为最终 benchmark。\n\n")
        f.write("最终 benchmark 应继续使用 standalone `lerobot-eval`。\n")

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

    print(f"\n[1/15] Loading embeddings...")
    embeddings, load_info = load_embeddings(embeddings_dir)
    print(f"  Loaded {len(embeddings)} embeddings from {load_info['file_count']} files")
    if load_info['duplicate_episode_indices']:
        print(f"  WARNING: {len(load_info['duplicate_episode_indices'])} duplicate episode indices detected")

    print(f"\n[2/15] Loading episode metadata...")
    metadata, meta_info = load_episode_metadata(dataset_root)
    print(f"  Loaded {len(metadata)} episodes metadata")
    if meta_info['duplicate_episode_indices']:
        print(f"  WARNING: {len(meta_info['duplicate_episode_indices'])} duplicate episode indices in metadata")

    print(f"\n[3/15] Aligning embeddings with metadata...")
    episode_indices, phi_globals, phi_wrists, obj_init_positions, goal_positions = \
        align_embeddings_with_metadata(embeddings, metadata)
    print(f"  Aligned {len(episode_indices)} episodes")

    phi_combined = compute_combined_embeddings(phi_globals, phi_wrists, args.lambda_wrist)

    print(f"\n[4/15] Checking embedding validity...")
    validation = check_embeddings_valid(phi_globals, phi_wrists)
    print(f"  Valid: {validation['valid']}")
    if validation['errors']:
        print(f"  Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"  Warnings: {validation['warnings']}")

    print(f"\n[5/15] Computing embedding statistics...")
    global_stats = compute_embedding_statistics(phi_globals, "global")
    wrist_stats = compute_embedding_statistics(phi_wrists, "wrist")
    combined_stats = compute_embedding_statistics(phi_combined, "combined")

    norms_g = np.linalg.norm(phi_globals, axis=1, keepdims=True)
    norms_w = np.linalg.norm(phi_wrists, axis=1, keepdims=True)
    phi_global_normalized = phi_globals / (norms_g + 1e-10)
    phi_wrist_normalized = phi_wrists / (norms_w + 1e-10)

    global_norm_stats = compute_embedding_statistics(phi_global_normalized, "global_normalized")
    wrist_norm_stats = compute_embedding_statistics(phi_wrist_normalized, "wrist_normalized")

    print(f"  Global effective rank: {global_stats['effective_rank']:.2f}")
    print(f"  Wrist effective rank: {wrist_stats['effective_rank']:.2f}")
    print(f"  Combined effective rank: {combined_stats['effective_rank']:.2f}")

    print(f"\n[6/15] Computing Spearman correlation...")
    phys_dist = pairwise_distances(obj_init_positions[:, :2])
    global_dist = pairwise_distances(phi_globals)
    wrist_dist = pairwise_distances(phi_wrists)
    combined_dist = pairwise_distances(phi_combined)
    global_norm_dist = pairwise_distances(phi_global_normalized)
    wrist_norm_dist = pairwise_distances(phi_wrist_normalized)

    spearman_results = {}
    for name, emb_dist in [
        ("global", global_dist), ("wrist", wrist_dist), ("combined", combined_dist),
        ("global_normalized", global_norm_dist), ("wrist_normalized", wrist_norm_dist),
    ]:
        rho, p_value, n_pairs = compute_spearman_correlation(phys_dist, emb_dist, seed=args.seed)
        spearman_results[name] = {"rho": rho, "p_value": p_value, "n_pairs": n_pairs}
        print(f"  {name}: rho={rho:.4f}, p={p_value:.2e}")

    print(f"\n[7/15] Running permutation tests...")
    permutation_tests = {}
    for name, emb_dist in [("global", global_dist), ("wrist", wrist_dist), ("combined", combined_dist)]:
        print(f"  Permutation test for {name}...")
        perm = permutation_test_spearman(phys_dist, emb_dist, n_permutations=1000, seed=args.seed)
        permutation_tests[name] = perm
        print(f"    {name}: observed_rho={perm['observed_rho']:.4f}, p={perm['permutation_p_value']:.4f}")

    print(f"\n[8/15] Running position probe...")
    probe_results = {}
    for name, phi in [("global", phi_globals), ("wrist", phi_wrists), ("combined", phi_combined)]:
        print(f"  Position probe for {name}...")
        probe = position_probe(phi, obj_init_positions[:, :2], n_shuffles=100, seed=args.seed)
        probe_results[name] = probe
        print(f"    {name}: Ridge R2_x={probe['ridge']['R2_x']:.4f}, shuffled={probe['shuffled_ridge']['R2_x']['mean']:.4f}")

    print(f"\n[9/15] Running neighbor overlap analysis...")
    neighbor_overlap = {}
    for name, phi in [("global", phi_globals), ("wrist", phi_wrists), ("combined", phi_combined)]:
        print(f"  Neighbor overlap for {name}...")
        overlap = neighbor_overlap_analysis(phi, obj_init_positions[:, :2], ks=[5, 10, 20], seed=args.seed)
        neighbor_overlap[name] = overlap
        print(f"    {name}: overlap@10={overlap['neighbor_overlap@10']:.4f}, random@10={overlap['random_neighbor_overlap@10']:.4f}")

    print(f"\n[10/15] Running grid classifiability...")
    grid_results = grid_classifiability(phi_combined, obj_init_positions[:, :2], n_shuffles=100, seed=args.seed)
    for grid_name, res in grid_results.items():
        status = res.get("status", "unknown")
        acc = res.get("accuracy")
        if acc is not None:
            print(f"  {grid_name}: accuracy={acc:.4f}, status={status}")
        else:
            print(f"  {grid_name}: status={status}")

    print(f"\n[11/15] Preparing fixed universe for bootstrap...")
    dbar_global, dbar_wrist, dbar_fallback = compute_dbar_from_embeddings(phi_globals, phi_wrists)
    K_global, K_wrist = build_kernel_matrices(phi_globals, phi_wrists, dbar_global, dbar_wrist)
    print(f"  dbar_global={dbar_global:.6f}, dbar_wrist={dbar_wrist:.6f}")
    print(f"  K_global shape={K_global.shape}, K_wrist shape={K_wrist.shape}")

    full_dist_global = pairwise_distances(phi_globals)
    full_upper_tri_global = full_dist_global[np.triu_indices(len(phi_globals), k=1)]
    full_p05_global = float(np.percentile(full_upper_tri_global, 5))

    full_dist_wrist = pairwise_distances(phi_wrists)
    full_upper_tri_wrist = full_dist_wrist[np.triu_indices(len(phi_wrists), k=1)]
    full_p05_wrist = float(np.percentile(full_upper_tri_wrist, 5))

    full_dist_combined = pairwise_distances(phi_combined)
    full_upper_tri_combined = full_dist_combined[np.triu_indices(len(phi_combined), k=1)]
    full_p05_combined = float(np.percentile(full_upper_tri_combined, 5))

    print(f"\n[12/15] Running subset comparison...")
    subset_comparison = {}

    if args.random_subset:
        print(f"  Loading random subset from {args.random_subset}")
        random_episodes = load_subset(args.random_subset)
        random_indices = match_subset_to_indices(random_episodes, episode_indices)
        print(f"  Matched {len(random_indices)}/{len(random_episodes)} random episodes")

        if random_indices:
            phi_g_sub = phi_globals[random_indices]
            phi_w_sub = phi_wrists[random_indices]
            phi_c_sub = phi_combined[random_indices]

            cov_g = compute_subset_coverage(phi_globals, random_indices, obj_init_positions, "Random")
            cov_w = compute_subset_coverage(phi_wrists, random_indices, obj_init_positions, "Random")
            cov_c = compute_subset_coverage(phi_combined, random_indices, obj_init_positions, "Random")

            red_g = compute_subset_redundancy(phi_g_sub, phi_globals, "Random", full_p05_global)
            red_w = compute_subset_redundancy(phi_w_sub, phi_wrists, "Random", full_p05_wrist)
            red_c = compute_subset_redundancy(phi_c_sub, phi_combined, "Random", full_p05_combined)

            ws_cov = compute_workspace_coverage(obj_init_positions, random_indices, "Random")

            print(f"  Running bootstrap for Random (global)...")
            boot_g = random_bootstrap_analysis(
                phi_globals, random_indices, n_bootstrap=args.n_bootstrap, seed=args.seed,
                full_p05_threshold=full_p05_global,
                K_global=K_global, K_wrist=K_wrist,
                dbar_global=dbar_global, dbar_wrist=dbar_wrist,
                n_episodes_total=len(episode_indices)
            )
            print(f"  Running bootstrap for Random (wrist)...")
            boot_w = random_bootstrap_analysis(
                phi_wrists, random_indices, n_bootstrap=args.n_bootstrap, seed=args.seed,
                full_p05_threshold=full_p05_wrist,
                K_global=K_global, K_wrist=K_wrist,
                dbar_global=dbar_global, dbar_wrist=dbar_wrist,
                n_episodes_total=len(episode_indices)
            )
            print(f"  Running bootstrap for Random (combined)...")
            boot_c = random_bootstrap_analysis(
                phi_combined, random_indices, n_bootstrap=args.n_bootstrap, seed=args.seed,
                full_p05_threshold=full_p05_combined,
                K_global=K_global, K_wrist=K_wrist,
                dbar_global=dbar_global, dbar_wrist=dbar_wrist,
                n_episodes_total=len(episode_indices)
            )

            subset_comparison["Random"] = {
                "coverage_global": cov_g,
                "coverage_wrist": cov_w,
                "coverage_combined": cov_c,
                "redundancy_global": red_g,
                "redundancy_wrist": red_w,
                "redundancy_combined": red_c,
                "workspace_coverage": ws_cov,
                "bootstrap_global": boot_g,
                "bootstrap_wrist": boot_w,
                "bootstrap_combined": boot_c,
            }

    if args.uniform_subset:
        print(f"  Loading uniform subset from {args.uniform_subset}")
        uniform_episodes = load_subset(args.uniform_subset)
        uniform_indices = match_subset_to_indices(uniform_episodes, episode_indices)
        print(f"  Matched {len(uniform_indices)}/{len(uniform_episodes)} uniform episodes")

        if uniform_indices:
            phi_g_sub = phi_globals[uniform_indices]
            phi_w_sub = phi_wrists[uniform_indices]
            phi_c_sub = phi_combined[uniform_indices]

            cov_g = compute_subset_coverage(phi_globals, uniform_indices, obj_init_positions, "Uniform")
            cov_w = compute_subset_coverage(phi_wrists, uniform_indices, obj_init_positions, "Uniform")
            cov_c = compute_subset_coverage(phi_combined, uniform_indices, obj_init_positions, "Uniform")

            red_g = compute_subset_redundancy(phi_g_sub, phi_globals, "Uniform", full_p05_global)
            red_w = compute_subset_redundancy(phi_w_sub, phi_wrists, "Uniform", full_p05_wrist)
            red_c = compute_subset_redundancy(phi_c_sub, phi_combined, "Uniform", full_p05_combined)

            ws_cov = compute_workspace_coverage(obj_init_positions, uniform_indices, "Uniform")

            print(f"  Running bootstrap for Uniform (global)...")
            boot_g = random_bootstrap_analysis(
                phi_globals, uniform_indices, n_bootstrap=args.n_bootstrap, seed=args.seed,
                full_p05_threshold=full_p05_global,
                K_global=K_global, K_wrist=K_wrist,
                dbar_global=dbar_global, dbar_wrist=dbar_wrist,
                n_episodes_total=len(episode_indices)
            )
            print(f"  Running bootstrap for Uniform (wrist)...")
            boot_w = random_bootstrap_analysis(
                phi_wrists, uniform_indices, n_bootstrap=args.n_bootstrap, seed=args.seed,
                full_p05_threshold=full_p05_wrist,
                K_global=K_global, K_wrist=K_wrist,
                dbar_global=dbar_global, dbar_wrist=dbar_wrist,
                n_episodes_total=len(episode_indices)
            )
            print(f"  Running bootstrap for Uniform (combined)...")
            boot_c = random_bootstrap_analysis(
                phi_combined, uniform_indices, n_bootstrap=args.n_bootstrap, seed=args.seed,
                full_p05_threshold=full_p05_combined,
                K_global=K_global, K_wrist=K_wrist,
                dbar_global=dbar_global, dbar_wrist=dbar_wrist,
                n_episodes_total=len(episode_indices)
            )

            subset_comparison["Uniform"] = {
                "coverage_global": cov_g,
                "coverage_wrist": cov_w,
                "coverage_combined": cov_c,
                "redundancy_global": red_g,
                "redundancy_wrist": red_w,
                "redundancy_combined": red_c,
                "workspace_coverage": ws_cov,
                "bootstrap_global": boot_g,
                "bootstrap_wrist": boot_w,
                "bootstrap_combined": boot_c,
            }

    if args.ours_subset:
        print(f"  Loading ours subset from {args.ours_subset}")
        ours_episodes = load_subset(args.ours_subset)
        ours_indices = match_subset_to_indices(ours_episodes, episode_indices)
        print(f"  Matched {len(ours_indices)}/{len(ours_episodes)} ours episodes")

        if ours_indices:
            phi_g_sub = phi_globals[ours_indices]
            phi_w_sub = phi_wrists[ours_indices]
            phi_c_sub = phi_combined[ours_indices]

            cov_g = compute_subset_coverage(phi_globals, ours_indices, obj_init_positions, "Ours")
            cov_w = compute_subset_coverage(phi_wrists, ours_indices, obj_init_positions, "Ours")
            cov_c = compute_subset_coverage(phi_combined, ours_indices, obj_init_positions, "Ours")

            red_g = compute_subset_redundancy(phi_g_sub, phi_globals, "Ours", full_p05_global)
            red_w = compute_subset_redundancy(phi_w_sub, phi_wrists, "Ours", full_p05_wrist)
            red_c = compute_subset_redundancy(phi_c_sub, phi_combined, "Ours", full_p05_combined)

            ws_cov = compute_workspace_coverage(obj_init_positions, ours_indices, "Ours")

            print(f"  Running bootstrap for Ours (global)...")
            boot_g = random_bootstrap_analysis(
                phi_globals, ours_indices, n_bootstrap=args.n_bootstrap, seed=args.seed,
                full_p05_threshold=full_p05_global,
                K_global=K_global, K_wrist=K_wrist,
                dbar_global=dbar_global, dbar_wrist=dbar_wrist,
                n_episodes_total=len(episode_indices)
            )
            print(f"  Running bootstrap for Ours (wrist)...")
            boot_w = random_bootstrap_analysis(
                phi_wrists, ours_indices, n_bootstrap=args.n_bootstrap, seed=args.seed,
                full_p05_threshold=full_p05_wrist,
                K_global=K_global, K_wrist=K_wrist,
                dbar_global=dbar_global, dbar_wrist=dbar_wrist,
                n_episodes_total=len(episode_indices)
            )
            print(f"  Running bootstrap for Ours (combined)...")
            boot_c = random_bootstrap_analysis(
                phi_combined, ours_indices, n_bootstrap=args.n_bootstrap, seed=args.seed,
                full_p05_threshold=full_p05_combined,
                K_global=K_global, K_wrist=K_wrist,
                dbar_global=dbar_global, dbar_wrist=dbar_wrist,
                n_episodes_total=len(episode_indices)
            )

            subset_comparison["Ours"] = {
                "coverage_global": cov_g,
                "coverage_wrist": cov_w,
                "coverage_combined": cov_c,
                "redundancy_global": red_g,
                "redundancy_wrist": red_w,
                "redundancy_combined": red_c,
                "workspace_coverage": ws_cov,
                "bootstrap_global": boot_g,
                "bootstrap_wrist": boot_w,
                "bootstrap_combined": boot_c,
            }

    print(f"\n[13/15] Computing Ours vs Uniform comparison...")
    ours_uniform_comparison = {}
    if args.ours_subset and args.uniform_subset and "Ours" in subset_comparison and "Uniform" in subset_comparison:
        ours_eps = set(ours_episodes)
        uniform_eps = set(uniform_episodes)
        overlap_eps = ours_eps & uniform_eps
        overlap_count = len(overlap_eps)
        overlap_ratio = overlap_count / min(len(ours_eps), len(uniform_eps)) if min(len(ours_eps), len(uniform_eps)) > 0 else 0

        ours_data = subset_comparison["Ours"]
        uniform_data = subset_comparison["Uniform"]

        ws_delta = {}
        if ours_data.get("workspace_coverage") and uniform_data.get("workspace_coverage"):
            for key in ["physical_unselected_mean_nearest", "physical_unselected_p95", "physical_unselected_max_radius"]:
                v1 = ours_data["workspace_coverage"].get(key)
                v2 = uniform_data["workspace_coverage"].get(key)
                if v1 is not None and v2 is not None:
                    ws_delta[key] = v1 - v2

        global_delta = {}
        if ours_data.get("coverage_global") and uniform_data.get("coverage_global"):
            for key in ["unselected_mean_nearest_distance", "unselected_p95_nearest_distance", "unselected_max_nearest_distance"]:
                v1 = ours_data["coverage_global"].get(key)
                v2 = uniform_data["coverage_global"].get(key)
                if v1 is not None and v2 is not None:
                    global_delta[key] = v1 - v2

        sic_delta = None
        if ours_data.get("bootstrap_global") and uniform_data.get("bootstrap_global"):
            sic_ours = ours_data["bootstrap_global"].get("normalized_fixed_sic", {}).get("observed")
            sic_uniform = uniform_data["bootstrap_global"].get("normalized_fixed_sic", {}).get("observed")
            if sic_ours is not None and sic_uniform is not None:
                sic_delta = sic_ours - sic_uniform

        warning_position_encoding = False
        spearman_global = spearman_results.get("global", {}).get("rho", 0)
        if abs(spearman_global) > 0.3 and overlap_ratio > 0.5:
            warning_position_encoding = True

        ours_uniform_comparison = {
            "episode_overlap_count": overlap_count,
            "episode_overlap_ratio": overlap_ratio,
            "workspace_coverage_delta": ws_delta,
            "global_mean_cover_delta": global_delta,
            "fixed_sic_delta": sic_delta,
            "warning_position_encoding": warning_position_encoding,
        }

        print(f"  Ours vs Uniform overlap: {overlap_count} ({overlap_ratio:.4f})")
        print(f"  Warning position encoding: {warning_position_encoding}")

    print(f"\n[14/15] Generating visualizations...")
    generate_visualizations(phi_globals, phi_wrists, phi_combined, obj_init_positions, output_dir)

    print(f"\n[15/15] Saving results...")
    analysis_results = {
        "n_episodes": len(episode_indices),
        "global_stats": global_stats,
        "wrist_stats": wrist_stats,
        "combined_stats": combined_stats,
        "global_normalized_stats": global_norm_stats,
        "wrist_normalized_stats": wrist_norm_stats,
        "validation": validation,
        "load_info": load_info,
        "meta_info": meta_info,
        "spearman": spearman_results,
        "permutation_tests": permutation_tests,
        "probe": probe_results,
        "neighbor_overlap": neighbor_overlap,
        "grid_classifiability": grid_results,
        "subset_comparison": subset_comparison,
        "ours_uniform_comparison": ours_uniform_comparison,
    }

    analysis_results["hypothesis_evaluation"] = evaluate_hypotheses(analysis_results)

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    print(f"  Summary saved to: {summary_path}")

    metrics_path = output_dir / "embedding_metrics.csv"
    with open(metrics_path, 'w') as f:
        f.write("metric,global,wrist,combined,global_normalized,wrist_normalized\n")
        for key in ["dimension", "n_episodes", "norm_mean", "norm_std", "pairwise_dist_mean",
                    "pairwise_dist_std", "pairwise_dist_p05", "pairwise_dist_p50", "pairwise_dist_p95",
                    "pairwise_dist_max", "effective_rank"]:
            vals = []
            for rep in ["global", "wrist", "combined", "global_normalized", "wrist_normalized"]:
                stats = analysis_results.get(f"{rep}_stats", {})
                vals.append(str(stats.get(key, "N/A")))
            f.write(f"{key},{','.join(vals)}\n")
    print(f"  Metrics saved to: {metrics_path}")

    coverage_path = output_dir / "coverage_comparison.csv"
    with open(coverage_path, 'w') as f:
        f.write("method,subset_size,global_mean,global_p95,global_max,wrist_mean,wrist_p95,wrist_max,combined_mean,combined_p95,combined_max\n")
        for method_name, method_data in subset_comparison.items():
            cov_g = method_data.get("coverage_global", {})
            cov_w = method_data.get("coverage_wrist", {})
            cov_c = method_data.get("coverage_combined", {})
            subset_size = cov_g.get("subset_size", 0) if cov_g else 0

            def safe_get(d, k):
                return d.get(k, "N/A") if d else "N/A"

            f.write(f"{method_name},{subset_size},"
                   f"{safe_get(cov_g, 'unselected_mean_nearest_distance')},"
                   f"{safe_get(cov_g, 'unselected_p95_nearest_distance')},"
                   f"{safe_get(cov_g, 'unselected_max_nearest_distance')},"
                   f"{safe_get(cov_w, 'unselected_mean_nearest_distance')},"
                   f"{safe_get(cov_w, 'unselected_p95_nearest_distance')},"
                   f"{safe_get(cov_w, 'unselected_max_nearest_distance')},"
                   f"{safe_get(cov_c, 'unselected_mean_nearest_distance')},"
                   f"{safe_get(cov_c, 'unselected_p95_nearest_distance')},"
                   f"{safe_get(cov_c, 'unselected_max_nearest_distance')}\n")
    print(f"  Coverage comparison saved to: {coverage_path}")

    generate_report(analysis_results, output_dir)

    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"{'='*60}")
    print(f"  Output directory: {output_dir}")
    print(f"  Files generated:")
    print(f"    - analysis_summary.json")
    print(f"    - embedding_metrics.csv")
    print(f"    - coverage_comparison.csv")
    print(f"    - analysis_report.md")
    print(f"    - figures/pca_positions.png")
    print(f"    - figures/physical_vs_embedding_distance.png")
    print(f"    - figures/pairwise_distance_histogram.png")

    h1_status = analysis_results["hypothesis_evaluation"]["H1"]["status"]
    h2_status = analysis_results["hypothesis_evaluation"]["H2"]["status"]
    h3_status = analysis_results["hypothesis_evaluation"]["H3"]["status"]
    print(f"\n  Hypothesis Results:")
    print(f"    H1: {h1_status}")
    print(f"    H2: {h2_status}")
    print(f"    H3: {h3_status}")


if __name__ == "__main__":
    main()