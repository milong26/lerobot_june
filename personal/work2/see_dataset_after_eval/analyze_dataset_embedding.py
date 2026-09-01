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
import json
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
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).parent))

from sic_v2 import check_embeddings_valid, compute_dbar_from_embeddings, build_kernel_matrices
from analysis_utils import compute_fixed_universe_sic_from_indices


def load_subset_episode_indices(path: Path) -> List[int]:
    """
    Load subset episode indices from JSON file.
    Priority: selected_episode_indices > episode_indices > episodes
    Raises ValueError if no valid key found or if duplicates exist.
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        episodes = data
    elif isinstance(data, dict):
        episodes = None
        for key in ["selected_episode_indices", "episode_indices", "episodes"]:
            if key in data:
                episodes = data[key]
                break
        if episodes is None:
            raise ValueError(
                f"Subset JSON {path} does not contain a supported episode list key. "
                f"Expected one of: selected_episode_indices, episode_indices, episodes"
            )
    else:
        raise ValueError(f"Subset JSON {path} has unsupported format: {type(data)}")

    if not isinstance(episodes, list):
        raise ValueError(f"Episode list in {path} is not a list")

    int_episodes = [int(x) for x in episodes]

    seen = set()
    duplicates = []
    for ep in int_episodes:
        if ep in seen:
            duplicates.append(ep)
        seen.add(ep)
    if duplicates:
        raise ValueError(
            f"Subset JSON {path} contains {len(duplicates)} duplicate episode indices: {duplicates[:10]}{'...' if len(duplicates) > 10 else ''}"
        )

    return int_episodes


def load_subset_if_provided(subset_path: Optional[Path]) -> Optional[List[int]]:
    """
    Load a subset file only when explicitly provided.

    - subset_path is None -> method not requested, returns None (no analysis).
    - subset_path is not None but does not exist -> raise FileNotFoundError
      (never silently omit a requested method).
    """
    if subset_path is None:
        return None
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_path}")
    return load_subset_episode_indices(subset_path)


def l2_normalize_rows(phi: np.ndarray) -> np.ndarray:
    """L2 normalize each row of embedding matrix."""
    norms = np.linalg.norm(phi, axis=1, keepdims=True)
    return phi / np.maximum(norms, 1e-10)


def load_embeddings(embeddings_dir: Path) -> Tuple[Dict[int, Dict], Dict]:
    """Load embedding cache with duplicate detection"""
    embeddings = {}
    duplicate_episode_indices = []
    invalid_files = []
    npy_file_count = 0
    valid_record_count = 0

    for f in sorted(embeddings_dir.glob("*.npy")):
        npy_file_count += 1
        try:
            data = np.load(f, allow_pickle=True).item()
            ep_idx = data.get("episode_index")
            if ep_idx is None:
                invalid_files.append({"file": f.name, "reason": "missing episode_index"})
                print(f"  WARNING: Missing episode_index in {f.name}")
                continue
            valid_record_count += 1
            if ep_idx in embeddings:
                duplicate_episode_indices.append(ep_idx)
                print(f"  WARNING: Duplicate episode_index {ep_idx} in {f.name}")
            embeddings[ep_idx] = {
                "phi_global": data["phi_global"],
                "phi_wrist": data["phi_wrist"]
            }
        except Exception as e:
            invalid_files.append({"file": f.name, "reason": str(e)})
            print(f"  Skip invalid file: {f.name} ({e})")

    load_info = {
        "npy_file_count": npy_file_count,
        "valid_record_count": valid_record_count,
        "embedding_unique_episode_count": len(embeddings),
        "duplicate_episode_indices": duplicate_episode_indices,
        "invalid_files": invalid_files,
        "file_count": valid_record_count,
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
    raw_record_count = len(data["episodes"])
    for ep in data["episodes"]:
        ep_idx = ep["episode_index"]
        if ep_idx in episodes:
            duplicate_indices.append(ep_idx)
            print(f"  WARNING: Duplicate episode_index {ep_idx} in metadata")
        episodes[ep_idx] = {
            "obj_init_pos": np.array(ep["obj_init_pos"]),
            "goal_pos": np.array(ep.get("goal_pos", ep.get("goal_pose", [0, 0, 0]))),
        }

    meta_info = {"duplicate_episode_indices": duplicate_indices, "record_count": raw_record_count}
    return episodes, meta_info


def align_embeddings_with_metadata(
    embeddings: Dict[int, Dict],
    metadata: Dict[int, Dict],
    load_info: Optional[Dict] = None,
    meta_info: Optional[Dict] = None
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Align embeddings with metadata.

    Returns:
        episode_indices, phi_globals, phi_wrists, obj_init_positions, goal_positions, alignment_info
    """
    embedding_episodes = set(embeddings.keys())
    metadata_episodes = set(metadata.keys())

    intersection = embedding_episodes & metadata_episodes
    missing_embeddings = sorted(metadata_episodes - embedding_episodes)
    missing_metadata = sorted(embedding_episodes - metadata_episodes)

    alignment_info = {
        "metadata_record_count": meta_info.get("record_count", len(metadata_episodes)) if meta_info else len(metadata_episodes),
        "metadata_episode_count": meta_info.get("record_count", len(metadata_episodes)) if meta_info else len(metadata_episodes),
        "metadata_unique_episode_count": len(metadata_episodes),
        "embedding_npy_file_count": load_info.get("npy_file_count", 0) if load_info else 0,
        "embedding_valid_record_count": load_info.get("valid_record_count", 0) if load_info else len(embeddings),
        "embedding_unique_episode_count": len(embedding_episodes),
        "embedding_file_count": load_info.get("file_count", 0) if load_info else len(embeddings),
        "intersection_count": len(intersection),
        "missing_embedding": missing_embeddings,
        "missing_metadata": missing_metadata,
        "duplicate_embedding_episode_index": load_info.get("duplicate_episode_indices", []) if load_info else [],
        "duplicate_metadata_episode_index": meta_info.get("duplicate_episode_indices", []) if meta_info else [],
        "invalid_embedding_files": load_info.get("invalid_files", []) if load_info else [],
    }

    if not intersection:
        raise ValueError("No intersection between embeddings and metadata!")

    episode_indices = sorted(intersection)
    phi_globals = np.array([embeddings[ep]["phi_global"] for ep in episode_indices])
    phi_wrists = np.array([embeddings[ep]["phi_wrist"] for ep in episode_indices])
    obj_init_positions = np.array([metadata[ep]["obj_init_pos"] for ep in episode_indices])
    goal_positions = np.array([metadata[ep]["goal_pos"] for ep in episode_indices])

    return episode_indices, phi_globals, phi_wrists, obj_init_positions, goal_positions, alignment_info


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

    shuffled_r2_x_list = list(results["shuffled_ridge"]["R2_x"])
    shuffled_r2_y_list = list(results["shuffled_ridge"]["R2_y"])
    shuffled_mae_x_list = list(results["shuffled_ridge"]["MAE_x"])
    shuffled_mae_y_list = list(results["shuffled_ridge"]["MAE_y"])

    for key in ["shuffled_ridge", "shuffled_knn"]:
        for metric in ["R2_x", "R2_y", "MAE_x", "MAE_y"]:
            arr = results[key][metric]
            results[key][metric] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "n_shuffles": n_shuffles,
            }

    for metric, null_list in [("R2_x", shuffled_r2_x_list), ("R2_y", shuffled_r2_y_list)]:
        null_arr = np.asarray(null_list)
        observed = results["ridge"][metric]
        results["shuffled_ridge"][metric]["observed"] = observed
        results["shuffled_ridge"][metric]["observed_percentile"] = float(np.mean(null_arr <= observed))
        results["shuffled_ridge"][metric]["empirical_p_value"] = float((np.sum(null_arr >= observed) + 1) / (len(null_arr) + 1))

    for metric, null_list in [("MAE_x", shuffled_mae_x_list), ("MAE_y", shuffled_mae_y_list)]:
        null_arr = np.asarray(null_list)
        observed = results["ridge"][metric]
        results["shuffled_ridge"][metric]["observed"] = observed
        results["shuffled_ridge"][metric]["observed_percentile"] = float(np.mean(null_arr >= observed))
        results["shuffled_ridge"][metric]["empirical_p_value"] = float((np.sum(null_arr <= observed) + 1) / (len(null_arr) + 1))

    return results


def neighbor_overlap_analysis(
    phi: np.ndarray,
    positions: np.ndarray,
    ks: List[int] = [5, 10, 20],
    seed: int = 42,
    n_permutations: int = 1
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

    if n_permutations > 1:
        for k in ks:
            perm_overlaps_list = []
            for perm_i in range(n_permutations):
                phi_perm = phi[rng.permutation(n)]
                emb_dist_perm = pairwise_distances(phi_perm)
                emb_neighbors_perm = np.argsort(emb_dist_perm, axis=1)[:, 1:k+1]
                phys_neighbors_fixed = np.argsort(phys_dist, axis=1)[:, 1:k+1]

                overlaps = []
                for i in range(n):
                    phys_set = set(phys_neighbors_fixed[i])
                    emb_set = set(emb_neighbors_perm[i])
                    intersection = phys_set & emb_set
                    union = phys_set | emb_set
                    overlaps.append(len(intersection) / len(union) if union else 0)

                perm_overlaps_list.append(float(np.mean(overlaps)))

            perm_arr = np.array(perm_overlaps_list)
            real_overlap = results[f"neighbor_overlap@{k}"]
            results[f"neighbor_overlap@{k}_null"] = {
                "mean": float(np.mean(perm_arr)),
                "std": float(np.std(perm_arr)),
                "observed_percentile": float(np.mean(perm_arr <= real_overlap)),
                "empirical_p_value": float((np.sum(perm_arr >= real_overlap) + 1) / (len(perm_arr) + 1)),
                "n_permutations": n_permutations,
            }

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
                "shuffled_balanced_accuracy_mean": None,
                "shuffled_balanced_accuracy_std": None,
                "empty_cells": empty_cells,
                "sparse_cells": sparse_cells,
                "total_cells": grid_x * grid_y,
                "status": "insufficient_samples",
                "min_class_count": min_class_count,
                "n_requested_shuffles": 0,
                "n_valid_shuffles": 0,
                "n_invalid_shuffles": 0,
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
                "shuffled_balanced_accuracy_mean": None,
                "shuffled_balanced_accuracy_std": None,
                "empty_cells": empty_cells,
                "sparse_cells": sparse_cells,
                "total_cells": grid_x * grid_y,
                "status": "cv_failed",
                "error": str(e),
                "n_requested_shuffles": 0,
                "n_valid_shuffles": 0,
                "n_invalid_shuffles": 0,
            }
            continue

        shuffled_accs = []
        shuffled_balanced_accs = []
        n_requested = n_shuffles
        n_valid = 0
        n_invalid = 0

        for _ in range(n_shuffles):
            grid_ids_shuffled = grid_ids[rng.permutation(len(grid_ids))]
            shuffled_cell_counts = np.bincount(grid_ids_shuffled, minlength=grid_x * grid_y)
            shuffled_min_class = int(np.min(shuffled_cell_counts[shuffled_cell_counts > 0])) if np.any(shuffled_cell_counts > 0) else 0

            if shuffled_min_class < 2:
                n_invalid += 1
                continue

            try:
                shuffled_cv = StratifiedKFold(n_splits=min(5, shuffled_min_class), shuffle=True, random_state=seed)
                acc = np.mean(cross_val_score(knn, phi, grid_ids_shuffled, cv=shuffled_cv, scoring='accuracy'))
                balanced_acc = np.mean(cross_val_score(knn, phi, grid_ids_shuffled, cv=shuffled_cv, scoring='balanced_accuracy'))
                shuffled_accs.append(acc)
                shuffled_balanced_accs.append(balanced_acc)
                n_valid += 1
            except Exception:
                n_invalid += 1

        if len(shuffled_accs) > 0:
            shuffled_mean = float(np.mean(shuffled_accs))
            shuffled_std = float(np.std(shuffled_accs))
            shuffled_balanced_mean = float(np.mean(shuffled_balanced_accs))
            shuffled_balanced_std = float(np.std(shuffled_balanced_accs))
        else:
            shuffled_mean = None
            shuffled_std = None
            shuffled_balanced_mean = None
            shuffled_balanced_std = None

        results[f"{grid_x}x{grid_y}"] = {
            "accuracy": float(np.mean(accuracies)),
            "accuracy_std": float(np.std(accuracies)),
            "balanced_accuracy": float(np.mean(balanced_accuracies)),
            "balanced_accuracy_std": float(np.std(balanced_accuracies)),
            "shuffled_accuracy_mean": shuffled_mean,
            "shuffled_accuracy_std": shuffled_std,
            "shuffled_balanced_accuracy_mean": shuffled_balanced_mean,
            "shuffled_balanced_accuracy_std": shuffled_balanced_std,
            "empty_cells": empty_cells,
            "sparse_cells": sparse_cells,
            "total_cells": grid_x * grid_y,
            "status": "ok",
            "n_splits": n_splits,
            "min_class_count": min_class_count,
            "n_requested_shuffles": n_requested,
            "n_valid_shuffles": n_valid,
            "n_invalid_shuffles": n_invalid,
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
) -> Dict:
    """
    Match subset episodes to indices in the aligned dataset.

    Returns dict with matched_indices, missing_episode_indices, duplicate_episode_indices,
    requested_subset_size, matched_subset_size.
    Raises ValueError if any episodes are missing from aligned universe or if the subset
    contains duplicate episode indices (duplicates are never silently de-duplicated).
    """
    idx_map = {ep: i for i, ep in enumerate(all_episode_indices)}
    matched = []
    missing = []
    duplicate = []
    seen = set()

    for ep in subset_episodes:
        if ep in seen:
            duplicate.append(ep)
            continue
        seen.add(ep)
        if ep in idx_map:
            matched.append(idx_map[ep])
        else:
            missing.append(ep)

    if duplicate:
        raise ValueError(
            f"{len(duplicate)} duplicate subset episode indices: {duplicate[:10]}{'...' if len(duplicate) > 10 else ''}. "
            f"Duplicate episodes are not allowed - refusing to de-duplicate silently."
        )

    if missing:
        raise ValueError(
            f"{len(missing)} subset episodes not in aligned dataset: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    return {
        "matched_indices": matched,
        "missing_episode_indices": missing,
        "duplicate_episode_indices": duplicate,
        "requested_subset_size": len(subset_episodes),
        "matched_subset_size": len(matched),
    }


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
        "unselected_median_nearest_distance": float(np.median(nearest_unselected)),
        "unselected_p90_nearest_distance": float(np.percentile(nearest_unselected, 90)),
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


def generate_bootstrap_subsets(
    n_total: int,
    subset_size: int,
    n_bootstrap: int,
    seed: int
) -> np.ndarray:
    """
    Pre-generate bootstrap random subset indices.
    Returns shape (n_bootstrap, subset_size) array of indices.
    All representations (global/wrist/combined/fixed-SIC) must use the same
    pre-generated subsets to ensure consistency.
    """
    rng = np.random.RandomState(seed)
    subsets = np.zeros((n_bootstrap, subset_size), dtype=int)
    for i in range(n_bootstrap):
        subsets[i] = rng.choice(n_total, subset_size, replace=False)
    return subsets


def random_bootstrap_fixed_sic(
    subset_indices: List[int],
    all_episode_indices: List[int],
    K_global: np.ndarray,
    K_wrist: np.ndarray,
    dbar_global: float,
    dbar_wrist: float,
    n_bootstrap: int,
    seed: int,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0,
    pre_generated_subsets: np.ndarray = None,
) -> Dict:
    """
    Compute bootstrap distribution for fixed SIC only.
    This is computed once as an independent evidence item, not per-representation.
    """
    rng = np.random.RandomState(seed)
    n_total = len(all_episode_indices)
    subset_size = len(subset_indices)

    if subset_size == 0 or subset_size >= n_total:
        return None

    if pre_generated_subsets is not None:
        bootstrap_indices = pre_generated_subsets
    else:
        bootstrap_indices = generate_bootstrap_subsets(n_total, subset_size, n_bootstrap, seed)

    bootstrap_sic_vals = []
    for i in range(n_bootstrap):
        sample_indices = bootstrap_indices[i]
        selected_episodes = [all_episode_indices[j] for j in sample_indices]

        sic_result = compute_fixed_universe_sic_from_indices(
            selected_episodes=selected_episodes,
            all_episode_indices=all_episode_indices,
            K_global=K_global,
            K_wrist=K_wrist,
            dbar_global=dbar_global,
            dbar_wrist=dbar_wrist,
            alpha=alpha,
            lambda_wrist=lambda_wrist,
        )
        bootstrap_sic_vals.append(sic_result["normalized_sic"])

        if (i + 1) % 200 == 0:
            print(f"    Fixed SIC Bootstrap {i + 1}/{n_bootstrap}")

    bootstrap_sic = np.array(bootstrap_sic_vals)

    selected_episodes = [all_episode_indices[i] for i in subset_indices]
    sic_obs = compute_fixed_universe_sic_from_indices(
        selected_episodes=selected_episodes,
        all_episode_indices=all_episode_indices,
        K_global=K_global,
        K_wrist=K_wrist,
        dbar_global=dbar_global,
        dbar_wrist=dbar_wrist,
        alpha=alpha,
        lambda_wrist=lambda_wrist,
    )
    observed_sic = sic_obs["normalized_sic"]

    better_fraction = float(np.sum(bootstrap_sic < observed_sic) / len(bootstrap_sic))

    return {
        "observed_fixed_sic": observed_sic,
        "observed": observed_sic,
        "bootstrap_distribution": bootstrap_sic.tolist(),
        "bootstrap_mean": float(np.mean(bootstrap_sic)),
        "bootstrap_std": float(np.std(bootstrap_sic)),
        "better_than_random_fraction": better_fraction,
        "percentile": float(np.sum(bootstrap_sic <= observed_sic) / len(bootstrap_sic)),
        "metadata": {
            "n_bootstrap": n_bootstrap,
            "seed": seed,
            "subset_size": subset_size,
            "alpha": alpha,
            "lambda_wrist": lambda_wrist,
        },
    }


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
    all_episode_indices: List[int] = None,
    precomputed_dist_matrix: np.ndarray = None,
    pre_generated_subsets: np.ndarray = None,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0,
) -> Dict:
    """
    Random bootstrap baseline analysis with coverage and redundancy.
    Fixed SIC is computed separately via random_bootstrap_fixed_sic().
    Uses pre-generated subsets if provided to ensure all representations
    use the exact same random samples.
    """
    rng = np.random.RandomState(seed)
    n_total = len(all_phi)
    subset_size = len(subset_indices)

    if subset_size == 0 or subset_size >= n_total:
        return None

    if precomputed_dist_matrix is not None:
        dist_matrix = precomputed_dist_matrix
    else:
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

    if all_episode_indices is None:
        all_episode_indices = list(range(n_total))

    if pre_generated_subsets is not None:
        bootstrap_indices = pre_generated_subsets
    else:
        bootstrap_indices = generate_bootstrap_subsets(n_total, subset_size, n_bootstrap, seed)

    print(f"  Running {n_bootstrap} bootstrap iterations...")
    for boot_i in range(n_bootstrap):
        sample_indices = bootstrap_indices[boot_i]

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

        lower_better = True
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

    results["metadata"] = {
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "subset_size": subset_size,
        "alpha": alpha,
        "lambda_wrist": lambda_wrist,
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
    K_global: np.ndarray,
    K_wrist: np.ndarray,
    dbar_global: float,
    dbar_wrist: float,
    name: str,
    alpha: float = 1.0,
    lambda_wrist: float = 1.0,
) -> Dict:
    """
    Compute fixed-universe SIC for a subset using precomputed K matrices and dbar values.
    Ensures consistency across all methods and bootstrap iterations.
    """
    if not subset_indices:
        return None

    subset_episodes = [all_episode_indices[i] for i in subset_indices]

    sic_result = compute_fixed_universe_sic_from_indices(
        selected_episodes=subset_episodes,
        all_episode_indices=all_episode_indices,
        K_global=K_global,
        K_wrist=K_wrist,
        dbar_global=dbar_global,
        dbar_wrist=dbar_wrist,
        alpha=alpha,
        lambda_wrist=lambda_wrist,
    )

    return {
        "method": name,
        "fixed_universe_sic": sic_result["fixed_universe_sic"],
        "normalized_sic": sic_result["normalized_sic"],
        "reference_anchor_count": sic_result["reference_anchor_count"],
        "dbar_global": dbar_global,
        "dbar_wrist": dbar_wrist,
    }


def evaluate_ours_vs_uniform(
    ours_data: Dict,
    uniform_data: Dict,
    ours_episodes: List[int],
    uniform_episodes: List[int],
) -> Dict:
    """
    Comprehensive Ours vs Uniform comparison.
    Evaluates whether Ours mainly reproduces Uniform initial-position coverage.
    Outputs raw delta metrics and evidence summary without hard-coded thresholds.
    """
    ours_eps = set(ours_episodes)
    uniform_eps = set(uniform_episodes)
    overlap_eps = ours_eps & uniform_eps
    overlap_count = len(overlap_eps)
    overlap_ratio = overlap_count / min(len(ours_eps), len(uniform_eps)) if min(len(ours_eps), len(uniform_eps)) > 0 else 0.0

    result = {
        "episode_overlap_count": overlap_count,
        "episode_overlap_ratio": overlap_ratio,
    }

    ours_ws = ours_data.get("workspace_coverage") or {}
    uniform_ws = uniform_data.get("workspace_coverage") or {}
    ws_delta = {}
    for key in ["physical_unselected_mean_nearest", "physical_unselected_p95", "physical_unselected_max_radius"]:
        v1 = ours_ws.get(key)
        v2 = uniform_ws.get(key)
        if v1 is not None and v2 is not None:
            ws_delta[key] = v1 - v2
    for grid_key in ["grid_7x4_ratio", "grid_14x8_ratio"]:
        v1 = ours_ws.get(grid_key)
        v2 = uniform_ws.get(grid_key)
        if v1 is not None and v2 is not None:
            ws_delta[grid_key] = v1 - v2
    result["workspace_coverage_delta"] = ws_delta

    ours_g = ours_data.get("coverage_global") or {}
    uniform_g = uniform_data.get("coverage_global") or {}
    global_delta = {}
    for key in ["unselected_mean_nearest_distance", "unselected_median_nearest_distance",
                 "unselected_p90_nearest_distance", "unselected_p95_nearest_distance",
                 "unselected_max_nearest_distance"]:
        v1 = ours_g.get(key)
        v2 = uniform_g.get(key)
        if v1 is not None and v2 is not None:
            global_delta[key] = v1 - v2
    result["global_coverage_delta"] = global_delta

    ours_w = ours_data.get("coverage_wrist") or {}
    uniform_w = uniform_data.get("coverage_wrist") or {}
    wrist_delta = {}
    for key in ["unselected_mean_nearest_distance", "unselected_median_nearest_distance",
                 "unselected_p90_nearest_distance", "unselected_p95_nearest_distance",
                 "unselected_max_nearest_distance"]:
        v1 = ours_w.get(key)
        v2 = uniform_w.get(key)
        if v1 is not None and v2 is not None:
            wrist_delta[key] = v1 - v2
    result["wrist_coverage_delta"] = wrist_delta

    ours_c = ours_data.get("coverage_combined") or {}
    uniform_c = uniform_data.get("coverage_combined") or {}
    combined_delta = {}
    for key in ["unselected_mean_nearest_distance", "unselected_median_nearest_distance",
                 "unselected_p90_nearest_distance", "unselected_p95_nearest_distance",
                 "unselected_max_nearest_distance"]:
        v1 = ours_c.get(key)
        v2 = uniform_c.get(key)
        if v1 is not None and v2 is not None:
            combined_delta[key] = v1 - v2
    result["combined_coverage_delta"] = combined_delta

    ours_gr = ours_data.get("redundancy_global") or {}
    uniform_gr = uniform_data.get("redundancy_global") or {}
    global_red_delta = {}
    for key in ["mean_nearest", "median_nearest", "p10_nearest", "p50_nearest", "p90_nearest", "redundancy_fraction"]:
        v1 = ours_gr.get(key)
        v2 = uniform_gr.get(key)
        if v1 is not None and v2 is not None:
            global_red_delta[key] = v1 - v2
    result["global_redundancy_delta"] = global_red_delta

    ours_wr = ours_data.get("redundancy_wrist") or {}
    uniform_wr = uniform_data.get("redundancy_wrist") or {}
    wrist_red_delta = {}
    for key in ["mean_nearest", "median_nearest", "p10_nearest", "p50_nearest", "p90_nearest", "redundancy_fraction"]:
        v1 = ours_wr.get(key)
        v2 = uniform_wr.get(key)
        if v1 is not None and v2 is not None:
            wrist_red_delta[key] = v1 - v2
    result["wrist_redundancy_delta"] = wrist_red_delta

    ours_cr = ours_data.get("redundancy_combined") or {}
    uniform_cr = uniform_data.get("redundancy_combined") or {}
    combined_red_delta = {}
    for key in ["mean_nearest", "median_nearest", "p10_nearest", "p50_nearest", "p90_nearest", "redundancy_fraction"]:
        v1 = ours_cr.get(key)
        v2 = uniform_cr.get(key)
        if v1 is not None and v2 is not None:
            combined_red_delta[key] = v1 - v2
    result["combined_redundancy_delta"] = combined_red_delta

    ours_fixed_sic = ours_data.get("fixed_sic") or {}
    uniform_fixed_sic = uniform_data.get("fixed_sic") or {}
    ours_sic = ours_fixed_sic.get("normalized_sic")
    uniform_sic = uniform_fixed_sic.get("normalized_sic")
    sic_delta = (ours_sic - uniform_sic) if (ours_sic is not None and uniform_sic is not None) else None
    result["fixed_sic_delta"] = sic_delta

    evidence_summary = {}
    eps = 1e-10

    def _relative_delta(ours_val, uniform_val):
        if ours_val is None or uniform_val is None:
            return None
        return (ours_val - uniform_val) / max(abs(uniform_val), eps)

    for key in ["physical_unselected_mean_nearest", "physical_unselected_p95", "physical_unselected_max_radius"]:
        v1 = ours_ws.get(key)
        v2 = uniform_ws.get(key)
        rd = _relative_delta(v1, v2)
        evidence_summary[f"workspace_{key}_relative_delta"] = rd

    for key in ["unselected_mean_nearest_distance", "unselected_median_nearest_distance",
                 "unselected_p90_nearest_distance", "unselected_p95_nearest_distance",
                 "unselected_max_nearest_distance"]:
        v1 = ours_g.get(key)
        v2 = uniform_g.get(key)
        rd = _relative_delta(v1, v2)
        evidence_summary[f"global_coverage_{key}_relative_delta"] = rd

    for key in ["unselected_mean_nearest_distance", "unselected_median_nearest_distance",
                 "unselected_p90_nearest_distance", "unselected_p95_nearest_distance",
                 "unselected_max_nearest_distance"]:
        v1 = ours_w.get(key)
        v2 = uniform_w.get(key)
        rd = _relative_delta(v1, v2)
        evidence_summary[f"wrist_coverage_{key}_relative_delta"] = rd

    for key in ["unselected_mean_nearest_distance", "unselected_median_nearest_distance",
                 "unselected_p90_nearest_distance", "unselected_p95_nearest_distance",
                 "unselected_max_nearest_distance"]:
        v1 = ours_c.get(key)
        v2 = uniform_c.get(key)
        rd = _relative_delta(v1, v2)
        evidence_summary[f"combined_coverage_{key}_relative_delta"] = rd

    for key in ["mean_nearest", "median_nearest", "p10_nearest", "p50_nearest", "p90_nearest", "redundancy_fraction"]:
        v1 = ours_gr.get(key)
        v2 = uniform_gr.get(key)
        rd = _relative_delta(v1, v2)
        evidence_summary[f"global_redundancy_{key}_relative_delta"] = rd

    for key in ["mean_nearest", "median_nearest", "p10_nearest", "p50_nearest", "p90_nearest", "redundancy_fraction"]:
        v1 = ours_wr.get(key)
        v2 = uniform_wr.get(key)
        rd = _relative_delta(v1, v2)
        evidence_summary[f"wrist_redundancy_{key}_relative_delta"] = rd

    for key in ["mean_nearest", "median_nearest", "p10_nearest", "p50_nearest", "p90_nearest", "redundancy_fraction"]:
        v1 = ours_cr.get(key)
        v2 = uniform_cr.get(key)
        rd = _relative_delta(v1, v2)
        evidence_summary[f"combined_redundancy_{key}_relative_delta"] = rd

    evidence_summary["fixed_sic_delta"] = sic_delta
    if ours_sic is not None and uniform_sic is not None:
        evidence_summary["fixed_sic_relative_delta"] = (ours_sic - uniform_sic) / max(abs(uniform_sic), eps)

    evidence_summary["note"] = (
        "No calibrated equivalence test was performed; inspect the multi-metric deltas above. "
        "Relative deltas are computed as (ours - uniform) / max(|uniform|, eps)."
    )

    result["evidence_summary"] = evidence_summary

    result["conclusion"] = "Insufficient evidence that Ours mainly reproduces Uniform initial-position coverage."

    return result


def evaluate_h1(analysis_results: Dict) -> Dict:
    """
    Evaluate H1: Embedding 是否具有可区分性？
    Based on: validity/collapse, position probe vs shuffled, neighbor overlap vs random.
    Uses null distribution evidence (empirical p-values) rather than fixed thresholds.
    """
    evidence = {}

    for rep in ["global", "wrist", "combined"]:
        rep_evidence = {
            "probe_x_significant": False,
            "probe_y_significant": False,
            "neighbor_significant": False,
            "validity": False,
        }

        probe = analysis_results.get("probe", {}).get(rep, {})
        if probe:
            shuffled_r2_x = probe.get("shuffled_ridge", {}).get("R2_x", {})
            shuffled_r2_y = probe.get("shuffled_ridge", {}).get("R2_y", {})

            emp_p_x = shuffled_r2_x.get("empirical_p_value")
            if emp_p_x is not None:
                rep_evidence["probe_x_significant"] = emp_p_x < 0.05
                rep_evidence["probe_x_evidence"] = "empirical_p_value"
            else:
                # No formal statistical test available. Do not fabricate significance
                # from 'observed > null_mean'.
                rep_evidence["probe_x_significant"] = False
                rep_evidence["probe_x_evidence"] = "unavailable"

            emp_p_y = shuffled_r2_y.get("empirical_p_value")
            if emp_p_y is not None:
                rep_evidence["probe_y_significant"] = emp_p_y < 0.05
                rep_evidence["probe_y_evidence"] = "empirical_p_value"
            else:
                rep_evidence["probe_y_significant"] = False
                rep_evidence["probe_y_evidence"] = "unavailable"

        overlap = analysis_results.get("neighbor_overlap", {}).get(rep, {})
        if overlap:
            null_data = overlap.get("neighbor_overlap@10_null", {})
            emp_p = null_data.get("empirical_p_value")
            if emp_p is not None:
                rep_evidence["neighbor_significant"] = emp_p < 0.05
                rep_evidence["neighbor_evidence"] = "empirical_p_value"
            else:
                rep_evidence["neighbor_significant"] = False
                rep_evidence["neighbor_evidence"] = "unavailable"
            rep_evidence["neighbor_overlap@10"] = overlap.get("neighbor_overlap@10", 0)
            rep_evidence["random_neighbor_overlap@10"] = overlap.get("random_neighbor_overlap@10", 0)
            rep_evidence["neighbor_null_mean"] = null_data.get("mean", None)
            rep_evidence["neighbor_null_std"] = null_data.get("std", None)
            rep_evidence["neighbor_null_observed_percentile"] = null_data.get("observed_percentile", None)

        validation = analysis_results.get("validation", {})
        if validation.get("valid", False):
            rep_evidence["validity"] = True

        evidence[rep] = rep_evidence

    n_probe_sig = sum(1 for rep in ["global", "wrist", "combined"]
                      if evidence.get(rep, {}).get("probe_x_significant") and evidence.get(rep, {}).get("probe_y_significant"))
    n_neighbor_sig = sum(1 for rep in ["global", "wrist", "combined"]
                         if evidence.get(rep, {}).get("neighbor_significant"))

    eff_rank_global = analysis_results.get("global_stats", {}).get("effective_rank", 0)
    dim_global = analysis_results.get("global_stats", {}).get("dimension", 1)
    eff_rank_ratio = eff_rank_global / dim_global if dim_global > 0 else 0

    if n_probe_sig >= 2 and n_neighbor_sig >= 1:
        status = "SUPPORTED"
    elif n_probe_sig >= 1 or n_neighbor_sig >= 1:
        status = "WEAK"
    else:
        status = "NOT SUPPORTED"

    return {
        "hypothesis": "H1",
        "status": status,
        "evidence": evidence,
        "effective_rank_ratio": eff_rank_ratio,
        "description": "Embedding 是否具有可区分性？"
    }


def evaluate_h2(analysis_results: Dict) -> Dict:
    """
    Evaluate H2: Embedding distance 是否具有 task-state geometry？
    Based on: Spearman + permutation test, position probe vs shuffled, neighbor overlap vs random.
    Uses permutation p-value and empirical null evidence rather than fixed rho threshold.
    """
    spearman_results = analysis_results.get("spearman", {})
    perm_tests = analysis_results.get("permutation_tests", {})
    probe_results = analysis_results.get("probe", {})
    overlap_results = analysis_results.get("neighbor_overlap", {})

    evidence = {}
    for rep in ["global", "wrist", "combined"]:
        rep_evidence = {
            "spearman_significant": False,
            "probe_x_significant": False,
            "probe_y_significant": False,
            "neighbor_significant": False,
        }

        perm = perm_tests.get(rep, {})
        if perm.get("permutation_p_value", 1.0) < 0.05:
            rep_evidence["spearman_significant"] = True

        probe = probe_results.get(rep, {})
        if probe:
            shuffled_r2_x = probe.get("shuffled_ridge", {}).get("R2_x", {})
            shuffled_r2_y = probe.get("shuffled_ridge", {}).get("R2_y", {})

            emp_p_x = shuffled_r2_x.get("empirical_p_value")
            if emp_p_x is not None:
                rep_evidence["probe_x_significant"] = emp_p_x < 0.05
                rep_evidence["probe_x_evidence"] = "empirical_p_value"
            else:
                rep_evidence["probe_x_significant"] = False
                rep_evidence["probe_x_evidence"] = "unavailable"

            emp_p_y = shuffled_r2_y.get("empirical_p_value")
            if emp_p_y is not None:
                rep_evidence["probe_y_significant"] = emp_p_y < 0.05
                rep_evidence["probe_y_evidence"] = "empirical_p_value"
            else:
                rep_evidence["probe_y_significant"] = False
                rep_evidence["probe_y_evidence"] = "unavailable"

        overlap = overlap_results.get(rep, {})
        if overlap:
            null_data = overlap.get("neighbor_overlap@10_null", {})
            emp_p = null_data.get("empirical_p_value")
            if emp_p is not None:
                rep_evidence["neighbor_significant"] = emp_p < 0.05
                rep_evidence["neighbor_evidence"] = "empirical_p_value"
            else:
                rep_evidence["neighbor_significant"] = False
                rep_evidence["neighbor_evidence"] = "unavailable"
            rep_evidence["neighbor_overlap@10"] = overlap.get("neighbor_overlap@10", 0)
            rep_evidence["random_neighbor_overlap@10"] = overlap.get("random_neighbor_overlap@10", 0)
            rep_evidence["neighbor_null_mean"] = null_data.get("mean", None)
            rep_evidence["neighbor_null_std"] = null_data.get("std", None)
            rep_evidence["neighbor_null_observed_percentile"] = null_data.get("observed_percentile", None)

        evidence[rep] = rep_evidence

    n_sig_spearman = sum(1 for rep in ["global", "wrist", "combined"]
                         if evidence.get(rep, {}).get("spearman_significant", False))
    n_probe_sig = sum(1 for rep in ["global", "wrist", "combined"]
                      if evidence.get(rep, {}).get("probe_x_significant") and evidence.get(rep, {}).get("probe_y_significant"))
    n_neighbor_sig = sum(1 for rep in ["global", "wrist", "combined"]
                         if evidence.get(rep, {}).get("neighbor_significant"))

    if n_sig_spearman >= 2 and n_probe_sig >= 1 and n_neighbor_sig >= 1:
        status = "SUPPORTED"
    elif n_sig_spearman >= 2 or (n_probe_sig >= 1 and n_neighbor_sig >= 1):
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
    Groups metrics by evidence family to avoid treating highly correlated metrics as independent.
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

    evidence_families = {
        "coverage_quality": [],
        "max_radius_tail": [],
        "redundancy": [],
        "fixed_sic": [],
    }

    for rep in ["global", "wrist", "combined"]:
        bootstrap = ours_data.get(f"bootstrap_{rep}", {})
        if bootstrap:
            for metric in ["mean_nearest", "p95_nearest"]:
                metric_data = bootstrap.get(metric, {})
                btr = metric_data.get("better_than_random_fraction", None)
                if btr is not None:
                    evidence_families["coverage_quality"].append({
                        "representation": rep,
                        "metric": metric,
                        "better_than_random_fraction": btr,
                    })

            max_data = bootstrap.get("max_radius", {})
            max_btr = max_data.get("better_than_random_fraction", None)
            if max_btr is not None:
                evidence_families["max_radius_tail"].append({
                    "representation": rep,
                    "metric": "max_radius",
                    "better_than_random_fraction": max_btr,
                })

            red_data = bootstrap.get("redundancy_fraction", {})
            red_btr = red_data.get("better_than_random_fraction", None)
            if red_btr is not None:
                evidence_families["redundancy"].append({
                    "representation": rep,
                    "metric": "redundancy_fraction",
                    "better_than_random_fraction": red_btr,
                })

    fixed_sic_bootstrap = ours_data.get("bootstrap_fixed_sic", {})
    if fixed_sic_bootstrap:
        sic_btr = fixed_sic_bootstrap.get("better_than_random_fraction", None)
        if sic_btr is not None:
            evidence_families["fixed_sic"].append({
                "representation": "combined",
                "metric": "normalized_fixed_sic",
                "better_than_random_fraction": sic_btr,
            })

    all_core_metrics = []
    for family_metrics in evidence_families.values():
        all_core_metrics.extend(family_metrics)

    if not all_core_metrics:
        return {
            "hypothesis": "H3",
            "status": "NOT EVALUATED",
            "evidence": {},
            "description": "Ours subset 是否显著优于 random coverage？"
        }

    n_random = sum(1 for m in all_core_metrics if 0.4 <= m["better_than_random_fraction"] <= 0.6)
    n_poor = sum(1 for m in all_core_metrics if m["better_than_random_fraction"] < 0.1)

    n_strong_families = 0
    for family_name, family_metrics in evidence_families.items():
        if not family_metrics:
            continue
        family_strong = sum(1 for m in family_metrics if m["better_than_random_fraction"] >= 0.95)
        if family_strong >= len(family_metrics) * 0.5:
            n_strong_families += 1

    n_weak_families = 0
    for family_name, family_metrics in evidence_families.items():
        if not family_metrics:
            continue
        family_weak = sum(1 for m in family_metrics if 0.8 <= m["better_than_random_fraction"] < 0.95)
        if family_weak >= len(family_metrics) * 0.5:
            n_weak_families += 1

    if n_random > len(all_core_metrics) * 0.5:
        status = "NOT SUPPORTED"
    elif n_strong_families >= 3 and n_poor == 0:
        status = "SUPPORTED"
    elif n_strong_families >= 2 or n_weak_families >= 2:
        status = "WEAK"
    else:
        status = "NOT SUPPORTED"

    return {
        "hypothesis": "H3",
        "status": status,
        "evidence": {
            "core_metrics": all_core_metrics,
            "evidence_families": {k: len(v) for k, v in evidence_families.items()},
            "evidence_families_detail": {k: v for k, v in evidence_families.items()},
            "n_strong_families": n_strong_families,
            "n_weak_families": n_weak_families,
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


def _format_report_value(val):
    """Format a report cell value. Returns 'N/A' for None/non-numeric, otherwise 4-dp."""
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        return f"{val:.4f}"
    return str(val)


def generate_report(
    analysis_results: Dict,
    output_dir: Path
):
    """Generate analysis_report.md"""
    report_path = output_dir / "analysis_report.md"

    hypothesis_eval = analysis_results.get("hypothesis_evaluation", {})

    def fmt(val):
        return _format_report_value(val)

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
        f.write(f"- Exact duplicate groups (global): {stats.get('n_exact_dup_global', 'N/A')}\n")
        f.write(f"- Exact duplicate groups (wrist): {stats.get('n_exact_dup_wrist', 'N/A')}\n")
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
        f.write("| Representation | Spearman XY | Permutation p | R2 x | R2 y | NN overlap@10 | Effective Rank |\n")
        f.write("|---------------|-------------|---------------|------|------|---------------|----------------|\n")

        perm_tests = analysis_results.get("permutation_tests", {})
        for rep in ["global", "wrist", "combined"]:
            stats = analysis_results.get(f"{rep}_stats", {})
            spearman = analysis_results.get("spearman", {}).get(rep, {})
            probe = analysis_results.get("probe", {}).get(rep, {})
            overlap = analysis_results.get("neighbor_overlap", {}).get(rep, {})

            spearman_rho = spearman.get('rho', 'N/A')
            perm = perm_tests.get(rep, {})
            perm_p = perm.get('permutation_p_value', 'N/A')
            r2_x = probe.get('ridge', {}).get('R2_x', 'N/A')
            r2_y = probe.get('ridge', {}).get('R2_y', 'N/A')
            nn_overlap = overlap.get('neighbor_overlap@10', 'N/A')
            eff_rank = stats.get('effective_rank', 'N/A')

            spearman_rho_str = f"{spearman_rho:.4f}" if isinstance(spearman_rho, float) else str(spearman_rho)
            perm_p_str = f"{perm_p:.2e}" if isinstance(perm_p, float) else str(perm_p)
            r2_x_str = f"{r2_x:.4f}" if isinstance(r2_x, float) else str(r2_x)
            r2_y_str = f"{r2_y:.4f}" if isinstance(r2_y, float) else str(r2_y)
            nn_overlap_str = f"{nn_overlap:.4f}" if isinstance(nn_overlap, float) else str(nn_overlap)
            eff_rank_str = f"{eff_rank:.2f}" if isinstance(eff_rank, float) else str(eff_rank)

            f.write(f"| {rep} | {spearman_rho_str} | {perm_p_str} | "
                   f"{r2_x_str} | {r2_y_str} | "
                   f"{nn_overlap_str} | {eff_rank_str} |\n")

        f.write("\n## Table 2: Subset Coverage Comparison\n\n")
        f.write("| Method | Global Cover | Wrist Cover | Combined Cover | Combined Max Radius | Combined Redundancy | Fixed SIC | Fixed SIC BTR | Global BTR | Wrist BTR | Combined BTR |\n")
        f.write("|--------|-------------|-------------|----------------|---------------------|---------------------|-----------|---------------|------------|-----------|--------------|\n")

        subset_comparison = analysis_results.get("subset_comparison", {})
        for method_name, method_data in subset_comparison.items():
            cov_g = method_data.get('coverage_global', {})
            cov_w = method_data.get('coverage_wrist', {})
            cov_c = method_data.get('coverage_combined', {})
            red_c = method_data.get('redundancy_combined', {})
            sic = method_data.get('fixed_sic', {})
            boot_g = method_data.get('bootstrap_global', {})
            boot_w = method_data.get('bootstrap_wrist', {})
            boot_c = method_data.get('bootstrap_combined', {})
            boot_fixed = method_data.get('bootstrap_fixed_sic', {})

            mean_cover_g = cov_g.get('unselected_mean_nearest_distance') if cov_g else None
            mean_cover_w = cov_w.get('unselected_mean_nearest_distance') if cov_w else None
            mean_cover_c = cov_c.get('unselected_mean_nearest_distance') if cov_c else None
            max_radius_c = cov_c.get('unselected_max_nearest_distance') if cov_c else None
            redundancy_c = red_c.get('redundancy_fraction') if red_c else None
            fixed_sic = sic.get('normalized_sic') if sic else None
            fixed_sic_btr = boot_fixed.get('better_than_random_fraction') if boot_fixed else None
            btr_g = boot_g.get('mean_nearest', {}).get('better_than_random_fraction') if boot_g else None
            btr_w = boot_w.get('mean_nearest', {}).get('better_than_random_fraction') if boot_w else None
            btr_c = boot_c.get('mean_nearest', {}).get('better_than_random_fraction') if boot_c else None

            f.write(f"| {method_name} | {fmt(mean_cover_g)} | "
                   f"{fmt(mean_cover_w)} | "
                   f"{fmt(mean_cover_c)} | "
                   f"{fmt(max_radius_c)} | "
                   f"{fmt(redundancy_c)} | "
                   f"{fmt(fixed_sic)} | "
                   f"{fmt(fixed_sic_btr)} | "
                   f"{fmt(btr_g)} | "
                   f"{fmt(btr_w)} | "
                   f"{fmt(btr_c)} |\n")

        f.write("\n## Ours vs Uniform Comparison\n\n")
        ours_uniform = analysis_results.get("ours_uniform_comparison", {})
        if ours_uniform:
            f.write(f"- Episode overlap count: {ours_uniform.get('episode_overlap_count', 'N/A')}\n")
            overlap_ratio_val = ours_uniform.get('episode_overlap_ratio')
            f.write(f"- Episode overlap ratio: {fmt(overlap_ratio_val)}\n")
            f.write(f"- Conclusion: {ours_uniform.get('conclusion', 'N/A')}\n\n")

            ev = ours_uniform.get("evidence_summary", {})
            if ev:
                f.write("### Evidence Summary\n\n")
                for k, v in ev.items():
                    f.write(f"- {k}: {fmt(v) if isinstance(v, (int, float)) else v}\n")
                f.write("\n")

            ws_delta = ours_uniform.get("workspace_coverage_delta", {})
            if ws_delta:
                f.write("### Workspace Coverage Delta (Ours - Uniform)\n\n")
                f.write("| Metric | Delta |\n")
                f.write("|--------|-------|\n")
                for k, v in ws_delta.items():
                    f.write(f"| {k} | {fmt(v)} |\n")
                f.write("\n")

            g_delta = ours_uniform.get("global_coverage_delta", {})
            if g_delta:
                f.write("### Global Coverage Delta (Ours - Uniform)\n\n")
                f.write("| Metric | Delta |\n")
                f.write("|--------|-------|\n")
                for k, v in g_delta.items():
                    f.write(f"| {k} | {fmt(v)} |\n")
                f.write("\n")

            w_delta = ours_uniform.get("wrist_coverage_delta", {})
            if w_delta:
                f.write("### Wrist Coverage Delta (Ours - Uniform)\n\n")
                f.write("| Metric | Delta |\n")
                f.write("|--------|-------|\n")
                for k, v in w_delta.items():
                    f.write(f"| {k} | {fmt(v)} |\n")
                f.write("\n")

            c_delta = ours_uniform.get("combined_coverage_delta", {})
            if c_delta:
                f.write("### Combined Coverage Delta (Ours - Uniform)\n\n")
                f.write("| Metric | Delta |\n")
                f.write("|--------|-------|\n")
                for k, v in c_delta.items():
                    f.write(f"| {k} | {fmt(v)} |\n")
                f.write("\n")

            g_red_delta = ours_uniform.get("global_redundancy_delta", {})
            if g_red_delta:
                f.write("### Global Redundancy Delta (Ours - Uniform)\n\n")
                f.write("| Metric | Delta |\n")
                f.write("|--------|-------|\n")
                for k, v in g_red_delta.items():
                    f.write(f"| {k} | {fmt(v)} |\n")
                f.write("\n")

            w_red_delta = ours_uniform.get("wrist_redundancy_delta", {})
            if w_red_delta:
                f.write("### Wrist Redundancy Delta (Ours - Uniform)\n\n")
                f.write("| Metric | Delta |\n")
                f.write("|--------|-------|\n")
                for k, v in w_red_delta.items():
                    f.write(f"| {k} | {fmt(v)} |\n")
                f.write("\n")

            c_red_delta = ours_uniform.get("combined_redundancy_delta", {})
            if c_red_delta:
                f.write("### Combined Redundancy Delta (Ours - Uniform)\n\n")
                f.write("| Metric | Delta |\n")
                f.write("|--------|-------|\n")
                for k, v in c_red_delta.items():
                    f.write(f"| {k} | {fmt(v)} |\n")
                f.write("\n")

            sic_delta = ours_uniform.get("fixed_sic_delta")
            f.write(f"- Fixed SIC Delta (Ours - Uniform): {fmt(sic_delta)}\n\n")

        f.write("## Hypothesis Evaluation\n\n")
        for h_name in ["H1", "H2", "H3"]:
            h_data = hypothesis_eval.get(h_name, {})
            f.write(f"### {h_name}: {h_data.get('description', '')}\n\n")
            f.write(f"- Status: **{h_data.get('status', 'N/A')}**\n")
            if h_name == "H1":
                ev = h_data.get("evidence", {})
                for rep in ["global", "wrist", "combined"]:
                    rep_ev = ev.get(rep, {})
                    f.write(f"- {rep}: probe_x_significant={rep_ev.get('probe_x_significant', False)}, probe_y_significant={rep_ev.get('probe_y_significant', False)}, neighbor_significant={rep_ev.get('neighbor_significant', False)}, validity={rep_ev.get('validity', False)}\n")
                eff_rank_ratio_val = h_data.get('effective_rank_ratio')
                f.write(f"- Effective rank ratio: {fmt(eff_rank_ratio_val)}\n")
            elif h_name == "H2":
                ev = h_data.get("evidence", {})
                for rep in ["global", "wrist", "combined"]:
                    rep_ev = ev.get(rep, {})
                    f.write(f"- {rep}: spearman_significant={rep_ev.get('spearman_significant', False)}, probe_x_significant={rep_ev.get('probe_x_significant', False)}, probe_y_significant={rep_ev.get('probe_y_significant', False)}, neighbor_significant={rep_ev.get('neighbor_significant', False)}\n")
                f.write(f"- Significant Spearman tests: {h_data.get('n_sig_spearman', 'N/A')}\n")
            elif h_name == "H3":
                ev = h_data.get("evidence", {})
                f.write(f"- Strong evidence families (n_strong_families): {ev.get('n_strong_families', 0)}\n")
                f.write(f"- Weak evidence families (n_weak_families): {ev.get('n_weak_families', 0)}\n")
                f.write(f"- Random-range metrics (n_random): {ev.get('n_random', 0)}\n")
                f.write(f"- Poor metrics (n_poor): {ev.get('n_poor', 0)}\n")
                families_detail = ev.get("evidence_families_detail", {})
                if families_detail:
                    f.write(f"- Evidence families detail:\n")
                    for fam_name, fam_items in families_detail.items():
                        f.write(f"  - {fam_name}: {fam_items}\n")
            f.write("\n")

        f.write("## Phase Representation Warning\n\n")
        f.write("当前 global embedding 主要来自 episode 前 5 帧。\n")
        f.write("当前 wrist embedding 主要来自 episode 20%-70% 区间平均 representation。\n")
        f.write("因此当前分析主要验证 initial / pre-grasp task-state representation。\n")
        f.write("它不能证明 transport / place / release phase representation 是充分的。\n")
        f.write("不要因为 embedding initial-state geometry 很强就得出完整 manipulation representation 已经有效的结论。\n")


def _numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _convert_to_python(obj):
    """Recursively convert numpy types in nested structures to Python native types."""
    if isinstance(obj, dict):
        return {k: _convert_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_python(item) for item in obj]
    else:
        return _numpy_to_python(obj)


def save_analysis_summary(
    analysis_results: Dict,
    output_dir: Path
):
    """Save complete analysis_summary.json"""
    summary_path = output_dir / "analysis_summary.json"

    alignment = analysis_results.get("alignment", {})

    summary = {
        "analysis_config": analysis_results.get("analysis_config", {}),
        "fixed_universe": analysis_results.get("fixed_universe", {}),
        "alignment": alignment,
        "validation": analysis_results.get("validation", {}),
        "representations": {
            "global": analysis_results.get("global_stats", {}),
            "wrist": analysis_results.get("wrist_stats", {}),
            "combined": analysis_results.get("combined_stats", {}),
            "global_normalized": analysis_results.get("global_normalized_stats", {}),
            "wrist_normalized": analysis_results.get("wrist_normalized_stats", {}),
        },
        "spearman": analysis_results.get("spearman", {}),
        "permutation_tests": analysis_results.get("permutation_tests", {}),
        "probe": analysis_results.get("probe", {}),
        "neighbor_overlap": analysis_results.get("neighbor_overlap", {}),
        "grid_classifiability": analysis_results.get("grid_classifiability", {}),
        "subset_comparison": analysis_results.get("subset_comparison", {}),
        "ours_uniform_comparison": analysis_results.get("ours_uniform_comparison", {}),
        "hypothesis_evaluation": analysis_results.get("hypothesis_evaluation", {}),
    }

    summary = _convert_to_python(summary)

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Analysis summary saved to: {summary_path}")


def save_embedding_metrics(
    analysis_results: Dict,
    output_dir: Path
):
    """Save embedding_metrics.csv"""
    import csv

    csv_path = output_dir / "embedding_metrics.csv"

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "representation", "dimension", "norm_mean", "norm_std", "norm_min", "norm_max",
            "pairwise_dist_mean", "pairwise_dist_std", "pairwise_dist_p05", "pairwise_dist_p25",
            "pairwise_dist_p50", "pairwise_dist_p75", "pairwise_dist_p95", "pairwise_dist_max",
            "effective_rank"
        ])

        for rep in ["global", "wrist", "combined", "global_normalized", "wrist_normalized"]:
            stats = analysis_results.get(f"{rep}_stats", {})
            writer.writerow([
                rep,
                stats.get("dimension", ""),
                stats.get("norm_mean", ""),
                stats.get("norm_std", ""),
                stats.get("norm_min", ""),
                stats.get("norm_max", ""),
                stats.get("pairwise_dist_mean", ""),
                stats.get("pairwise_dist_std", ""),
                stats.get("pairwise_dist_p05", ""),
                stats.get("pairwise_dist_p25", ""),
                stats.get("pairwise_dist_p50", ""),
                stats.get("pairwise_dist_p75", ""),
                stats.get("pairwise_dist_p95", ""),
                stats.get("pairwise_dist_max", ""),
                stats.get("effective_rank", ""),
            ])

    print(f"Embedding metrics saved to: {csv_path}")


def save_coverage_comparison(
    analysis_results: Dict,
    output_dir: Path
):
    """Save coverage_comparison.csv with comprehensive metrics."""
    import csv

    csv_path = output_dir / "coverage_comparison.csv"

    subset_comparison = analysis_results.get("subset_comparison", {})

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "method",
            "global_mean_nearest", "global_median_nearest", "global_p90_nearest", "global_p95_nearest", "global_max_radius", "global_redundancy",
            "wrist_mean_nearest", "wrist_median_nearest", "wrist_p90_nearest", "wrist_p95_nearest", "wrist_max_radius", "wrist_redundancy",
            "combined_mean_nearest", "combined_median_nearest", "combined_p90_nearest", "combined_p95_nearest", "combined_max_radius", "combined_redundancy",
            "fixed_sic", "fixed_sic_btr",
        ])

        for method_name, method_data in subset_comparison.items():
            cov_g = method_data.get('coverage_global', {})
            cov_w = method_data.get('coverage_wrist', {})
            cov_c = method_data.get('coverage_combined', {})
            red_g = method_data.get('redundancy_global', {})
            red_w = method_data.get('redundancy_wrist', {})
            red_c = method_data.get('redundancy_combined', {})
            sic = method_data.get('fixed_sic', {})
            boot_fixed = method_data.get('bootstrap_fixed_sic', {})

            def _get(d, *keys):
                for k in keys:
                    v = d.get(k)
                    if v is not None:
                        return v
                return ''

            writer.writerow([
                method_name,
                _get(cov_g, 'unselected_mean_nearest_distance'),
                _get(cov_g, 'unselected_median_nearest_distance'),
                _get(cov_g, 'unselected_p90_nearest_distance'),
                _get(cov_g, 'unselected_p95_nearest_distance'),
                _get(cov_g, 'unselected_max_nearest_distance'),
                _get(red_g, 'redundancy_fraction'),
                _get(cov_w, 'unselected_mean_nearest_distance'),
                _get(cov_w, 'unselected_median_nearest_distance'),
                _get(cov_w, 'unselected_p90_nearest_distance'),
                _get(cov_w, 'unselected_p95_nearest_distance'),
                _get(cov_w, 'unselected_max_nearest_distance'),
                _get(red_w, 'redundancy_fraction'),
                _get(cov_c, 'unselected_mean_nearest_distance'),
                _get(cov_c, 'unselected_median_nearest_distance'),
                _get(cov_c, 'unselected_p90_nearest_distance'),
                _get(cov_c, 'unselected_p95_nearest_distance'),
                _get(cov_c, 'unselected_max_nearest_distance'),
                _get(red_c, 'redundancy_fraction'),
                _get(sic, 'normalized_sic'),
                _get(boot_fixed, 'better_than_random_fraction'),
            ])

    print(f"Coverage comparison saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Dataset Embedding Analysis")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--random-subset", type=Path, default=None)
    parser.add_argument("--uniform-subset", type=Path, default=None)
    parser.add_argument("--ours-subset", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-wrist", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)

    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Dataset Embedding Analysis")
    print("=" * 60)

    print("\n[1/15] Loading embeddings...")
    embeddings, load_info = load_embeddings(args.embeddings_dir)
    print(f"  Loaded {len(embeddings)} episodes from {load_info['file_count']} files")
    if load_info['duplicate_episode_indices']:
        print(f"  WARNING: {len(load_info['duplicate_episode_indices'])} duplicate episode indices")
    if load_info['invalid_files']:
        print(f"  WARNING: {len(load_info['invalid_files'])} invalid embedding files")

    print("\n[2/15] Loading metadata...")
    metadata, meta_info = load_episode_metadata(args.dataset_root)
    print(f"  Loaded {len(metadata)} episodes from metadata")

    print("\n[3/15] Aligning embeddings with metadata...")
    episode_indices, phi_globals, phi_wrists, obj_init_positions, goal_positions, alignment_info = \
        align_embeddings_with_metadata(embeddings, metadata, load_info=load_info, meta_info=meta_info)
    print(f"  Aligned {len(episode_indices)} episodes")

    n_episodes = len(episode_indices)
    print(f"  Global embedding shape: {phi_globals.shape}")
    print(f"  Wrist embedding shape: {phi_wrists.shape}")

    print("\n[4/15] Computing normalized and combined embeddings...")
    phi_global_normalized = l2_normalize_rows(phi_globals)
    phi_wrist_normalized = l2_normalize_rows(phi_wrists)
    phi_combined = compute_combined_embeddings(phi_globals, phi_wrists, lambda_wrist=args.lambda_wrist)
    print(f"  Combined embedding shape: {phi_combined.shape}")

    print("\n[5/15] Computing embedding statistics...")
    global_stats = compute_embedding_statistics(phi_globals, "global")
    wrist_stats = compute_embedding_statistics(phi_wrists, "wrist")
    combined_stats = compute_embedding_statistics(phi_combined, "combined")
    global_normalized_stats = compute_embedding_statistics(phi_global_normalized, "global_normalized")
    wrist_normalized_stats = compute_embedding_statistics(phi_wrist_normalized, "wrist_normalized")
    print(f"  Global effective rank: {global_stats['effective_rank']:.2f}")
    print(f"  Wrist effective rank: {wrist_stats['effective_rank']:.2f}")
    print(f"  Combined effective rank: {combined_stats['effective_rank']:.2f}")

    print("\n[6/15] Validating embeddings...")
    validation = check_embeddings_valid(phi_globals, phi_wrists)
    print(f"  Valid: {validation['valid']}")
    print(f"  Exact duplicate groups (global): {validation['stats'].get('n_exact_dup_global', 'N/A')}")
    print(f"  Exact duplicate groups (wrist): {validation['stats'].get('n_exact_dup_wrist', 'N/A')}")
    print(f"  Near duplicates (global): {validation['stats'].get('n_near_dup_global', 'N/A')}")
    print(f"  Near duplicates (wrist): {validation['stats'].get('n_near_dup_wrist', 'N/A')}")

    print("\n[7/15] Computing Spearman correlations...")
    phys_dist_2d = pairwise_distances(obj_init_positions[:, :2])

    global_dist = pairwise_distances(phi_globals)
    wrist_dist = pairwise_distances(phi_wrists)
    combined_dist = pairwise_distances(phi_combined)
    global_normalized_dist = pairwise_distances(phi_global_normalized)
    wrist_normalized_dist = pairwise_distances(phi_wrist_normalized)

    spearman_results = {}
    for name, emb_dist in [
        ("global", global_dist), ("wrist", wrist_dist), ("combined", combined_dist),
        ("global_normalized", global_normalized_dist), ("wrist_normalized", wrist_normalized_dist),
    ]:
        rho, p_value, n_pairs = compute_spearman_correlation(phys_dist_2d, emb_dist, seed=args.seed)
        spearman_results[name] = {"rho": rho, "p_value": p_value, "n_pairs": n_pairs}
        print(f"  {name} Spearman rho: {rho:.4f}, p: {p_value:.2e}, n_pairs: {n_pairs}")

    print("\n[8/15] Running permutation tests...")
    perm_tests = {}
    for name, emb_dist in [("global", global_dist), ("wrist", wrist_dist), ("combined", combined_dist)]:
        print(f"  Testing {name}...")
        perm_result = permutation_test_spearman(phys_dist_2d, emb_dist, n_permutations=1000, seed=args.seed)
        perm_tests[name] = perm_result
        print(f"    Observed rho: {perm_result['observed_rho']:.4f}, p-value: {perm_result['permutation_p_value']:.4f}")

    print("\n[9/15] Running position probes...")
    probe_results = {}
    for name, phi in [("global", phi_globals), ("wrist", phi_wrists), ("combined", phi_combined)]:
        print(f"  Probing {name}...")
        probe_result = position_probe(phi, obj_init_positions, n_shuffles=100, seed=args.seed)
        probe_results[name] = probe_result
        print(f"    Ridge R2_x: {probe_result['ridge']['R2_x']:.4f}, R2_y: {probe_result['ridge']['R2_y']:.4f}")

    print("\n[10/15] Analyzing neighborhood preservation...")
    overlap_results = {}
    for name, phi in [("global", phi_globals), ("wrist", phi_wrists), ("combined", phi_combined)]:
        print(f"  Analyzing {name}...")
        overlap_result = neighbor_overlap_analysis(phi, obj_init_positions, ks=[5, 10, 20], seed=args.seed, n_permutations=100)
        overlap_results[name] = overlap_result
        print(f"    NN overlap@10: {overlap_result.get('neighbor_overlap@10', 'N/A'):.4f}")

    print("\n[11/15] Computing grid classifiability...")
    grid_results = grid_classifiability(phi_combined, obj_init_positions, grid_sizes=[(7, 4), (14, 8)], n_shuffles=100, seed=args.seed)
    for grid_name, grid_data in grid_results.items():
        print(f"  {grid_name}: accuracy={grid_data.get('accuracy', 'N/A')}, status={grid_data.get('status', 'N/A')}")

    print("\n[12/15] Computing subset coverage and redundancy...")
    subset_comparison = {}

    dbar_global, dbar_wrist, dbar_fallback_used = compute_dbar_from_embeddings(phi_globals, phi_wrists)
    K_global, K_wrist = build_kernel_matrices(phi_globals, phi_wrists, dbar_global, dbar_wrist)

    print(f"  dbar_global: {dbar_global:.4f}, dbar_wrist: {dbar_wrist:.4f}")

    subsets_to_process = {}
    random_episodes = load_subset_if_provided(args.random_subset)
    if random_episodes is not None:
        subsets_to_process["Random"] = random_episodes

    uniform_episodes = load_subset_if_provided(args.uniform_subset)
    if uniform_episodes is not None:
        subsets_to_process["Uniform"] = uniform_episodes

    ours_episodes = load_subset_if_provided(args.ours_subset)
    if ours_episodes is not None:
        subsets_to_process["Ours"] = ours_episodes

    global_p05 = float(np.percentile(global_dist[np.triu_indices(n_episodes, k=1)], 5))
    wrist_p05 = float(np.percentile(wrist_dist[np.triu_indices(n_episodes, k=1)], 5))
    combined_p05 = float(np.percentile(combined_dist[np.triu_indices(n_episodes, k=1)], 5))

    for method_name, episodes in subsets_to_process.items():
        print(f"\n  Processing {method_name} subset ({len(episodes)} episodes)...")

        match_result = match_subset_to_indices(episodes, episode_indices)
        subset_indices = match_result["matched_indices"]

        subset_comparison[method_name] = {
            "requested_subset_size": match_result["requested_subset_size"],
            "matched_subset_size": match_result["matched_subset_size"],
            "missing_episode_indices": match_result["missing_episode_indices"],
            "duplicate_episode_indices": match_result["duplicate_episode_indices"],
        }

        print(f"    Computing workspace coverage...")
        workspace_cov = compute_workspace_coverage(obj_init_positions, subset_indices, method_name)

        print(f"    Computing global coverage...")
        cov_global = compute_subset_coverage(phi_globals, subset_indices, obj_init_positions, f"{method_name}_global")

        print(f"    Computing wrist coverage...")
        cov_wrist = compute_subset_coverage(phi_wrists, subset_indices, obj_init_positions, f"{method_name}_wrist")

        print(f"    Computing combined coverage...")
        cov_combined = compute_subset_coverage(phi_combined, subset_indices, obj_init_positions, f"{method_name}_combined")

        subset_phi_global = phi_globals[subset_indices]
        subset_phi_wrist = phi_wrists[subset_indices]
        subset_phi_combined = phi_combined[subset_indices]

        print(f"    Computing global redundancy...")
        red_global = compute_subset_redundancy(subset_phi_global, phi_globals, f"{method_name}_global", global_p05)

        print(f"    Computing wrist redundancy...")
        red_wrist = compute_subset_redundancy(subset_phi_wrist, phi_wrists, f"{method_name}_wrist", wrist_p05)

        print(f"    Computing combined redundancy...")
        red_combined = compute_subset_redundancy(subset_phi_combined, phi_combined, f"{method_name}_combined", combined_p05)

        print(f"    Computing fixed-universe SIC...")
        fixed_sic = compute_fixed_universe_sic_for_subset(
            subset_indices, episode_indices, K_global, K_wrist, dbar_global, dbar_wrist, method_name,
            alpha=args.alpha, lambda_wrist=args.lambda_wrist
        )

        subset_comparison[method_name].update({
            "workspace_coverage": workspace_cov,
            "coverage_global": cov_global,
            "coverage_wrist": cov_wrist,
            "coverage_combined": cov_combined,
            "redundancy_global": red_global,
            "redundancy_wrist": red_wrist,
            "redundancy_combined": red_combined,
            "fixed_sic": fixed_sic,
        })

    print("\n[13/15] Running bootstrap analysis...")

    for method_name in subsets_to_process:
        print(f"\n  Bootstrap for {method_name}...")

        episodes = subsets_to_process[method_name]
        match_result = match_subset_to_indices(episodes, episode_indices)
        subset_indices = match_result["matched_indices"]

        # Pre-generate consistent random subsets shared by global/wrist/combined/fixed-SIC
        # for this method so every representation is evaluated on the exact same samples.
        pre_generated_subsets = generate_bootstrap_subsets(
            n_total=n_episodes,
            subset_size=len(subset_indices),
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )

        print(f"    Global bootstrap...")
        boot_global = random_bootstrap_analysis(
            phi_globals, subset_indices, n_bootstrap=args.n_bootstrap, seed=args.seed,
            full_p05_threshold=global_p05,
            K_global=K_global, K_wrist=K_wrist,
            dbar_global=dbar_global, dbar_wrist=dbar_wrist,
            all_episode_indices=episode_indices,
            precomputed_dist_matrix=global_dist,
            alpha=args.alpha,
            lambda_wrist=args.lambda_wrist,
            pre_generated_subsets=pre_generated_subsets,
        )

        print(f"    Wrist bootstrap...")
        boot_wrist = random_bootstrap_analysis(
            phi_wrists, subset_indices, n_bootstrap=args.n_bootstrap, seed=args.seed,
            full_p05_threshold=wrist_p05,
            K_global=K_global, K_wrist=K_wrist,
            dbar_global=dbar_global, dbar_wrist=dbar_wrist,
            all_episode_indices=episode_indices,
            precomputed_dist_matrix=wrist_dist,
            alpha=args.alpha,
            lambda_wrist=args.lambda_wrist,
            pre_generated_subsets=pre_generated_subsets,
        )

        print(f"    Combined bootstrap...")
        boot_combined = random_bootstrap_analysis(
            phi_combined, subset_indices, n_bootstrap=args.n_bootstrap, seed=args.seed,
            full_p05_threshold=combined_p05,
            K_global=K_global, K_wrist=K_wrist,
            dbar_global=dbar_global, dbar_wrist=dbar_wrist,
            all_episode_indices=episode_indices,
            precomputed_dist_matrix=combined_dist,
            alpha=args.alpha,
            lambda_wrist=args.lambda_wrist,
            pre_generated_subsets=pre_generated_subsets,
        )
        
        print(f"    Fixed SIC bootstrap (once, independent)...")
        boot_fixed_sic = random_bootstrap_fixed_sic(
            subset_indices=subset_indices,
            all_episode_indices=episode_indices,
            K_global=K_global, K_wrist=K_wrist,
            dbar_global=dbar_global, dbar_wrist=dbar_wrist,
            n_bootstrap=args.n_bootstrap, seed=args.seed,
            alpha=args.alpha,
            lambda_wrist=args.lambda_wrist,
            pre_generated_subsets=pre_generated_subsets,
        )

        subset_comparison[method_name]["bootstrap_global"] = boot_global
        subset_comparison[method_name]["bootstrap_wrist"] = boot_wrist
        subset_comparison[method_name]["bootstrap_combined"] = boot_combined
        subset_comparison[method_name]["bootstrap_fixed_sic"] = boot_fixed_sic

    print("\n[14/15] Evaluating Ours vs Uniform...")
    ours_uniform_comparison = {}
    if "Ours" in subset_comparison and "Uniform" in subset_comparison:
        ours_episodes = subsets_to_process.get("Ours", [])
        uniform_episodes = subsets_to_process.get("Uniform", [])

        ours_uniform_comparison = evaluate_ours_vs_uniform(
            subset_comparison["Ours"],
            subset_comparison["Uniform"],
            ours_episodes,
            uniform_episodes,
        )
        print(f"  Episode overlap: {ours_uniform_comparison.get('episode_overlap_count', 0)}")
        print(f"  Conclusion: {ours_uniform_comparison.get('conclusion', 'N/A')}")

    print("\n[15/15] Evaluating hypotheses...")
    analysis_results_for_hypotheses = {
        "validation": validation,
        "global_stats": global_stats,
        "wrist_stats": wrist_stats,
        "combined_stats": combined_stats,
        "spearman": spearman_results,
        "permutation_tests": perm_tests,
        "probe": probe_results,
        "neighbor_overlap": overlap_results,
        "grid_classifiability": grid_results,
        "subset_comparison": subset_comparison,
    }

    hypothesis_evaluation = evaluate_hypotheses(analysis_results_for_hypotheses)
    for h_name, h_data in hypothesis_evaluation.items():
        print(f"  {h_name}: {h_data['status']}")

    print("\n" + "=" * 60)
    print("Generating outputs...")
    print("=" * 60)

    analysis_results = {
        "n_episodes": n_episodes,
        "alignment": alignment_info,
        "validation": validation,
        "global_stats": global_stats,
        "wrist_stats": wrist_stats,
        "combined_stats": combined_stats,
        "global_normalized_stats": global_normalized_stats,
        "wrist_normalized_stats": wrist_normalized_stats,
        "spearman": spearman_results,
        "permutation_tests": perm_tests,
        "probe": probe_results,
        "neighbor_overlap": overlap_results,
        "grid_classifiability": grid_results,
        "subset_comparison": subset_comparison,
        "ours_uniform_comparison": ours_uniform_comparison,
        "hypothesis_evaluation": hypothesis_evaluation,
        "analysis_config": {
            "seed": args.seed,
            "n_bootstrap": args.n_bootstrap,
            "alpha": args.alpha,
            "lambda_wrist": args.lambda_wrist,
        },
        "fixed_universe": {
            "reference_episode_count": n_episodes,
            "dbar_global": dbar_global,
            "dbar_wrist": dbar_wrist,
            "dbar_fallback_used": dbar_fallback_used,
        },
    }

    print("\n  Generating report...")
    generate_report(analysis_results, output_dir)

    print("\n  Saving analysis summary...")
    save_analysis_summary(analysis_results, output_dir)

    print("\n  Saving embedding metrics...")
    save_embedding_metrics(analysis_results, output_dir)

    print("\n  Saving coverage comparison...")
    save_coverage_comparison(analysis_results, output_dir)

    print("\n  Generating visualizations...")
    generate_visualizations(phi_globals, phi_wrists, phi_combined, obj_init_positions, output_dir)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()