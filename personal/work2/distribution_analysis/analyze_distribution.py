#!/usr/bin/env python
"""
Distribution Analysis for Configuration-Aware Hierarchical Acquisition

Selection Distribution Visualization:
- Candidate Pool (all available episodes)
- Random Selection subset (fixed budget)
- DemInf Selection subset (fixed budget)
- Ours (configuration-aware) Selection subset (fixed budget)

Generates:
1. Configuration distribution comparison (4-panel main figure)
2. Distribution statistics (coverage, entropy, diversity, JS divergence, EMD)
3. Region coverage (auxiliary quantitative analysis)

Usage:
    python analyze_distribution.py \
        --task Disassemble-v3 \
        --dataset-dir /path/to/dataset_view/disassemble-v3_corner \
        --candidate-pool /path/to/episode_initial_states.json \
        --visual-embedding-dir /path/to/shared_embeddings/... \
        --action-descriptor-dir /path/to/action_descriptors/... \
        --selection-ours /path/to/ours_subset.json \
        --selection-random /path/to/random_subset.json \
        --selection-deminf /path/to/deminf_subset.json \
        --selection-budget 112 \
        --output-dir personal/work2/distribution_analysis/results/
"""

import sys
import json
import logging
import argparse
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

WORK2_ROOT = Path(__file__).resolve().parent.parent
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))

from our_v5.select_our_v5 import (
    load_visual_embeddings,
    load_rand_vecs,
    build_rand_vec_regions,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLORS = {
    "candidate": "#B0B0B0",
    "random":    "#E74C3C",
    "deminf":    "#3498DB",
    "ours":      "#2ECC71",
}

MARKERS = {
    "candidate": "o",
    "random":    "^",
    "deminf":    "D",
    "ours":      "s",
}

LABELS = {
    "candidate": "Candidate Pool",
    "random":    "Random",
    "deminf":    "DemInf",
    "ours":      "Ours",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_action_descriptors_recursive(descriptor_path: Path) -> Dict[int, np.ndarray]:
    descriptors = {}
    for f in sorted(descriptor_path.rglob("*.npy")):
        if f.name == "action_descriptor.npy":
            continue
        try:
            data = np.load(str(f), allow_pickle=True).item()
            ep_idx = data.get("episode_index")
            if ep_idx is not None and "action_descriptor" in data:
                descriptors[int(ep_idx)] = data["action_descriptor"]
        except Exception:
            continue
    return descriptors


def setup_logging(output_dir: Path) -> logging.Logger:
    log_file = output_dir / "analysis.log"
    logger = logging.getLogger("distribution_analysis")
    logger.setLevel(logging.DEBUG)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_subset_indices(subset_path: Path, logger: logging.Logger) -> List[int]:
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_path}")
    with open(subset_path, "r") as f:
        data = json.load(f)
    for key in ["selected_episode_indices", "selected_episode_ids"]:
        if key in data:
            indices = data[key]
            logger.info(f"Loaded {len(indices)} indices from {subset_path.name} (key={key})")
            return [int(x) for x in indices]
    raise ValueError(f"No episode index list found in {subset_path}. "
                     f"Expected keys: selected_episode_indices or selected_episode_ids")


def load_candidate_pool_indices(pool_path: Path, logger: logging.Logger) -> List[int]:
    if not pool_path.exists():
        raise FileNotFoundError(f"Candidate pool metadata not found: {pool_path}")
    with open(pool_path, "r") as f:
        metadata = json.load(f)
    episodes = metadata.get("episodes", [])
    indices = [int(ep["episode_index"]) for ep in episodes if "episode_index" in ep]
    logger.info(f"Loaded {len(indices)} candidate pool episodes from {pool_path.name}")
    return indices


def load_all_episode_data(
    candidate_indices: List[int],
    dataset_dir: Path,
    visual_embedding_dir: Path,
    action_descriptor_dir: Path,
    logger: logging.Logger,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    rand_vecs = load_rand_vecs(dataset_dir, candidate_indices)
    visual_embeddings = load_visual_embeddings(visual_embedding_dir)
    action_descriptors = load_action_descriptors_recursive(action_descriptor_dir)

    valid = [ep for ep in candidate_indices
             if ep in rand_vecs and ep in visual_embeddings and ep in action_descriptors]
    logger.info(f"Episodes with complete data (rand_vec+visual+action): {len(valid)} / {len(candidate_indices)}")

    if len(valid) == 0:
        raise RuntimeError("No episodes have complete data. Cannot proceed with analysis.")

    rand_vecs_filtered = {ep: rand_vecs[ep] for ep in valid}
    visual_filtered = {ep: visual_embeddings[ep] for ep in valid}
    action_filtered = {ep: action_descriptors[ep] for ep in valid}

    return rand_vecs_filtered, visual_filtered, action_filtered


# ---------------------------------------------------------------------------
# Budget selection helpers
# ---------------------------------------------------------------------------
def select_random_budget(candidate_indices: List[int], budget: int, seed: int = SEED) -> List[int]:
    rng = np.random.RandomState(seed)
    pool = list(candidate_indices)
    if len(pool) > budget:
        selected = rng.choice(pool, size=budget, replace=False).tolist()
    else:
        selected = pool
    return [int(x) for x in selected]


def select_budget(indices: List[int], budget: int) -> List[int]:
    if len(indices) <= budget:
        return list(indices)
    return list(indices[:budget])


# ---------------------------------------------------------------------------
# PCA projection (single fit on full candidate pool)
# ---------------------------------------------------------------------------
def compute_pca_2d(rand_vecs: Dict[int, np.ndarray]) -> Tuple[np.ndarray, List[int], PCA, StandardScaler]:
    indices = sorted(rand_vecs.keys())
    vecs = np.array([rand_vecs[i] for i in indices])

    scaler = StandardScaler()
    vecs_scaled = scaler.fit_transform(vecs)

    pca = PCA(n_components=2, random_state=SEED)
    coords_2d = pca.fit_transform(vecs_scaled)
    return coords_2d, indices, pca, scaler


def _indices_to_rows(selected_indices: List[int], candidate_indices: List[int]) -> List[int]:
    idx_to_row = {ep: i for i, ep in enumerate(candidate_indices)}
    return [idx_to_row[ep] for ep in selected_indices if ep in idx_to_row]


# ---------------------------------------------------------------------------
# Main visualization: 4-panel configuration distribution comparison
# ---------------------------------------------------------------------------
def visualize_configuration_distribution(
    candidate_coords: np.ndarray,
    candidate_indices: List[int],
    random_indices: List[int],
    deminf_indices: List[int],
    ours_indices: List[int],
    budget: int,
    output_dir: Path,
    logger: logging.Logger,
):
    random_rows = _indices_to_rows(random_indices, candidate_indices)
    deminf_rows = _indices_to_rows(deminf_indices, candidate_indices)
    ours_rows = _indices_to_rows(ours_indices, candidate_indices)

    x_min = candidate_coords[:, 0].min() - 0.5
    x_max = candidate_coords[:, 0].max() + 0.5
    y_min = candidate_coords[:, 1].min() - 0.5
    y_max = candidate_coords[:, 1].max() + 0.5

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 10.0))

    panels = [
        (axes[0, 0], "A", "Candidate Pool",
         [], None, None, len(candidate_indices)),
        (axes[0, 1], "B", f"Random (Budget={budget})",
         random_rows, COLORS["random"], MARKERS["random"], len(random_indices)),
        (axes[1, 0], "C", f"DemInf (Budget={budget})",
         deminf_rows, COLORS["deminf"], MARKERS["deminf"], len(deminf_indices)),
        (axes[1, 1], "D", f"Ours (Budget={budget})",
         ours_rows, COLORS["ours"], MARKERS["ours"], len(ours_indices)),
    ]

    for ax, letter, title, rows, color, marker, count in panels:
        # Background: full candidate pool
        ax.scatter(candidate_coords[:, 0], candidate_coords[:, 1],
                   c=COLORS["candidate"], s=8, alpha=0.30, marker="o", zorder=1)
        # Overlay selected episodes
        if rows:
            ax.scatter(candidate_coords[rows, 0], candidate_coords[rows, 1],
                       c=color, s=30, alpha=0.88, marker=marker, zorder=2,
                       edgecolors="white", linewidths=0.6)
        ax.set_title(f"{letter}: {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("PC 1", fontsize=10)
        ax.set_ylabel("PC 2", fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.25)
        ax.text(0.02, 0.98, f"n={count}", transform=ax.transAxes,
                fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle("Selection Distribution in Configuration Space",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    for fmt, dpi in [("pdf", 300), ("png", 300)]:
        path = output_dir / f"configuration_distribution_comparison.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", format=fmt)
    plt.close(fig)
    logger.info("Configuration distribution comparison figure saved")


# ---------------------------------------------------------------------------
# Distribution statistics
# ---------------------------------------------------------------------------
def compute_distribution_statistics(
    rand_vecs: Dict[int, np.ndarray],
    visual_embeddings: Dict[int, np.ndarray],
    action_descriptors: Dict[int, np.ndarray],
    episode_to_region: Dict[int, int],
    num_regions: int,
    candidate_indices: List[int],
    random_indices: List[int],
    deminf_indices: List[int],
    ours_indices: List[int],
    budget: int,
    logger: logging.Logger,
) -> Dict:
    def _extract_vectors(data_dict, indices):
        return [data_dict[ep] for ep in indices if ep in data_dict]

    def _coverage_ratio(indices):
        covered = set()
        for ep in indices:
            if ep in episode_to_region:
                covered.add(episode_to_region[ep])
        return len(covered) / num_regions if num_regions > 0 else 0.0

    def _region_entropy(indices):
        region_counts = {}
        for ep in indices:
            if ep in episode_to_region:
                rid = episode_to_region[ep]
                region_counts[rid] = region_counts.get(rid, 0) + 1
        total = sum(region_counts.values())
        if total == 0:
            return 0.0
        probs = np.array([c / total for c in region_counts.values()])
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    def _avg_nearest_neighbor(vecs):
        if len(vecs) < 2:
            return 0.0
        arr = np.array(vecs)
        nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
        nn.fit(arr)
        distances, _ = nn.kneighbors(arr)
        return float(np.mean(distances[:, 1]))

    def _mean_pairwise_distance(vecs):
        if len(vecs) < 2:
            return 0.0
        arr = np.array(vecs)
        n = len(arr)
        total = 0.0
        count = 0
        step = max(1, n // 200)
        for i in range(0, n, step):
            for j in range(i + 1, n, step):
                total += np.linalg.norm(arr[i] - arr[j])
                count += 1
        return float(total / count) if count > 0 else 0.0

    def _compute_js_divergence(indices):
        region_counts = np.zeros(num_regions)
        for ep in indices:
            rid = episode_to_region.get(ep)
            if rid is not None:
                region_counts[rid] += 1
        total_sel = region_counts.sum()
        if total_sel == 0:
            return 0.0
        p = region_counts / total_sel

        candidate_counts = np.zeros(num_regions)
        for ep in candidate_indices:
            rid = episode_to_region.get(ep)
            if rid is not None:
                candidate_counts[rid] += 1
        total_cand = candidate_counts.sum()
        if total_cand == 0:
            return 0.0
        q = candidate_counts / total_cand

        p_smooth = p + 1e-10
        q_smooth = q + 1e-10
        p_norm = p_smooth / p_smooth.sum()
        q_norm = q_smooth / q_smooth.sum()
        return float(jensenshannon(p_norm, q_norm))

    def _compute_emd(indices):
        region_counts = np.zeros(num_regions)
        for ep in indices:
            rid = episode_to_region.get(ep)
            if rid is not None:
                region_counts[rid] += 1
        total_sel = region_counts.sum()
        if total_sel == 0:
            return 0.0
        p = region_counts / total_sel

        candidate_counts = np.zeros(num_regions)
        for ep in candidate_indices:
            rid = episode_to_region.get(ep)
            if rid is not None:
                candidate_counts[rid] += 1
        total_cand = candidate_counts.sum()
        if total_cand == 0:
            return 0.0
        q = candidate_counts / total_cand

        distances_1d = np.arange(num_regions, dtype=float)
        return float(wasserstein_distance(distances_1d, distances_1d, p, q))

    def _compute_method_stats(indices):
        vecs = _extract_vectors(rand_vecs, indices)
        vis = _extract_vectors(visual_embeddings, indices)
        act = _extract_vectors(action_descriptors, indices)
        return {
            "method": "",
            "selection_budget": budget,
            "coverage_ratio": round(_coverage_ratio(indices), 4),
            "entropy": round(_region_entropy(indices), 4),
            "nearest_neighbor_distance": round(_avg_nearest_neighbor(vecs), 4),
            "pairwise_configuration_distance": round(_mean_pairwise_distance(vecs), 4),
            "visual_diversity": round(_mean_pairwise_distance(vis), 4),
            "action_diversity": round(_mean_pairwise_distance(act), 4),
            "js_divergence": round(_compute_js_divergence(indices), 4),
            "emd_distance": round(_compute_emd(indices), 4),
        }

    random_stats = _compute_method_stats(random_indices)
    random_stats["method"] = "random"
    deminf_stats = _compute_method_stats(deminf_indices)
    deminf_stats["method"] = "deminf"
    ours_stats = _compute_method_stats(ours_indices)
    ours_stats["method"] = "ours"

    stats = {
        "candidate_pool_size": len(candidate_indices),
        "selection_budget": budget,
        "num_regions": num_regions,
        "random": random_stats,
        "deminf": deminf_stats,
        "ours": ours_stats,
    }

    logger.info("Distribution statistics computed:")
    for method in ["random", "deminf", "ours"]:
        logger.info(f"  {method}: {stats[method]}")

    return stats


def save_statistics(stats: Dict, output_dir: Path, logger: logging.Logger):
    json_path = output_dir / "statistics.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistics JSON saved: {json_path}")

    csv_path = output_dir / "statistics.csv"
    metrics = [
        "selection_budget",
        "coverage_ratio",
        "entropy",
        "nearest_neighbor_distance",
        "pairwise_configuration_distance",
        "visual_diversity",
        "action_diversity",
        "js_divergence",
        "emd_distance",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method"] + metrics)
        for method in ["random", "deminf", "ours"]:
            row = [method] + [stats[method][m] for m in metrics]
            writer.writerow(row)
    logger.info(f"Statistics CSV saved: {csv_path}")


# ---------------------------------------------------------------------------
# Region coverage (auxiliary quantitative analysis)
# ---------------------------------------------------------------------------
def compute_region_coverage(
    episode_to_region: Dict[int, int],
    num_regions: int,
    candidate_indices: List[int],
    random_indices: List[int],
    deminf_indices: List[int],
    ours_indices: List[int],
    output_dir: Path,
    logger: logging.Logger,
) -> Dict:
    methods = ["Candidate Pool", "Random", "DemInf", "Ours"]
    method_indices = [candidate_indices, random_indices, deminf_indices, ours_indices]

    region_counts = np.zeros((len(methods), num_regions), dtype=int)
    for mi, indices in enumerate(method_indices):
        for ep in indices:
            rid = episode_to_region.get(ep)
            if rid is not None:
                region_counts[mi, rid] += 1

    # Bar chart
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    x = np.arange(num_regions)
    width = 0.20
    bar_colors = [COLORS["candidate"], COLORS["random"], COLORS["deminf"], COLORS["ours"]]
    for mi in range(len(methods)):
        ax.bar(x + mi * width, region_counts[mi, :], width,
               label=methods[mi], color=bar_colors[mi], alpha=0.8)

    ax.set_xlabel("Configuration Region ID", fontsize=11)
    ax.set_ylabel("Episode Count", fontsize=11)
    ax.set_title("Region Coverage Comparison", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([str(i) for i in range(num_regions)], fontsize=6, rotation=90)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    for fmt, dpi in [("pdf", 300), ("png", 300)]:
        path = output_dir / f"region_coverage_comparison.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight", format=fmt)
    plt.close(fig)
    logger.info("Region coverage comparison figure saved")

    # CSV
    csv_path = output_dir / "region_coverage.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["region_id", "candidate_count", "random_count", "deminf_count", "ours_count"])
        for j in range(num_regions):
            writer.writerow([j, int(region_counts[0, j]), int(region_counts[1, j]),
                             int(region_counts[2, j]), int(region_counts[3, j])])
    logger.info(f"Region coverage CSV saved: {csv_path}")

    coverage_data = {}
    for j in range(num_regions):
        coverage_data[f"region_{j}"] = {
            "region_id": j,
            "candidate_count": int(region_counts[0, j]),
            "random_count": int(region_counts[1, j]),
            "deminf_count": int(region_counts[2, j]),
            "ours_count": int(region_counts[3, j]),
        }
    return coverage_data


# ---------------------------------------------------------------------------
# Auto path resolution
# ---------------------------------------------------------------------------
def resolve_path_auto(base_dir: Path, task: str, pattern_keywords: List[str], logger: logging.Logger) -> Optional[Path]:
    task_lower = task.lower()
    task_underscore = task.replace("-", "_").lower()
    task_no_v3 = task.replace("-v3", "").replace("-", "_").lower()
    task_variants = [task_lower, task_underscore, task_no_v3]

    known_subdirs = []
    for d in base_dir.iterdir():
        if d.is_dir():
            known_subdirs.append(d)
            for sub in d.iterdir():
                if sub.is_dir():
                    known_subdirs.append(sub)

    for variant in task_variants:
        for keyword in pattern_keywords:
            for c in known_subdirs:
                name = c.name.lower()
                if variant in name and keyword in name:
                    logger.info(f"Auto-resolved dir: {c}")
                    return c
            for c in known_subdirs:
                name = c.name.lower()
                if variant in name:
                    logger.info(f"Auto-resolved dir: {c}")
                    return c
    return None


def find_subset_json_in_dir(dir_path: Path, logger: logging.Logger) -> Optional[Path]:
    if dir_path.is_file() and dir_path.suffix == ".json":
        return dir_path
    for f in sorted(dir_path.rglob("*.json")):
        if f.name == "info.json":
            continue
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
            if "selected_episode_indices" in data or "selected_episode_ids" in data:
                logger.info(f"Found subset JSON: {f}")
                return f
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Distribution Analysis for Configuration-Aware Selection")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name, e.g. Disassemble-v3, Pick-Place-v3, Coffee-Button-v3")
    parser.add_argument("--candidate-pool", type=str, default=None,
                        help="Path to episode_initial_states.json (candidate pool metadata)")
    parser.add_argument("--selection-ours", type=str, default=None,
                        help="Path to ours selection subset JSON")
    parser.add_argument("--selection-random", type=str, default=None,
                        help="Path to random selection subset JSON")
    parser.add_argument("--selection-deminf", type=str, default=None,
                        help="Path to DemInf selection subset JSON")
    parser.add_argument("--visual-embedding-dir", type=str, default=None,
                        help="Path to visual embedding cache directory")
    parser.add_argument("--action-descriptor-dir", type=str, default=None,
                        help="Path to action descriptor cache directory")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Path to dataset directory (contains episode_initial_states.json)")
    parser.add_argument("--selection-budget", type=int, default=112,
                        help="Fixed episode budget for all selection methods (default: 112)")
    parser.add_argument("--output-dir", type=str,
                        default="personal/work2/distribution_analysis/results",
                        help="Output directory for results")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("=" * 60)
    logger.info(f"Distribution Analysis started for task: {args.task}")
    logger.info(f"Selection budget: {args.selection_budget}")
    logger.info("=" * 60)

    work2 = Path(__file__).resolve().parent.parent

    # Resolve dataset dir
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        dataset_dir = resolve_path_auto(work2, args.task, ["dataset_view", "dataset"], logger)
        if dataset_dir is None:
            logger.error(f"Cannot find dataset directory for task {args.task}. "
                         f"Please provide --dataset-dir explicitly.")
            sys.exit(1)

    # Resolve candidate pool
    if args.candidate_pool:
        pool_path = Path(args.candidate_pool)
    else:
        pool_path = dataset_dir / "episode_initial_states.json"
        if not pool_path.exists():
            logger.error(f"Cannot find episode_initial_states.json in {dataset_dir}. "
                         f"Please provide --candidate-pool explicitly.")
            sys.exit(1)

    # Resolve visual embeddings
    if args.visual_embedding_dir:
        visual_dir = Path(args.visual_embedding_dir)
    else:
        visual_dir = resolve_path_auto(work2, args.task, ["embedding", "shared_embeddings"], logger)
        if visual_dir is None:
            logger.error(f"Cannot find visual embedding directory for task {args.task}. "
                         f"Please provide --visual-embedding-dir explicitly.")
            sys.exit(1)

    # Resolve action descriptors
    if args.action_descriptor_dir:
        action_dir = Path(args.action_descriptor_dir)
    else:
        action_dir = resolve_path_auto(work2, args.task, ["action_descriptor", "action_descriptors"], logger)
        if action_dir is None:
            logger.error(f"Cannot find action descriptor directory for task {args.task}. "
                         f"Please provide --action-descriptor-dir explicitly.")
            sys.exit(1)

    # Resolve ours selection
    if args.selection_ours:
        ours_path = Path(args.selection_ours)
    else:
        ours_dir = resolve_path_auto(work2, args.task, ["our_v5", "ours", "subzerocore"], logger)
        if ours_dir is None:
            logger.error(f"Cannot find ours selection subset for task {args.task}. "
                         f"Please provide --selection-ours explicitly.")
            sys.exit(1)
        ours_path = find_subset_json_in_dir(ours_dir, logger)
        if ours_path is None:
            logger.error(f"Cannot find subset JSON in {ours_dir}. "
                         f"Please provide --selection-ours explicitly.")
            sys.exit(1)

    # Resolve random selection
    if args.selection_random:
        random_path = Path(args.selection_random)
    else:
        random_dir = resolve_path_auto(work2, args.task, ["random"], logger)
        if random_dir is None:
            logger.error(f"Cannot find random selection subset for task {args.task}. "
                         f"Please provide --selection-random explicitly.")
            sys.exit(1)
        random_path = find_subset_json_in_dir(random_dir, logger)
        if random_path is None:
            logger.error(f"Cannot find subset JSON in {random_dir}. "
                         f"Please provide --selection-random explicitly.")
            sys.exit(1)

    # Resolve DemInf selection (optional)
    deminf_path = None
    if args.selection_deminf:
        deminf_path = Path(args.selection_deminf)
    else:
        deminf_dir = resolve_path_auto(work2, args.task, ["deminf"], logger)
        if deminf_dir is not None:
            deminf_path = find_subset_json_in_dir(deminf_dir, logger)
        if deminf_path is None:
            logger.warning("DemInf selection file not found. DemInf analysis will be skipped.")

    logger.info(f"Dataset dir: {dataset_dir}")
    logger.info(f"Candidate pool: {pool_path}")
    logger.info(f"Visual embeddings: {visual_dir}")
    logger.info(f"Action descriptors: {action_dir}")
    logger.info(f"Ours selection: {ours_path}")
    logger.info(f"Random selection: {random_path}")
    logger.info(f"DemInf selection: {deminf_path}")

    # 1. Load candidate pool episode index
    candidate_indices = load_candidate_pool_indices(pool_path, logger)

    # 2. Load rand_vec, visual embedding, action descriptor
    logger.info("Loading episode data (rand_vec, visual embedding, action descriptor)...")
    rand_vecs, visual_embeddings, action_descriptors = load_all_episode_data(
        candidate_indices, dataset_dir, visual_dir, action_dir, logger
    )
    valid_candidate_indices = sorted(rand_vecs.keys())

    # 3. Load Random, DemInf, Ours selection index
    raw_random_indices = load_subset_indices(random_path, logger)
    raw_ours_indices = load_subset_indices(ours_path, logger)
    raw_deminf_indices = []
    if deminf_path is not None and deminf_path.exists():
        raw_deminf_indices = load_subset_indices(deminf_path, logger)

    logger.info(f"Candidate pool size: {len(candidate_indices)}")
    logger.info(f"Raw random subset size: {len(raw_random_indices)}")
    logger.info(f"Raw ours subset size: {len(raw_ours_indices)}")
    if raw_deminf_indices:
        logger.info(f"Raw DemInf subset size: {len(raw_deminf_indices)}")

    # 4. Apply selection budget uniformly
    budget = args.selection_budget
    logger.info(f"Applying selection budget: {budget}")

    random_indices = select_random_budget(valid_candidate_indices, budget, seed=SEED)
    ours_indices = select_budget(raw_ours_indices, budget)
    deminf_indices = select_budget(raw_deminf_indices, budget) if raw_deminf_indices else []

    logger.info(f"Random (budget={budget}): {len(random_indices)} episodes")
    logger.info(f"Ours (budget={budget}): {len(ours_indices)} episodes")
    if deminf_indices:
        logger.info(f"DemInf (budget={budget}): {len(deminf_indices)} episodes")

    # 5. Build regions (auxiliary)
    logger.info("Building rand_vec regions via KMeans (auxiliary)...")
    episode_data_for_regions = {
        ep: {
            "rand_vec": rand_vecs[ep],
            "visual_embedding": visual_embeddings[ep],
            "action_descriptor": action_descriptors[ep],
        }
        for ep in valid_candidate_indices
    }
    episode_to_region, num_regions = build_rand_vec_regions(
        episode_data_for_regions,
        region_ratio=0.1,
        min_regions=16,
        max_regions=128,
        seed=SEED,
    )
    logger.info(f"Created {num_regions} configuration regions")

    # 6. PCA projection (single fit on full candidate pool)
    logger.info("Computing PCA 2D projection (StandardScaler + PCA)...")
    pca_coords, pca_indices, pca_model, scaler = compute_pca_2d(rand_vecs)

    # 7. Generate configuration distribution 4-panel main figure
    logger.info("Generating configuration distribution comparison figure...")
    visualize_configuration_distribution(
        pca_coords, pca_indices, random_indices, deminf_indices, ours_indices,
        budget, output_dir, logger
    )

    # 8. Compute distribution statistics
    logger.info("Computing distribution statistics...")
    stats = compute_distribution_statistics(
        rand_vecs, visual_embeddings, action_descriptors,
        episode_to_region, num_regions,
        valid_candidate_indices, random_indices, deminf_indices, ours_indices,
        budget, logger
    )
    save_statistics(stats, output_dir, logger)

    # 9. Region coverage (auxiliary)
    logger.info("Computing region coverage (auxiliary)...")
    compute_region_coverage(
        episode_to_region, num_regions,
        valid_candidate_indices, random_indices, deminf_indices, ours_indices,
        output_dir, logger
    )

    logger.info("=" * 60)
    logger.info(f"Analysis complete! All results saved to: {output_dir}")
    logger.info("=" * 60)

    print(f"\nAnalysis complete. Results saved to: {output_dir}")
    print(f"  - configuration_distribution_comparison.png/pdf  (main figure)")
    print(f"  - statistics.json")
    print(f"  - statistics.csv")
    print(f"  - region_coverage_comparison.png/pdf  (auxiliary)")
    print(f"  - region_coverage.csv  (auxiliary)")
    print(f"  - analysis.log")


if __name__ == "__main__":
    main()