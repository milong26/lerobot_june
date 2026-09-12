#!/usr/bin/env python
"""
Distribution Analysis for Configuration-Aware Hierarchical Acquisition

Analyzes demonstration distribution differences between:
- Candidate Pool (all available episodes)
- Random Selection subset
- Ours (configuration-aware) Selection subset

Generates:
1. Configuration space PCA visualization
2. Region coverage comparison
3. Distribution statistics (coverage ratio, entropy, diversity metrics)
4. ICLR-style comparison figures

Usage:
    python analyze_distribution.py \
        --task Disassemble-v3 \
        --candidate-pool /path/to/episode_initial_states.json \
        --selection-ours /path/to/ours_subset.json \
        --selection-random /path/to/random_subset.json \
        --visual-embedding-dir /path/to/visual/embeddings \
        --action-descriptor-dir /path/to/action/descriptors \
        --output-dir personal/work2/distribution_analysis/results/
"""

import sys
import json
import logging
import argparse
import time
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
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
    print(f"Loaded {len(descriptors)} action descriptors (recursive)")
    return descriptors


def setup_logging(output_dir: Path) -> logging.Logger:
    log_file = output_dir / "analysis.log"
    logger = logging.getLogger("distribution_analysis")
    logger.setLevel(logging.DEBUG)
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


def compute_pca_2d(rand_vecs: Dict[int, np.ndarray]) -> Tuple[np.ndarray, List[int]]:
    indices = sorted(rand_vecs.keys())
    vecs = np.array([rand_vecs[i] for i in indices])
    pca = PCA(n_components=2, random_state=SEED)
    coords_2d = pca.fit_transform(vecs)
    return coords_2d, indices


def plot_configuration_space(
    candidate_coords: np.ndarray,
    candidate_indices: List[int],
    random_indices: List[int],
    ours_indices: List[int],
    output_dir: Path,
    logger: logging.Logger,
):
    idx_to_row = {ep: i for i, ep in enumerate(candidate_indices)}

    random_rows = [idx_to_row[ep] for ep in random_indices if ep in idx_to_row]
    ours_rows = [idx_to_row[ep] for ep in ours_indices if ep in idx_to_row]

    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    ax.scatter(candidate_coords[:, 0], candidate_coords[:, 1],
               c="lightgray", s=12, alpha=0.4, label="Candidate Pool", marker="o", zorder=1)
    ax.scatter(candidate_coords[random_rows, 0], candidate_coords[random_rows, 1],
               c="#e74c3c", s=25, alpha=0.85, label="Random Selection", marker="^", zorder=2)
    ax.scatter(candidate_coords[ours_rows, 0], candidate_coords[ours_rows, 1],
               c="#2ecc71", s=25, alpha=0.85, label="Ours Selection", marker="s", zorder=3)

    ax.set_title("Configuration Space", fontsize=14, fontweight="bold")
    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel("PC 2", fontsize=12)
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    pdf_path = output_dir / "configuration_space_comparison.pdf"
    png_path = output_dir / "configuration_space_comparison.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    logger.info(f"Configuration space figure saved: {pdf_path}, {png_path}")


def plot_region_coverage(
    episode_to_region: Dict[int, int],
    num_regions: int,
    random_indices: List[int],
    ours_indices: List[int],
    output_dir: Path,
    logger: logging.Logger,
) -> Dict:
    region_set = set(range(num_regions))

    random_regions = set()
    for ep in random_indices:
        if ep in episode_to_region:
            random_regions.add(episode_to_region[ep])

    ours_regions = set()
    for ep in ours_indices:
        if ep in episode_to_region:
            ours_regions.add(episode_to_region[ep])

    total_regions = len(region_set)
    random_covered = len(random_regions)
    ours_covered = len(ours_regions)
    random_ratio = random_covered / total_regions if total_regions > 0 else 0.0
    ours_ratio = ours_covered / total_regions if total_regions > 0 else 0.0

    region_sizes = {}
    for ep, rid in episode_to_region.items():
        region_sizes[rid] = region_sizes.get(rid, 0) + 1

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    x = np.arange(total_regions)
    width = 0.35

    random_coverage = [1 if r in random_regions else 0 for r in range(total_regions)]
    ours_coverage = [1 if r in ours_regions else 0 for r in range(total_regions)]

    ax.bar(x - width / 2, random_coverage, width, label="Random Selection", color="#e74c3c", alpha=0.8)
    ax.bar(x + width / 2, ours_coverage, width, label="Ours Selection", color="#2ecc71", alpha=0.8)

    ax.set_xlabel("Configuration Region ID", fontsize=12)
    ax.set_ylabel("Covered", fontsize=12)
    ax.set_title("Configuration Region Coverage Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(total_regions)], fontsize=7, rotation=90)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    pdf_path = output_dir / "region_coverage_comparison.pdf"
    png_path = output_dir / "region_coverage_comparison.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    logger.info(f"Region coverage figure saved: {pdf_path}, {png_path}")

    coverage_data = {
        "total_regions": total_regions,
        "random_covered_regions": random_covered,
        "ours_covered_regions": ours_covered,
        "random_coverage_ratio": round(random_ratio, 4),
        "ours_coverage_ratio": round(ours_ratio, 4),
        "random_region_ids": sorted(list(random_regions)),
        "ours_region_ids": sorted(list(ours_regions)),
    }

    csv_path = output_dir / "region_coverage.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Total Regions", "Covered Regions", "Coverage Ratio"])
        writer.writerow(["Random", total_regions, random_covered, f"{random_ratio:.4f}"])
        writer.writerow(["Ours", total_regions, ours_covered, f"{ours_ratio:.4f}"])
    logger.info(f"Region coverage CSV saved: {csv_path}")

    return coverage_data


def compute_distribution_statistics(
    rand_vecs: Dict[int, np.ndarray],
    visual_embeddings: Dict[int, np.ndarray],
    action_descriptors: Dict[int, np.ndarray],
    episode_to_region: Dict[int, int],
    num_regions: int,
    random_indices: List[int],
    ours_indices: List[int],
    candidate_indices: List[int],
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

    def _visual_diversity(vecs):
        return _mean_pairwise_distance(vecs)

    def _action_diversity(vecs):
        return _mean_pairwise_distance(vecs)

    random_vecs = _extract_vectors(rand_vecs, random_indices)
    ours_vecs = _extract_vectors(rand_vecs, ours_indices)
    random_visual = _extract_vectors(visual_embeddings, random_indices)
    ours_visual = _extract_vectors(visual_embeddings, ours_indices)
    random_action = _extract_vectors(action_descriptors, random_indices)
    ours_action = _extract_vectors(action_descriptors, ours_indices)

    stats = {
        "candidate_pool_size": len(candidate_indices),
        "random_subset_size": len(random_indices),
        "ours_subset_size": len(ours_indices),
        "num_regions": num_regions,
        "random": {
            "configuration_coverage_ratio": round(_coverage_ratio(random_indices), 4),
            "region_entropy": round(_region_entropy(random_indices), 4),
            "avg_nearest_neighbor_distance": round(_avg_nearest_neighbor(random_vecs), 4),
            "mean_pairwise_configuration_distance": round(_mean_pairwise_distance(random_vecs), 4),
            "visual_diversity": round(_visual_diversity(random_visual), 4),
            "action_diversity": round(_action_diversity(random_action), 4),
        },
        "ours": {
            "configuration_coverage_ratio": round(_coverage_ratio(ours_indices), 4),
            "region_entropy": round(_region_entropy(ours_indices), 4),
            "avg_nearest_neighbor_distance": round(_avg_nearest_neighbor(ours_vecs), 4),
            "mean_pairwise_configuration_distance": round(_mean_pairwise_distance(ours_vecs), 4),
            "visual_diversity": round(_visual_diversity(ours_visual), 4),
            "action_diversity": round(_action_diversity(ours_action), 4),
        },
    }

    logger.info("Distribution statistics computed:")
    for method in ["random", "ours"]:
        logger.info(f"  {method}: {stats[method]}")

    return stats


def save_statistics(stats: Dict, coverage_data: Dict, output_dir: Path, logger: logging.Logger):
    json_path = output_dir / "statistics.json"
    combined = {**stats, "region_coverage": coverage_data}
    with open(json_path, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"Statistics JSON saved: {json_path}")

    csv_path = output_dir / "statistics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Random", "Ours"])
        for metric in ["configuration_coverage_ratio", "region_entropy",
                        "avg_nearest_neighbor_distance", "mean_pairwise_configuration_distance",
                        "visual_diversity", "action_diversity"]:
            writer.writerow([
                metric,
                stats["random"][metric],
                stats["ours"][metric],
            ])
    logger.info(f"Statistics CSV saved: {csv_path}")


def plot_iclr_comparison(
    candidate_coords: np.ndarray,
    candidate_indices: List[int],
    random_indices: List[int],
    ours_indices: List[int],
    episode_to_region: Dict[int, int],
    num_regions: int,
    stats: Dict,
    output_dir: Path,
    logger: logging.Logger,
):
    idx_to_row = {ep: i for i, ep in enumerate(candidate_indices)}
    random_rows = [idx_to_row[ep] for ep in random_indices if ep in idx_to_row]
    ours_rows = [idx_to_row[ep] for ep in ours_indices if ep in idx_to_row]

    fig = plt.figure(figsize=(13.0, 5.5))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(candidate_coords[:, 0], candidate_coords[:, 1],
                c="lightgray", s=8, alpha=0.35, label="Candidate Pool", marker="o")
    ax1.set_title("Candidate Pool", fontsize=11, fontweight="bold")
    ax1.set_xlabel("PC 1", fontsize=9)
    ax1.set_ylabel("PC 2", fontsize=9)
    ax1.legend(fontsize=8, loc="best", framealpha=0.85)
    ax1.grid(True, alpha=0.25)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(candidate_coords[:, 0], candidate_coords[:, 1],
                c="lightgray", s=8, alpha=0.35, marker="o")
    ax2.scatter(candidate_coords[random_rows, 0], candidate_coords[random_rows, 1],
                c="#e74c3c", s=20, alpha=0.85, marker="^", label="Random Selection")
    ax2.set_title("Random Selection", fontsize=11, fontweight="bold")
    ax2.set_xlabel("PC 1", fontsize=9)
    ax2.set_ylabel("PC 2", fontsize=9)
    ax2.legend(fontsize=8, loc="best", framealpha=0.85)
    ax2.grid(True, alpha=0.25)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(candidate_coords[:, 0], candidate_coords[:, 1],
                c="lightgray", s=8, alpha=0.35, marker="o")
    ax3.scatter(candidate_coords[ours_rows, 0], candidate_coords[ours_rows, 1],
                c="#2ecc71", s=20, alpha=0.85, marker="s", label="Ours Selection")
    ax3.set_title("Ours Selection", fontsize=11, fontweight="bold")
    ax3.set_xlabel("PC 1", fontsize=9)
    ax3.set_ylabel("PC 2", fontsize=9)
    ax3.legend(fontsize=8, loc="best", framealpha=0.85)
    ax3.grid(True, alpha=0.25)

    plt.suptitle("Configuration-Space Distribution Comparison", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    pdf_path = output_dir / "distribution_comparison_iclr.pdf"
    png_path = output_dir / "distribution_comparison_iclr.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    logger.info(f"ICLR comparison figure saved: {pdf_path}, {png_path}")


def plot_region_coverage_bar(
    coverage_data: Dict,
    stats: Dict,
    output_dir: Path,
    logger: logging.Logger,
):
    fig, ax = plt.subplots(figsize=(6.5, 3.5))

    methods = ["Random", "Ours"]
    covered = [coverage_data["random_covered_regions"], coverage_data["ours_covered_regions"]]
    uncovered = [
        coverage_data["total_regions"] - coverage_data["random_covered_regions"],
        coverage_data["total_regions"] - coverage_data["ours_covered_regions"],
    ]

    x = np.arange(len(methods))
    width = 0.5

    bars1 = ax.bar(x, covered, width, label="Covered Regions", color="#2ecc71", alpha=0.85)
    bars2 = ax.bar(x, uncovered, width, bottom=covered, label="Uncovered Regions", color="#e74c3c", alpha=0.6)

    for i, (c, u) in enumerate(zip(covered, uncovered)):
        ax.text(i, c / 2, f"{c}", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        ax.text(i, c + u / 2, f"{u}", ha="center", va="center", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.set_ylabel("Number of Regions", fontsize=11)
    ax.set_title("Configuration Region Coverage", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    pdf_path = output_dir / "region_coverage_bar.pdf"
    png_path = output_dir / "region_coverage_bar.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    logger.info(f"Region coverage bar figure saved: {pdf_path}, {png_path}")


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
                    logger.info(f"Auto-resolved: {c}")
                    return c
            for c in known_subdirs:
                name = c.name.lower()
                if variant in name:
                    logger.info(f"Auto-resolved: {c}")
                    return c
    return None


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
    parser.add_argument("--visual-embedding-dir", type=str, default=None,
                        help="Path to visual embedding cache directory")
    parser.add_argument("--action-descriptor-dir", type=str, default=None,
                        help="Path to action descriptor cache directory")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Path to dataset directory (contains episode_initial_states.json)")
    parser.add_argument("--output-dir", type=str,
                        default="personal/work2/distribution_analysis/results",
                        help="Output directory for results")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info(f"=" * 60)
    logger.info(f"Distribution Analysis started for task: {args.task}")
    logger.info(f"=" * 60)

    work2 = Path(__file__).resolve().parent.parent

    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        dataset_dir = resolve_path_auto(work2, args.task, ["dataset_view", "dataset"], logger)
        if dataset_dir is None:
            logger.error(f"Cannot find dataset directory for task {args.task}. "
                         f"Please provide --dataset-dir explicitly.")
            sys.exit(1)

    if args.candidate_pool:
        pool_path = Path(args.candidate_pool)
    else:
        pool_path = dataset_dir / "episode_initial_states.json"
        if not pool_path.exists():
            logger.error(f"Cannot find episode_initial_states.json in {dataset_dir}. "
                         f"Please provide --candidate-pool explicitly.")
            sys.exit(1)

    if args.visual_embedding_dir:
        visual_dir = Path(args.visual_embedding_dir)
    else:
        visual_dir = resolve_path_auto(work2, args.task, ["embedding", "shared_embeddings"], logger)
        if visual_dir is None:
            logger.error(f"Cannot find visual embedding directory for task {args.task}. "
                         f"Please provide --visual-embedding-dir explicitly.")
            sys.exit(1)

    if args.action_descriptor_dir:
        action_dir = Path(args.action_descriptor_dir)
    else:
        action_dir = resolve_path_auto(work2, args.task, ["action_descriptor", "action_descriptors"], logger)
        if action_dir is None:
            logger.error(f"Cannot find action descriptor directory for task {args.task}. "
                         f"Please provide --action-descriptor-dir explicitly.")
            sys.exit(1)

    if args.selection_ours:
        ours_path = Path(args.selection_ours)
    else:
        ours_path = resolve_path_auto(work2, args.task, ["our_v5", "ours", "subzerocore"], logger)
        if ours_path is None:
            logger.error(f"Cannot find ours selection subset for task {args.task}. "
                         f"Please provide --selection-ours explicitly.")
            sys.exit(1)

    if args.selection_random:
        random_path = Path(args.selection_random)
    else:
        random_path = resolve_path_auto(work2, args.task, ["random"], logger)
        if random_path is None:
            logger.error(f"Cannot find random selection subset for task {args.task}. "
                         f"Please provide --selection-random explicitly.")
            sys.exit(1)

    logger.info(f"Dataset dir: {dataset_dir}")
    logger.info(f"Candidate pool: {pool_path}")
    logger.info(f"Visual embeddings: {visual_dir}")
    logger.info(f"Action descriptors: {action_dir}")
    logger.info(f"Ours selection: {ours_path}")
    logger.info(f"Random selection: {random_path}")

    candidate_indices = load_candidate_pool_indices(pool_path, logger)
    random_indices = load_subset_indices(random_path, logger)
    ours_indices = load_subset_indices(ours_path, logger)

    logger.info(f"Candidate pool size: {len(candidate_indices)}")
    logger.info(f"Random subset size: {len(random_indices)}")
    logger.info(f"Ours subset size: {len(ours_indices)}")

    logger.info("Loading episode data (rand_vec, visual embedding, action descriptor)...")
    rand_vecs, visual_embeddings, action_descriptors = load_all_episode_data(
        candidate_indices, dataset_dir, visual_dir, action_dir, logger
    )

    valid_candidate_indices = sorted(rand_vecs.keys())

    logger.info("Building rand_vec regions via KMeans...")
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

    logger.info("Computing PCA 2D projection...")
    pca_coords, pca_indices = compute_pca_2d(rand_vecs)

    logger.info("Generating configuration space visualization...")
    plot_configuration_space(pca_coords, pca_indices, random_indices, ours_indices, output_dir, logger)

    logger.info("Computing region coverage...")
    coverage_data = plot_region_coverage(
        episode_to_region, num_regions, random_indices, ours_indices, output_dir, logger
    )

    logger.info("Computing distribution statistics...")
    stats = compute_distribution_statistics(
        rand_vecs, visual_embeddings, action_descriptors,
        episode_to_region, num_regions,
        random_indices, ours_indices, valid_candidate_indices, logger
    )

    save_statistics(stats, coverage_data, output_dir, logger)

    logger.info("Generating ICLR-style comparison figure...")
    plot_iclr_comparison(
        pca_coords, pca_indices, random_indices, ours_indices,
        episode_to_region, num_regions, stats, output_dir, logger
    )

    logger.info("Generating region coverage bar chart...")
    plot_region_coverage_bar(coverage_data, stats, output_dir, logger)

    logger.info(f"=" * 60)
    logger.info(f"Analysis complete! All results saved to: {output_dir}")
    logger.info(f"=" * 60)

    print(f"\nAnalysis complete. Results saved to: {output_dir}")
    print(f"  - configuration_space_comparison.pdf/png")
    print(f"  - region_coverage_comparison.pdf/png")
    print(f"  - region_coverage_bar.pdf/png")
    print(f"  - distribution_comparison_iclr.pdf/png")
    print(f"  - statistics.json")
    print(f"  - statistics.csv")
    print(f"  - region_coverage.csv")
    print(f"  - analysis.log")


if __name__ == "__main__":
    main()