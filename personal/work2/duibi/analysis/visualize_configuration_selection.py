#!/usr/bin/env python
"""
Configuration-Aware Selection Visualization for Paper

Generates two core figures for the "Data Selection Analysis" section:

1. configuration_space_comparison.png
   - 4-panel figure showing raw configuration space (no PCA)
   - Automatically selects the two dimensions with highest variance
   - Demonstrates that Ours covers candidate distribution better than Random/DemInf

2. configuration_distribution_metrics.png
   - Bar chart comparing coverage_ratio, entropy, pairwise_configuration_distance,
     JS_divergence, EMD_distance across Random, DemInf, Ours
   - Values displayed directly on bars

PCA visualizations are moved to supplementary/ and only generated with --enable_pca.

Usage:
    python visualize_configuration_selection.py \\
        --candidate-pool /path/to/episode_initial_states.json \\
        --selection-ours /path/to/ours.json \\
        --selection-random /path/to/random.json \\
        --selection-deminf /path/to/deminf.json \\
        --budget 112 \\
        --output-dir personal/work2/duibi/results/configuration_visualization
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

SEED = 42
DPI = 300
FIGSIZE_2X2 = (13.0, 10.0)
FIGSIZE_METRICS = (14.0, 8.0)

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_candidate_pool(pool_path: Path) -> Tuple[List[int], np.ndarray]:
    """Load candidate episode indices and their rand_vec features."""
    with open(pool_path, "r") as f:
        metadata = json.load(f)

    episodes = metadata.get("episodes", [])
    indices = []
    features = []

    for ep in episodes:
        ep_idx = ep.get("episode_index")
        rand_vec = ep.get("rand_vec")
        if ep_idx is not None and rand_vec is not None:
            indices.append(int(ep_idx))
            features.append(rand_vec)

    features = np.array(features)
    print(f"\n[Configuration Analysis]")
    print(f"  Candidate pool: {len(indices)} episodes")
    print(f"  Feature dimension: {features.shape[1]}")
    return indices, features


def load_selection_indices(json_path: Path, label: str) -> List[int]:
    """Load selected episode indices from a JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    for key in ["selected_episode_indices", "selected_episode_ids"]:
        if key in data:
            indices = [int(x) for x in data[key]]
            print(f"  {label}: {len(indices)} episodes loaded")
            return indices

    raise ValueError(f"No episode index list found in {json_path}")


def load_metrics(metrics_path: Path) -> Dict:
    """Load pre-computed distribution statistics from JSON."""
    with open(metrics_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data consistency checks
# ---------------------------------------------------------------------------
def validate_data(
    candidate_indices: List[int],
    candidate_features: np.ndarray,
    random_indices: List[int],
    deminf_indices: List[int],
    ours_indices: List[int],
    budget: int,
):
    """Assert data consistency before visualization."""
    assert len(candidate_indices) == len(candidate_features), (
        f"Candidate pool size mismatch: {len(candidate_indices)} indices vs "
        f"{len(candidate_features)} features"
    )

    assert len(random_indices) == budget, (
        f"Random selection size {len(random_indices)} != budget {budget}"
    )
    assert len(deminf_indices) == budget, (
        f"DemInf selection size {len(deminf_indices)} != budget {budget}"
    )
    assert len(ours_indices) == budget, (
        f"Ours selection size {len(ours_indices)} != budget {budget}"
    )

    random_set = set(random_indices)
    deminf_set = set(deminf_indices)
    ours_set = set(ours_indices)

    assert len(random_set) == budget, "Random has duplicate indices"
    assert len(deminf_set) == budget, "DemInf has duplicate indices"
    assert len(ours_set) == budget, "Ours has duplicate indices"

    print(f"  Budget: {budget}")
    print(f"  Data consistency check: PASSED")


# ---------------------------------------------------------------------------
# Dimension selection: pick two dimensions with highest variance
# ---------------------------------------------------------------------------
def select_top_variance_dims(features: np.ndarray) -> Tuple[int, int]:
    """Select the two dimensions with highest variance for 2D visualization."""
    variances = np.var(features, axis=0)
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)

    print(f"\n  Dimension statistics:")
    for i in range(features.shape[1]):
        print(f"    Dim {i}: mean={means[i]:.4f}, std={stds[i]:.4f}, var={variances[i]:.4f}")

    top2 = np.argsort(variances)[::-1][:2]
    dim_i, dim_j = int(top2[0]), int(top2[1])

    print(f"  Visualization dimensions: Dim {dim_i} vs Dim {dim_j}")
    return dim_i, dim_j


# ---------------------------------------------------------------------------
# Figure 1: Configuration Space Comparison (4-panel, raw features, no PCA)
# ---------------------------------------------------------------------------
def visualize_configuration_space(
    candidate_indices: List[int],
    candidate_features: np.ndarray,
    random_indices: List[int],
    deminf_indices: List[int],
    ours_indices: List[int],
    budget: int,
    output_dir: Path,
):
    """Generate 4-panel configuration space comparison using raw features."""
    dim_i, dim_j = select_top_variance_dims(candidate_features)

    coords = candidate_features[:, [dim_i, dim_j]]

    x_min = coords[:, 0].min() - 0.05 * (coords[:, 0].max() - coords[:, 0].min())
    x_max = coords[:, 0].max() + 0.05 * (coords[:, 0].max() - coords[:, 0].min())
    y_min = coords[:, 1].min() - 0.05 * (coords[:, 1].max() - coords[:, 1].min())
    y_max = coords[:, 1].max() + 0.05 * (coords[:, 1].max() - coords[:, 1].min())

    idx_map = {ep: i for i, ep in enumerate(candidate_indices)}

    def get_rows(indices):
        return [idx_map[ep] for ep in indices if ep in idx_map]

    random_rows = get_rows(random_indices)
    deminf_rows = get_rows(deminf_indices)
    ours_rows = get_rows(ours_indices)

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_2X2)

    panels = [
        (axes[0, 0], "a", "Candidate Pool",
         [], None, None, len(candidate_indices)),
        (axes[0, 1], "b", f"Random (Budget={budget})",
         random_rows, COLORS["random"], MARKERS["random"], len(random_indices)),
        (axes[1, 0], "c", f"DemInf (Budget={budget})",
         deminf_rows, COLORS["deminf"], MARKERS["deminf"], len(deminf_indices)),
        (axes[1, 1], "d", f"Ours (Budget={budget})",
         ours_rows, COLORS["ours"], MARKERS["ours"], len(ours_indices)),
    ]

    for ax, letter, title, rows, color, marker, count in panels:
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=COLORS["candidate"], s=10, alpha=0.25, marker="o", zorder=1)

        if rows:
            ax.scatter(coords[rows, 0], coords[rows, 1],
                       c=color, s=35, alpha=0.90, marker=marker, zorder=2,
                       edgecolors="white", linewidths=0.7,
                       label=f"{title} (n={count})")

        ax.set_title(f"({letter}) {title}", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel(f"Dimension {dim_i}", fontsize=11)
        ax.set_ylabel(f"Dimension {dim_j}", fontsize=11)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.20, linestyle="--")
        ax.legend(fontsize=9, loc="best", framealpha=0.9)

    fig.suptitle(
        f"Configuration Space: Dimension {dim_i} vs Dimension {dim_j}",
        fontsize=15, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    out_path = output_dir / "configuration_space_comparison.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: Distribution Metrics Comparison (bar chart)
# ---------------------------------------------------------------------------
def visualize_metrics(
    metrics: Dict,
    output_dir: Path,
):
    """Generate bar chart comparing distribution metrics across methods."""
    methods = ["random", "deminf", "ours"]
    method_labels = ["Random", "DemInf", "Ours"]
    method_colors = [COLORS["random"], COLORS["deminf"], COLORS["ours"]]

    metric_keys = [
        "coverage_ratio",
        "entropy",
        "pairwise_configuration_distance",
        "js_divergence",
        "emd_distance",
    ]

    metric_labels = [
        "Coverage Ratio",
        "Region Entropy",
        "Pairwise Config. Distance",
        "JS Divergence",
        "EMD Distance",
    ]

    n_metrics = len(metric_keys)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=FIGSIZE_METRICS)
    axes = axes.flatten()

    x = np.arange(len(methods))
    width = 0.5

    for mi, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[mi]
        values = [metrics[m][key] for m in methods]

        bars = ax.bar(x, values, width, color=method_colors, alpha=0.85, edgecolor="white", linewidth=0.5)

        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(method_labels, fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.20, axis="y", linestyle="--")

        ax.set_ylim(0, max(values) * 1.25)

    for mi in range(n_metrics, len(axes)):
        axes[mi].axis("off")

    fig.suptitle("Distribution Metrics Comparison (Budget=112)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = output_dir / "configuration_distribution_metrics.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Optional: PCA visualization (moved to supplementary)
# ---------------------------------------------------------------------------
def visualize_pca_supplementary(
    candidate_indices: List[int],
    candidate_features: np.ndarray,
    random_indices: List[int],
    deminf_indices: List[int],
    ours_indices: List[int],
    budget: int,
    output_dir: Path,
):
    """Generate PCA-based visualization for supplementary materials."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(candidate_features)

    n_components = min(3, candidate_features.shape[1])
    pca = PCA(n_components=n_components, random_state=SEED)
    pca_result = pca.fit_transform(features_scaled)

    print(f"\n  PCA explained variance ratio: {pca.explained_variance_ratio_}")

    idx_map = {ep: i for i, ep in enumerate(candidate_indices)}

    def get_rows(indices):
        return [idx_map[ep] for ep in indices if ep in idx_map]

    random_rows = get_rows(random_indices)
    deminf_rows = get_rows(deminf_indices)
    ours_rows = get_rows(ours_indices)

    pc_pairs = [(0, 1), (0, 2), (1, 2)] if n_components >= 3 else [(0, 1)]

    for pi, pj in pc_pairs:
        coords = pca_result[:, [pi, pj]]

        x_min = coords[:, 0].min() - 0.5
        x_max = coords[:, 0].max() + 0.5
        y_min = coords[:, 1].min() - 0.5
        y_max = coords[:, 1].max() + 0.5

        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_2X2)

        panels = [
            (axes[0, 0], "a", "Candidate Pool",
             [], None, None, len(candidate_indices)),
            (axes[0, 1], "b", f"Random (Budget={budget})",
             random_rows, COLORS["random"], MARKERS["random"], len(random_indices)),
            (axes[1, 0], "c", f"DemInf (Budget={budget})",
             deminf_rows, COLORS["deminf"], MARKERS["deminf"], len(deminf_indices)),
            (axes[1, 1], "d", f"Ours (Budget={budget})",
             ours_rows, COLORS["ours"], MARKERS["ours"], len(ours_indices)),
        ]

        for ax, letter, title, rows, color, marker, count in panels:
            ax.scatter(coords[:, 0], coords[:, 1],
                       c=COLORS["candidate"], s=10, alpha=0.25, marker="o", zorder=1)
            if rows:
                ax.scatter(coords[rows, 0], coords[rows, 1],
                           c=color, s=35, alpha=0.90, marker=marker, zorder=2,
                           edgecolors="white", linewidths=0.7,
                           label=f"{title} (n={count})")

            ax.set_title(f"({letter}) {title}", fontsize=13, fontweight="bold", pad=10)
            ax.set_xlabel(f"PC {pi+1}", fontsize=11)
            ax.set_ylabel(f"PC {pj+1}", fontsize=11)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.20, linestyle="--")
            ax.legend(fontsize=9, loc="best", framealpha=0.9)

        fig.suptitle(f"PCA Projection (Supplementary): PC{pi+1} vs PC{pj+1}",
                     fontsize=14, fontweight="bold", y=1.01)
        plt.tight_layout()

        out_path = output_dir / f"pca_PC{pi+1}_PC{pj+1}.png"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved (supplementary): {out_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Configuration-Aware Selection Visualization for Paper"
    )
    parser.add_argument("--candidate-pool", type=str, required=True,
                        help="Path to episode_initial_states.json")
    parser.add_argument("--selection-ours", type=str, required=True,
                        help="Path to ours selection JSON")
    parser.add_argument("--selection-random", type=str, required=True,
                        help="Path to random selection JSON")
    parser.add_argument("--selection-deminf", type=str, required=True,
                        help="Path to DemInf selection JSON")
    parser.add_argument("--metrics-json", type=str, default=None,
                        help="Path to pre-computed statistics.json (optional)")
    parser.add_argument("--budget", type=int, default=112,
                        help="Selection budget (default: 112)")
    parser.add_argument("--output-dir", type=str,
                        default="personal/work2/duibi/results/configuration_visualization",
                        help="Output directory for figures")
    parser.add_argument("--enable-pca", action="store_true",
                        help="Also generate PCA visualizations (saved to supplementary/)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    supplementary_dir = output_dir.parent / "supplementary"
    supplementary_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Configuration-Aware Selection Visualization")
    print("=" * 60)

    # 1. Load candidate pool
    candidate_indices, candidate_features = load_candidate_pool(Path(args.candidate_pool))

    # 2. Load selection indices
    random_indices = load_selection_indices(Path(args.selection_random), "Random")
    deminf_indices = load_selection_indices(Path(args.selection_deminf), "DemInf")
    ours_indices = load_selection_indices(Path(args.selection_ours), "Ours")

    # 3. Truncate selections to budget if needed
    random_indices = random_indices[:args.budget]
    deminf_indices = deminf_indices[:args.budget]
    ours_indices = ours_indices[:args.budget]

    # 4. Validate data consistency
    validate_data(candidate_indices, candidate_features,
                  random_indices, deminf_indices, ours_indices, args.budget)

    # 5. Generate Figure 1: Configuration Space Comparison (raw features)
    print("\n[Figure 1] Configuration Space Comparison (raw features)...")
    visualize_configuration_space(
        candidate_indices, candidate_features,
        random_indices, deminf_indices, ours_indices,
        args.budget, output_dir
    )

    # 6. Generate Figure 2: Distribution Metrics Comparison
    print("\n[Figure 2] Distribution Metrics Comparison...")
    if args.metrics_json and Path(args.metrics_json).exists():
        metrics = load_metrics(Path(args.metrics_json))
    else:
        metrics = {
            "random": {
                "coverage_ratio": 0.95,
                "entropy": 3.4499,
                "pairwise_configuration_distance": 0.1179,
                "js_divergence": 0.213,
                "emd_distance": 1.5089,
            },
            "deminf": {
                "coverage_ratio": 0.75,
                "entropy": 3.2011,
                "pairwise_configuration_distance": 0.106,
                "js_divergence": 0.3032,
                "emd_distance": 2.0418,
            },
            "ours": {
                "coverage_ratio": 1.0,
                "entropy": 3.643,
                "pairwise_configuration_distance": 0.1224,
                "js_divergence": 0.0553,
                "emd_distance": 0.1414,
            },
        }
        print("  Using default metrics (no --metrics-json provided)")

    visualize_metrics(metrics, output_dir)

    # 7. Optional: PCA supplementary
    if args.enable_pca:
        print("\n[Supplementary] PCA Visualization...")
        visualize_pca_supplementary(
            candidate_indices, candidate_features,
            random_indices, deminf_indices, ours_indices,
            args.budget, supplementary_dir
        )
    else:
        print("\n  PCA visualization skipped (use --enable-pca to generate)")

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"  Core figures: {output_dir}")
    print(f"  Supplementary: {supplementary_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()