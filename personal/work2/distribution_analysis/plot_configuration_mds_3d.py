#!/usr/bin/env python3
"""
3D MDS visualization for configuration-aware demonstration selection.

Visualization:
6D configuration vector
        |
pairwise configuration distance
        |
3D MDS embedding
        |
Grid-Uniform / Random / DemInf / Ours comparison

"""

import os
import json
import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


# ============================================================
# Load data
# ============================================================

def load_selection(path):
    """Load selected episode indices."""
    with open(path, "r") as f:
        data = json.load(f)

    if "selected_episode_indices" in data:
        return data["selected_episode_indices"]
    elif "indices" in data:
        return data["indices"]
    else:
        raise ValueError(f"Cannot find selected indices in {path}")


def load_rand_vec(path):
    """
    Load configuration vectors.

    Expected format 1:
    {
        "episodes": [
            {"episode_index": 0, "rand_vec": [...]}
        ]
    }

    Expected format 2:
    {
        "0": {"rand_vec": [...]}
    }
    """
    with open(path, "r") as f:
        data = json.load(f)

    configs = []
    episode_ids = []

    if "episodes" in data:
        for episode in data["episodes"]:
            if "rand_vec" not in episode:
                continue
            configs.append(episode["rand_vec"])
            episode_ids.append(episode.get("episode_index", len(episode_ids)))
    else:
        for k, v in data.items():
            if not isinstance(v, dict):
                continue
            if "rand_vec" not in v:
                continue

            configs.append(v["rand_vec"])
            episode_ids.append(int(k))

    configs = np.asarray(configs, dtype=np.float32)

    logging.info(f"Loaded configuration vectors: {configs.shape}")

    return configs, episode_ids


# ============================================================
# MDS
# ============================================================

def compute_mds_embedding(configs):
    logging.info("Computing pairwise configuration distance...")

    distance_matrix = pairwise_distances(
        configs,
        metric="euclidean"
    )

    logging.info("Running 3D MDS...")

    mds = MDS(
        n_components=3,
        dissimilarity="precomputed",
        random_state=42,
        normalized_stress="auto",
        n_init=4,
        max_iter=1000
    )

    embedding = mds.fit_transform(distance_matrix)
    stress = mds.stress_

    logging.info(f"MDS stress: {stress:.4f}")

    return embedding, stress


# ============================================================
# Visualization
# ============================================================

def plot_method(ax, xyz, indices, title, color):
    selected = xyz[indices]

    ax.scatter(
        selected[:, 0],
        selected[:, 1],
        selected[:, 2],
        s=35,
        alpha=0.85,
        c=color,
        edgecolors="none"
    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("MDS-1")
    ax.set_ylabel("MDS-2")
    ax.set_zlabel("MDS-3")


def set_same_limits(axs, xyz):
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    margin = (maxs - mins) * 0.1

    for ax in axs:
        ax.set_xlim(mins[0] - margin[0], maxs[0] + margin[0])
        ax.set_ylim(mins[1] - margin[1], maxs[1] + margin[1])
        ax.set_zlim(mins[2] - margin[2], maxs[2] + margin[2])


def visualize(embedding, selections, output_path):
    fig = plt.figure(figsize=(14, 12))

    methods = [
        ("Grid-Uniform", "grid", "#377eb8"),
        ("Random", "random", "#e41a1c"),
        ("DemInf", "deminf", "#ff7f00"),
        ("Ours", "ours", "#4daf4a")
    ]

    axes = []

    for i, (name, key, color) in enumerate(methods):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        axes.append(ax)

        plot_method(
            ax,
            embedding,
            selections[key],
            f"{name} (Budget={len(selections[key])})",
            color
        )

        ax.view_init(elev=25, azim=45)

    set_same_limits(axes, embedding)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--candidate-pool", required=True)
    parser.add_argument("--selection-grid", required=True)
    parser.add_argument("--selection-random", required=True)
    parser.add_argument("--selection-deminf", required=True)
    parser.add_argument("--selection-ours", required=True)
    parser.add_argument(
        "--output-dir",
        default="personal/work2/distribution_analysis/results"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------
    # Load configuration
    # -----------------------------
    configs, episode_ids = load_rand_vec(args.candidate_pool)

    id_to_index = {eid: i for i, eid in enumerate(episode_ids)}

    # -----------------------------
    # Load selections
    # -----------------------------
    selections_raw = {
        "grid": load_selection(args.selection_grid),
        "random": load_selection(args.selection_random),
        "deminf": load_selection(args.selection_deminf),
        "ours": load_selection(args.selection_ours)
    }

    selections = {}
    for k, v in selections_raw.items():
        selections[k] = [
            id_to_index[x]
            for x in v
            if x in id_to_index
        ]
        logging.info(f"{k}: {len(selections[k])}")

    # -----------------------------
    # MDS
    # -----------------------------
    embedding, stress = compute_mds_embedding(configs)

    # -----------------------------
    # Plot
    # -----------------------------
    output_path = os.path.join(
        args.output_dir,
        "configuration_space_mds_3d.png"
    )

    visualize(embedding, selections, output_path)

    # -----------------------------
    # Save statistics
    # -----------------------------
    stats = {
        "feature_dimension": int(configs.shape[1]),
        "num_candidates": int(len(configs)),
        "mds_stress": float(stress),
        "embedding": "3D MDS using pairwise configuration distance",
        "budget": {k: len(v) for k, v in selections.items()}
    }

    with open(
        os.path.join(args.output_dir, "configuration_space_mds_statistics.json"),
        "w"
    ) as f:
        json.dump(stats, f, indent=4)

    logging.info(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()