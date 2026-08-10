"""
Step 3: Selection strategies for Ours.
Multiple candidate strategies that can be compared experimentally.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

SUBSETS_DIR = Path(__file__).parent / "subsets"


def compute_pairwise_distances(X):
    """Compute pairwise Euclidean distances."""
    return cdist(X, X, metric="euclidean")


def compute_pairwise_cosine(X):
    """Compute pairwise cosine distances."""
    from sklearn.metrics.pairwise import cosine_distances
    return cosine_distances(X)


class B0Initializer:
    """Generate initial B0 seed set using various strategies."""

    @staticmethod
    def uniform_grid(obj_positions, grid_size=3):
        """Uniform grid over workspace."""
        x_min, x_max = obj_positions[:, 0].min(), obj_positions[:, 0].max()
        y_min, y_max = obj_positions[:, 1].min(), obj_positions[:, 1].max()

        x_edges = np.linspace(x_min, x_max, grid_size + 1)
        y_edges = np.linspace(y_min, y_max, grid_size + 1)

        selected = []
        for i in range(grid_size):
            for j in range(grid_size):
                mask = (
                    (obj_positions[:, 0] >= x_edges[i]) & (obj_positions[:, 0] < x_edges[i + 1]) &
                    (obj_positions[:, 1] >= y_edges[j]) & (obj_positions[:, 1] < y_edges[j + 1])
                )
                candidates = np.where(mask)[0]
                if len(candidates) > 0:
                    # Pick closest to cell center
                    cx = (x_edges[i] + x_edges[i + 1]) / 2
                    cy = (y_edges[j] + y_edges[j + 1]) / 2
                    dists = np.sqrt((obj_positions[candidates, 0] - cx) ** 2 +
                                    (obj_positions[candidates, 1] - cy) ** 2)
                    selected.append(candidates[dists.argmin()])

        return np.array(selected)

    @staticmethod
    def quantile_grid(obj_positions, grid_size=3):
        """Quantile-based grid over workspace."""
        x_quantiles = np.linspace(0, 1, grid_size + 1)
        y_quantiles = np.linspace(0, 1, grid_size + 1)

        x_edges = np.quantile(obj_positions[:, 0], x_quantiles)
        y_edges = np.quantile(obj_positions[:, 1], y_quantiles)

        selected = []
        for i in range(grid_size):
            for j in range(grid_size):
                mask = (
                    (obj_positions[:, 0] >= x_edges[i]) & (obj_positions[:, 0] <= x_edges[i + 1]) &
                    (obj_positions[:, 1] >= y_edges[j]) & (obj_positions[:, 1] <= y_edges[j + 1])
                )
                candidates = np.where(mask)[0]
                if len(candidates) > 0:
                    cx = (x_edges[i] + x_edges[i + 1]) / 2
                    cy = (y_edges[j] + y_edges[j + 1]) / 2
                    dists = np.sqrt((obj_positions[candidates, 0] - cx) ** 2 +
                                    (obj_positions[candidates, 1] - cy) ** 2)
                    selected.append(candidates[dists.argmin()])

        return np.array(list(set(selected)))

    @staticmethod
    def random(obj_positions, n, rng=None):
        """Random selection."""
        if rng is None:
            rng = np.random.RandomState(42)
        indices = rng.choice(len(obj_positions), size=min(n, len(obj_positions)), replace=False)
        return np.sort(indices)

    @staticmethod
    def fps(obj_positions, n, rng=None):
        """Farthest Point Sampling in physical space."""
        if rng is None:
            rng = np.random.RandomState(42)

        n_total = len(obj_positions)
        selected = [rng.randint(n_total)]
        min_dists = np.full(n_total, np.inf)

        for _ in range(1, n):
            last = obj_positions[selected[-1], :2]  # x, y only
            dists = np.sqrt(((obj_positions[:, :2] - last) ** 2).sum(axis=1))
            min_dists = np.minimum(min_dists, dists)
            next_idx = np.argmax(min_dists)
            selected.append(next_idx)

        return np.array(selected)


class OursSelector:
    """Coverage-based selection strategies."""

    def __init__(self, embeddings, obj_positions, episode_indices, strategy="sic"):
        """
        Args:
            embeddings: (N, D) array - PCA or raw embeddings
            obj_positions: (N, 3) array - object positions
            episode_indices: (N,) array - episode indices
            strategy: selection strategy name
        """
        self.embeddings = embeddings
        self.obj_positions = obj_positions
        self.episode_indices = episode_indices
        self.strategy = strategy
        self.n_total = len(embeddings)
        self.selected = []
        self.selection_log = []

    def select(self, target_size, b0_indices=None):
        """
        Select target_size episodes.
        If b0_indices provided, start from them.
        """
        if b0_indices is not None:
            self.selected = list(b0_indices)
            print(f"Starting with B0: {len(b0_indices)} episodes")

        remaining = target_size - len(self.selected)
        if remaining <= 0:
            return np.array(self.selected)

        print(f"Selecting {remaining} more episodes using {self.strategy}...")

        if self.strategy == "sic":
            self._select_sic(remaining)
        elif self.strategy == "coverage_greedy":
            self._select_coverage_greedy(remaining)
        elif self.strategy == "fps_embedding":
            self._select_fps_embedding(remaining)
        elif self.strategy == "undercovered":
            self._select_undercovered(remaining)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        return np.array(self.selected)

    def _select_sic(self, n_select):
        """SIC-like saturated coverage selection."""
        print(f"  DEBUG: embeddings shape = {self.embeddings.shape}")
        print(f"  DEBUG: n_total = {self.n_total}")
        print(f"  DEBUG: selected (B0) = {self.selected}")
        
        dist_matrix = compute_pairwise_distances(self.embeddings)
        print(f"  DEBUG: dist_matrix shape = {dist_matrix.shape}")

        for _ in range(n_select):
            if len(self.selected) >= self.n_total:
                break

            available = np.array([i for i in range(self.n_total) if i not in self.selected])
            if len(available) == 0:
                break

            # Compute marginal gain for each candidate
            scores = np.zeros(len(available))
            for j, cand in enumerate(available):
                # Distance to nearest selected
                dists_to_selected = dist_matrix[cand, self.selected]
                min_dist = dists_to_selected.min()

                # SIC-style: saturate at some threshold
                # Use median pairwise distance as d_bar
                if len(self.selected) > 1:
                    d_bar = np.median(dist_matrix[np.ix_(self.selected, self.selected)])
                else:
                    d_bar = np.median(dist_matrix)

                # Marginal gain: higher if far from selected
                scores[j] = min(min_dist / d_bar, 1.0)

            best_idx = np.argmax(scores)
            chosen = available[best_idx]
            self.selected.append(int(chosen))

            self.selection_log.append({
                "step": len(self.selected),
                "chosen": int(chosen),
                "score": float(scores[best_idx]),
                "min_dist_to_selected": float(dist_matrix[chosen, self.selected[:-1]].min()) if len(self.selected) > 1 else 0,
            })

    def _select_coverage_greedy(self, n_select):
        """Greedy coverage: pick point farthest from current selection."""
        dist_matrix = compute_pairwise_distances(self.embeddings)

        for _ in range(n_select):
            if len(self.selected) >= self.n_total:
                break

            available = np.array([i for i in range(self.n_total) if i not in self.selected])
            if len(available) == 0:
                break

            # For each candidate, compute min distance to selected
            min_dists = np.array([dist_matrix[cand, self.selected].min() for cand in available])
            best_idx = np.argmax(min_dists)
            chosen = available[best_idx]
            self.selected.append(int(chosen))

            self.selection_log.append({
                "step": len(self.selected),
                "chosen": int(chosen),
                "min_dist": float(min_dists[best_idx]),
            })

    def _select_fps_embedding(self, n_select):
        """FPS in embedding space."""
        dist_matrix = compute_pairwise_distances(self.embeddings)
        min_dists = np.full(self.n_total, np.inf)

        for _ in range(n_select):
            if len(self.selected) >= self.n_total:
                break

            last = self.selected[-1]
            dists = dist_matrix[:, last]
            min_dists = np.minimum(min_dists, dists)

            # Mask out already selected
            for s in self.selected:
                min_dists[s] = -1

            chosen = int(np.argmax(min_dists))
            self.selected.append(chosen)

            self.selection_log.append({
                "step": len(self.selected),
                "chosen": chosen,
                "min_dist": float(min_dists[chosen]),
            })

    def _select_undercovered(self, n_select):
        """Identify under-covered regions and select from them."""
        dist_matrix = compute_pairwise_distances(self.embeddings)

        # Compute density of all points
        k = min(10, self.n_total - 1)
        knn_dists = np.sort(dist_matrix, axis=1)[:, 1:k + 1]
        density = 1.0 / (knn_dists.mean(axis=1) + 1e-8)

        for _ in range(n_select):
            if len(self.selected) >= self.n_total:
                break

            available = np.array([i for i in range(self.n_total) if i not in self.selected])
            if len(available) == 0:
                break

            # Score: low density (sparse area) + far from selected
            density_scores = density[available]
            # Invert: higher score for lower density
            density_scores = 1.0 / (density_scores + 1e-8)

            if len(self.selected) > 0:
                dists_to_selected = dist_matrix[np.ix_(available, self.selected)]
                min_dists = dists_to_selected.min(axis=1)
                # Combined score
                scores = density_scores * min_dists
            else:
                scores = density_scores

            best_idx = np.argmax(scores)
            chosen = available[best_idx]
            self.selected.append(int(chosen))

            self.selection_log.append({
                "step": len(self.selected),
                "chosen": int(chosen),
                "score": float(scores[best_idx]),
                "density": float(density[chosen]),
            })

    def save_selection(self, method_name, config=None):
        """Save selection results."""
        SUBSETS_DIR.mkdir(parents=True, exist_ok=True)

        result = {
            "method": method_name,
            "strategy": self.strategy,
            "subset_size": len(self.selected),
            "selected_episode_indices": [int(x) for x in self.selected],
            "selection_log": self.selection_log,
            "config": config or {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        save_path = SUBSETS_DIR / f"{method_name}_subset.json"
        with open(save_path, "w") as f:
            json.dump(result, f, indent=2)

        print(f"Saved selection to {save_path}")
        return result