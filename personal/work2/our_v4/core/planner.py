"""
Planner Module for V4

Implements the full adaptive collection pipeline with action embeddings:
- Stage 1: coarse uniform coverage (one episode per coarse cell)
- Stage 2: adaptive acquisition based on spatial need + visual disagreement + action disagreement

Action embeddings are computed ONLY AFTER episode acquisition (strict causal access).
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_v4.core.adaptive_grid import (
    AdaptiveCell, build_initial_grid, cell_center, contains,
    split_cell, get_leaf_cells, get_neighbor_cells,
    compute_cell_area, pick_next_target_in_cell,
)
from our_v4.core.pool_adapter import (
    AcquisitionState, load_episode_positions, find_nearest_unselected_episode,
    acquire_episode,
)
from our_v4.core.visual_embedding import (
    load_acquired_visual_embedding, build_combined_embedding,
)
from our_v4.core.action_embedding import (
    load_acquired_action_sequence, build_action_embedding_for_acquired_episode,
)
from our_v4.core.scoring import (
    compute_spatial_need, compute_visual_disagreement, compute_action_disagreement,
    normalize_scores, compute_cell_priority as _compute_cell_priority,
)
from our_v4.config import (
    TOTAL_BUDGET, INITIAL_GRID_X, INITIAL_GRID_Y, INITIAL_BUDGET,
    SPLIT_X, SPLIT_Y, MAX_DEPTH, SPATIAL_WEIGHT, VISUAL_WEIGHT, ACTION_WEIGHT,
    MIN_MAPPING_TOLERANCE, SEED,
)


class V4Planner:
    """Adaptive grid-based episode selection planner with action embeddings."""

    def __init__(
        self,
        dataset_root: str,
        embedding_dir: str,
        grid_x: int = INITIAL_GRID_X,
        grid_y: int = INITIAL_GRID_Y,
        total_budget: int = TOTAL_BUDGET,
        initial_budget: int = INITIAL_BUDGET,
        max_depth: int = MAX_DEPTH,
        spatial_weight: float = SPATIAL_WEIGHT,
        visual_weight: float = VISUAL_WEIGHT,
        action_weight: float = ACTION_WEIGHT,
        seed: int = SEED,
    ):
        self.dataset_root = dataset_root
        self.embedding_dir = Path(embedding_dir)
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.total_budget = total_budget
        self.initial_budget = initial_budget
        self.max_depth = max_depth
        self.spatial_weight = spatial_weight
        self.visual_weight = visual_weight
        self.action_weight = action_weight
        self.seed = seed

        self.rng = np.random.RandomState(seed)

        self.ep_indices, self.ep_positions = load_episode_positions(dataset_root)
        self.all_positions_dict = {
            idx: (float(self.ep_positions[i][0]),
                  float(self.ep_positions[i][1]),
                  float(self.ep_positions[i][2]))
            for i, idx in enumerate(self.ep_indices)
        }

        x_min = float(self.ep_positions[:, 0].min())
        x_max = float(self.ep_positions[:, 0].max())
        y_min = float(self.ep_positions[:, 1].min())
        y_max = float(self.ep_positions[:, 1].max())
        self.workspace_bounds = {
            "x": (x_min, x_max),
            "y": (y_min, y_max),
        }

        initial_cells = build_initial_grid(self.workspace_bounds, grid_x, grid_y)
        self.cells: Dict[str, AdaptiveCell] = {c.cell_id: c for c in initial_cells}

        self.state = AcquisitionState()

        self.initial_stage_indices: List[int] = []
        self.adaptive_stage_indices: List[int] = []

    def _all_selected_positions(self) -> List[Tuple[float, float, float]]:
        """Get all positions of acquired episodes."""
        return list(self.state.acquired_positions.values())

    def _update_cell_membership(self):
        """Update which acquired episodes belong to which cells."""
        for cell in self.cells.values():
            cell.sample_episode_indices = []
            cell.sample_positions = []

        for ep_idx, pos in self.state.acquired_positions.items():
            for cell in get_leaf_cells(self.cells):
                if contains(cell, pos):
                    cell.sample_episode_indices.append(ep_idx)
                    cell.sample_positions.append(pos)
                    break

    def compute_all_cell_priorities(self):
        """
        Compute priorities for all active leaf cells in a unified manner:
        1. Calculate raw spatial, visual, action scores for each leaf cell
        2. Min-max normalize each score type across all cells
        3. Compute final_priority = spatial_w * norm_spatial + visual_w * norm_visual + action_w * norm_action
        4. Store final priority and normalized components on each cell
        """
        self._update_cell_membership()

        leaf_cells = get_leaf_cells(self.cells)
        if not leaf_cells:
            return

        raw_spatial_scores = []
        raw_visual_scores = []
        raw_action_scores = []

        for cell in leaf_cells:
            raw_s, raw_v, raw_a, _ = _compute_cell_priority(
                cell, self.cells, self.state.acquired_positions,
                self.state.visual_embeddings,
                self.state.action_embeddings,
                spatial_weight=self.spatial_weight,
                visual_weight=self.visual_weight,
                action_weight=self.action_weight,
            )
            raw_spatial_scores.append(raw_s)
            raw_visual_scores.append(raw_v)
            raw_action_scores.append(raw_a)

        norm_spatial = normalize_scores(raw_spatial_scores)
        norm_visual = normalize_scores(raw_visual_scores)
        norm_action = normalize_scores(raw_action_scores)

        for i, cell in enumerate(leaf_cells):
            cell.priority = (
                self.spatial_weight * norm_spatial[i] +
                self.visual_weight * norm_visual[i] +
                self.action_weight * norm_action[i]
            )
            cell._norm_spatial = norm_spatial[i]
            cell._norm_visual = norm_visual[i]
            cell._norm_action = norm_action[i]
            cell._raw_spatial = raw_spatial_scores[i]
            cell._raw_visual = raw_visual_scores[i]
            cell._raw_action = raw_action_scores[i]

    def initialize_uniform_collection(self):
        """
        Stage 1: Coarse uniform coverage.
        Iterate over all coarse cells, use cell_center as target,
        map to nearest unused episode, and acquire.
        NO random selection.
        """
        print(f"\n{'='*60}")
        print(f"Stage 1: Coarse Uniform Collection ({self.initial_budget} episodes)")
        print(f"Grid: {self.grid_x} x {self.grid_y} = {self.grid_x * self.grid_y} cells")
        print(f"{'='*60}")

        coarse_cells = [c for c in self.cells.values() if c.depth == 0]

        for cell in coarse_cells:
            target_pos = cell_center(cell)

            ep_idx, actual_pos, mapping_dist, fallback = acquire_episode(
                target_pos,
                self.ep_indices,
                self.ep_positions,
                self.state,
                mapping_tolerance=MIN_MAPPING_TOLERANCE,
            )

            log_entry = {
                "step": self.state.step + 1,
                "stage": "initial",
                "episode_index": ep_idx,
                "selected_cell_id": cell.cell_id,
                "cell_depth": cell.depth,
                "target_init_pos": list(target_pos),
                "actual_init_pos": list(actual_pos),
                "mapping_distance": mapping_dist,
                "mapping_fallback": fallback,
                "raw_spatial_score": 0.0,
                "raw_visual_score": 0.0,
                "raw_action_score": 0.0,
                "normalized_spatial_score": 0.0,
                "normalized_visual_score": 0.0,
                "normalized_action_score": 0.0,
                "spatial_score": 0.0,
                "visual_score": 0.0,
                "action_score": 0.0,
                "final_priority": 0.0,
                "split": False,
                "split_parent_id": None,
                "created_child_ids": [],
                "n_acquired": self.state.n_acquired() + 1,
            }

            self.state.acquire(ep_idx, actual_pos, log_entry)
            self.initial_stage_indices.append(ep_idx)

            emb = load_acquired_visual_embedding(
                ep_idx, self.embedding_dir, self.state.acquired_indices
            )
            self.state.visual_embeddings[ep_idx] = emb

            action_seq = load_acquired_action_sequence(
                self.dataset_root, ep_idx, self.state.acquired_indices
            )
            action_emb = build_action_embedding_for_acquired_episode(action_seq)
            self.state.action_embeddings[ep_idx] = action_emb

            self._update_cell_membership()

            print(f"  Step {self.state.step}: cell={cell.cell_id}, "
                  f"ep={ep_idx}, dist={mapping_dist:.4f}, fallback={fallback}")

        print(f"\nStage 1 complete: {self.state.n_acquired()} episodes acquired")
        print(f"  Initial episodes: {sorted(self.initial_stage_indices)}")

    def select_highest_priority_cell(self) -> Optional[AdaptiveCell]:
        """Select the active leaf cell with highest priority."""
        leaf_cells = get_leaf_cells(self.cells)
        if not leaf_cells:
            return None

        best_cell = max(leaf_cells, key=lambda c: c.priority)
        return best_cell

    def maybe_split_cell(self, cell: AdaptiveCell) -> Tuple[bool, Optional[str], List[str]]:
        """
        Split a cell if it is high-value and depth < MAX_DEPTH.
        Split decision is based on pre-acquisition information (n_samples >= 2).
        After split, reassign historical samples to child leaf cells.
        """
        if cell.depth >= self.max_depth:
            return False, None, []

        if cell.n_samples >= 2:
            parent_id = cell.cell_id
            children = split_cell(cell, SPLIT_X, SPLIT_Y)
            child_ids = [c.cell_id for c in children]
            for child in children:
                self.cells[child.cell_id] = child

            self._update_cell_membership()

            return True, parent_id, child_ids

        return False, None, []

    def collect_one_step(self) -> Dict:
        """
        Execute one adaptive acquisition step:
        1. Update cell statistics (compute priorities)
        2. Select highest priority cell
        3. Generate new target init_pos
        4. Map to nearest unused episode
        5. Acquire episode
        6. Load visual embedding
        7. Load action sequence and build action embedding
        8. Maybe split cell
        """
        self.compute_all_cell_priorities()

        best_cell = self.select_highest_priority_cell()
        if best_cell is None:
            raise RuntimeError("No available cells for collection")

        cell_sample_positions = list(best_cell.sample_positions)
        if not cell_sample_positions:
            cell_sample_positions = [cell_center(best_cell)]

        target_pos = pick_next_target_in_cell(
            best_cell, cell_sample_positions, SPLIT_X, SPLIT_Y, rng=self.rng
        )

        ep_idx, actual_pos, mapping_dist, fallback = acquire_episode(
            target_pos,
            self.ep_indices,
            self.ep_positions,
            self.state,
            mapping_tolerance=MIN_MAPPING_TOLERANCE,
        )

        raw_spatial, raw_visual, raw_action, _ = _compute_cell_priority(
            best_cell, self.cells, self.state.acquired_positions,
            self.state.visual_embeddings,
            self.state.action_embeddings,
            spatial_weight=self.spatial_weight,
            visual_weight=self.visual_weight,
            action_weight=self.action_weight,
        )

        did_split, split_parent_id, created_child_ids = self.maybe_split_cell(best_cell)

        final_priority = best_cell.priority

        log_entry = {
            "step": self.state.step + 1,
            "stage": "adaptive",
            "episode_index": ep_idx,
            "selected_cell_id": best_cell.cell_id,
            "cell_depth": best_cell.depth,
            "target_init_pos": list(target_pos),
            "actual_init_pos": list(actual_pos),
            "mapping_distance": mapping_dist,
            "mapping_fallback": fallback,
            "raw_spatial_score": float(raw_spatial),
            "raw_visual_score": float(raw_visual),
            "raw_action_score": float(raw_action),
            "normalized_spatial_score": float(getattr(best_cell, '_norm_spatial', 0.0)),
            "normalized_visual_score": float(getattr(best_cell, '_norm_visual', 0.0)),
            "normalized_action_score": float(getattr(best_cell, '_norm_action', 0.0)),
            "spatial_score": float(getattr(best_cell, '_norm_spatial', 0.0)),
            "visual_score": float(getattr(best_cell, '_norm_visual', 0.0)),
            "action_score": float(getattr(best_cell, '_norm_action', 0.0)),
            "final_priority": float(final_priority),
            "split": did_split,
            "split_parent_id": split_parent_id,
            "created_child_ids": created_child_ids,
            "n_acquired": self.state.n_acquired() + 1,
        }

        self.state.acquire(ep_idx, actual_pos, log_entry)
        self.adaptive_stage_indices.append(ep_idx)

        emb = load_acquired_visual_embedding(
            ep_idx, self.embedding_dir, self.state.acquired_indices
        )
        self.state.visual_embeddings[ep_idx] = emb

        action_seq = load_acquired_action_sequence(
            self.dataset_root, ep_idx, self.state.acquired_indices
        )
        action_emb = build_action_embedding_for_acquired_episode(action_seq)
        self.state.action_embeddings[ep_idx] = action_emb

        self._update_cell_membership()

        print(f"  Step {self.state.step}: cell={best_cell.cell_id}(d={best_cell.depth}), "
              f"ep={ep_idx}, dist={mapping_dist:.4f}, fallback={fallback}, "
              f"raw_spatial={raw_spatial:.4f}, raw_visual={raw_visual:.4f}, "
              f"raw_action={raw_action:.4f}, priority={final_priority:.4f}, split={did_split}")

        return log_entry

    def run_adaptive_collection(self, total_budget: int = TOTAL_BUDGET) -> Dict:
        """
        Run the full adaptive collection pipeline.
        Stage 1: coarse uniform (initial_budget episodes)
        Stage 2: adaptive acquisition until total_budget reached.
        """
        start_time = time.time()

        self.initialize_uniform_collection()

        print(f"\n{'='*60}")
        print(f"Stage 2: Adaptive Collection")
        print(f"Target: {total_budget} total episodes")
        print(f"Remaining: {total_budget - self.state.n_acquired()} episodes")
        print(f"{'='*60}")

        while self.state.n_acquired() < total_budget:
            self.collect_one_step()

        elapsed = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"Collection complete!")
        print(f"Total episodes: {self.state.n_acquired()}")
        print(f"Initial stage: {len(self.initial_stage_indices)}")
        print(f"Adaptive stage: {len(self.adaptive_stage_indices)}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"{'='*60}")

        return self._build_result()

    def _build_result(self) -> Dict:
        """Build the final result dictionary."""
        mapping_fallbacks = [h.get("mapping_fallback", False) for h in self.state.history]

        mapping_distances = [h["mapping_distance"] for h in self.state.history]
        fallback_count = sum(1 for f in mapping_fallbacks if f)
        fallback_ratio = fallback_count / len(mapping_fallbacks) if mapping_fallbacks else 0.0
        mean_mapping_dist = float(np.mean(mapping_distances)) if mapping_distances else 0.0
        max_mapping_dist = float(np.max(mapping_distances)) if mapping_distances else 0.0

        selected_episode_indices = [h["episode_index"] for h in self.state.history]
        initial_stage_indices = [h["episode_index"] for h in self.state.history if h["stage"] == "initial"]
        adaptive_stage_indices = [h["episode_index"] for h in self.state.history if h["stage"] == "adaptive"]

        return {
            "selected_episode_indices": selected_episode_indices,
            "target_init_positions": [
                list(h["target_init_pos"]) for h in self.state.history
            ],
            "actual_init_positions": [
                list(h["actual_init_pos"]) for h in self.state.history
            ],
            "mapping_distances": mapping_distances,
            "mapping_fallbacks": mapping_fallbacks,
            "initial_stage_indices": initial_stage_indices,
            "adaptive_stage_indices": adaptive_stage_indices,
            "selection_method": "dynamicgrid_v4_action",
            "parameters": {
                "total_budget": self.total_budget,
                "initial_grid_x": self.grid_x,
                "initial_grid_y": self.grid_y,
                "initial_budget": self.initial_budget,
                "max_depth": self.max_depth,
                "spatial_weight": self.spatial_weight,
                "visual_weight": self.visual_weight,
                "action_weight": self.action_weight,
                "seed": self.seed,
            },
            "acquisition_log": self.state.history,
            "mapping_stats": {
                "fallback_count": fallback_count,
                "fallback_ratio": fallback_ratio,
                "mean_mapping_distance": mean_mapping_dist,
                "max_mapping_distance": max_mapping_dist,
            },
        }

    def validate_causal_access(self) -> bool:
        """
        Validate that no unacquired episode embeddings were accessed.
        Checks both visual_embeddings and action_embeddings.
        """
        for ep_idx in self.state.visual_embeddings:
            if ep_idx not in self.state.acquired_indices:
                raise RuntimeError(
                    f"CAUSAL VIOLATION: Episode {ep_idx} has visual embedding "
                    f"but was never acquired!"
                )
        for ep_idx in self.state.action_embeddings:
            if ep_idx not in self.state.acquired_indices:
                raise RuntimeError(
                    f"CAUSAL VIOLATION: Episode {ep_idx} has action embedding "
                    f"but was never acquired!"
                )
        return True