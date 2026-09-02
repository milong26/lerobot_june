"""
Pool Adapter: Bridge between algorithm targets and episode pool

Reads only episode_initial_states.json for episode_index and obj_init_pos.
Maps algorithm-generated target init_pos to nearest unused episode.
Simulates acquisition without reading any episode data before acquisition.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional


def load_episode_positions(dataset_root: str) -> Tuple[List[int], np.ndarray]:
    """
    Load episode_index and obj_init_pos from episode_initial_states.json.
    Does NOT read any image, action, trajectory, or success data.
    """
    json_path = Path(dataset_root) / "episode_initial_states.json"
    if not json_path.exists():
        raise FileNotFoundError(f"episode_initial_states.json not found at {json_path}")

    with open(json_path, "r") as f:
        meta = json.load(f)

    episodes = meta["episodes"]
    indices = [ep["episode_index"] for ep in episodes]
    positions = np.array([ep["obj_init_pos"] for ep in episodes])

    return indices, positions


def find_nearest_unselected_episode(
    target_pos: Tuple[float, float, float],
    all_indices: List[int],
    all_positions: np.ndarray,
    used_indices: Set[int],
) -> Tuple[int, Tuple[float, float, float], float]:
    """
    Find the nearest unused episode to the target init_pos.

    Returns:
        (episode_index, actual_obj_init_pos, mapping_distance)
    """
    target_2d = np.array([target_pos[0], target_pos[1]])

    best_idx = None
    best_dist = float("inf")
    best_actual_pos = None

    for i, ep_idx in enumerate(all_indices):
        if ep_idx in used_indices:
            continue
        ep_pos = all_positions[i]
        ep_2d = np.array([ep_pos[0], ep_pos[1]])
        dist = np.linalg.norm(target_2d - ep_2d)
        if dist < best_dist:
            best_dist = dist
            best_idx = ep_idx
            best_actual_pos = (float(ep_pos[0]), float(ep_pos[1]), float(ep_pos[2]))

    if best_idx is None:
        raise RuntimeError("No unused episodes available in pool")

    return best_idx, best_actual_pos, float(best_dist)


def acquire_episode(
    target_pos: Tuple[float, float, float],
    all_indices: List[int],
    all_positions: np.ndarray,
    state: "AcquisitionState",
    mapping_tolerance: float = 0.1,
) -> Tuple[int, Tuple[float, float, float], float, bool]:
    """
    Acquire an episode for the given target position.

    Calls find_nearest_unselected_episode internally and ensures no duplicate episodes.
    Uses mapping_tolerance to determine if the mapping is a good fit.

    Args:
        target_pos: the target (x, y, z) position
        all_indices: all episode indices
        all_positions: all episode positions array
        state: current acquisition state (for used_indices)
        mapping_tolerance: maximum distance for a "good" mapping (meters)

    Returns:
        (episode_index, actual_pos, mapping_distance, fallback)
        - fallback=False if mapping_distance <= mapping_tolerance (good fit)
        - fallback=True if mapping_distance > mapping_tolerance (still selected but imperfect)
    """
    ep_idx, actual_pos, mapping_dist = find_nearest_unselected_episode(
        target_pos,
        all_indices,
        all_positions,
        state.acquired_indices,
    )

    fallback = mapping_dist > mapping_tolerance

    return ep_idx, actual_pos, mapping_dist, fallback


class AcquisitionState:
    """Tracks the state of the acquisition process."""

    def __init__(self):
        self.acquired_indices: Set[int] = set()
        self.acquired_positions: Dict[int, Tuple[float, float, float]] = {}
        self.visual_embeddings: Dict[int, Dict] = {}  # ep_idx -> {"phi_global", "phi_wrist"}
        self.step: int = 0
        self.history: List[Dict] = []

    def acquire(
        self,
        episode_index: int,
        actual_pos: Tuple[float, float, float],
        log_entry: Dict,
    ):
        """Record a newly acquired episode."""
        self.acquired_indices.add(episode_index)
        self.acquired_positions[episode_index] = actual_pos
        self.step += 1
        self.history.append(log_entry)

    def is_acquired(self, episode_index: int) -> bool:
        return episode_index in self.acquired_indices

    def n_acquired(self) -> int:
        return len(self.acquired_indices)