"""
Action Embedding Module for V6

Uses V5's action descriptor approach instead of V4's custom action embedding.
Action descriptors are loaded from pre-computed cache (V5 format).

Strict causal access: descriptors can only be loaded for acquired episodes.
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Set, Optional

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_acquired_action_descriptor(
    ep_idx: int,
    action_descriptor_dir: Path,
    acquired_indices: Set[int],
) -> np.ndarray:
    """
    Load action descriptor for an episode ONLY if it has been acquired.

    Uses V5's action descriptor format (pre-computed cache).
    This is a strict causal guard: if ep_idx is not in acquired_indices,
    a RuntimeError is raised to prevent future information leakage.

    Args:
        ep_idx: episode index to load
        action_descriptor_dir: directory containing action descriptor .npy files
        acquired_indices: set of officially acquired episode indices

    Returns:
        numpy array of action descriptor

    Raises:
        RuntimeError: if ep_idx has not been acquired yet (CAUSAL VIOLATION)
    """
    if ep_idx not in acquired_indices:
        raise RuntimeError(
            f"CAUSAL VIOLATION: Attempted to load action descriptor for episode {ep_idx} "
            f"which has NOT been acquired yet. acquired_indices={sorted(acquired_indices)}"
        )

    # Try loading from individual episode file
    descriptor_file = action_descriptor_dir / f"({ep_idx}).npy"
    if descriptor_file.exists():
        data = np.load(str(descriptor_file), allow_pickle=True).item()
        return data["action_descriptor"]

    # Try loading from combined file
    combined_file = action_descriptor_dir / "action_descriptor.npy"
    if combined_file.exists():
        data = np.load(str(combined_file), allow_pickle=True).item()
        descriptors = data["descriptors"]
        indices = data["episode_indices"]
        for i, idx in enumerate(indices):
            if int(idx) == ep_idx:
                return descriptors[i]

    raise FileNotFoundError(
        f"Action descriptor not found for episode {ep_idx} in {action_descriptor_dir}"
    )


def build_action_embedding_for_acquired_episode(
    action_descriptor: np.ndarray,
) -> np.ndarray:
    """
    Build action embedding for an acquired episode using V5's action descriptor.

    In V6, we directly use the pre-computed action descriptor from V5's pipeline.
    This is simpler than V4's custom temporal+statistical embedding.

    Args:
        action_descriptor: pre-computed action descriptor from V5 pipeline

    Returns:
        1-D numpy array (the action embedding)
    """
    return action_descriptor.copy()