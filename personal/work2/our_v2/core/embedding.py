"""
V2 Embedding Construction Module

Implements combined state-action embedding building for Dynamic Anchor v2.
Supports global+wrist visual embeddings and optional action trajectory embeddings.
"""

import sys
import warnings
import numpy as np
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from our_v2.config import VISUAL_WEIGHT, ACTION_WEIGHT, VISUAL_NORMALIZE, ACTION_NORMALIZE


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2 normalize a 1-D vector."""
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return vec
    return vec / norm


def load_episode_embeddings(embedding_dir: Path) -> Dict[int, Dict]:
    """
    Load global and wrist embeddings for all episodes.

    Args:
        embedding_dir: directory containing .npy embedding cache files.

    Returns:
        Dict[episode_index, {"phi_global": np.ndarray, "phi_wrist": np.ndarray}]
    """
    embeddings = {}

    for f in embedding_dir.glob("*.npy"):
        try:
            data = np.load(f, allow_pickle=True).item()
            ep_idx = data.get("episode_index")
            if ep_idx is not None:
                embeddings[ep_idx] = {
                    "phi_global": data["phi_global"],
                    "phi_wrist": data["phi_wrist"],
                }
        except Exception as e:
            print(f"  Skip invalid file: {f.name} ({e})")

    print(f"Loaded {len(embeddings)} episode embeddings")
    return embeddings


def build_combined_embedding(
    embeddings: Dict[int, Dict],
) -> Dict[int, np.ndarray]:
    """
    Concatenate global and wrist embeddings for each episode.

    Args:
        embeddings: dict from load_episode_embeddings.

    Returns:
        Dict[episode_index, combined_embedding_array]
    """
    combined = {}
    for ep_idx, data in embeddings.items():
        combined[ep_idx] = np.concatenate(
            [data["phi_global"], data["phi_wrist"]], axis=0
        )
    return combined


def build_state_action_embedding(
    embeddings: Dict[int, Dict],
    action_embeddings: Optional[Dict[int, np.ndarray]] = None,
    use_action: bool = False,
) -> Dict[int, np.ndarray]:
    """
    Build state-action joint embedding with proper normalization and weighting.

    Processing pipeline:
    1. Visual: normalize global, normalize wrist, concat, normalize again.
    2. Action: normalize if available.
    3. Fusion: concat(visual * VISUAL_WEIGHT, action * ACTION_WEIGHT).
    4. Fallback: if action not available, use visual-only.

    Args:
        embeddings: visual embedding dict with phi_global and phi_wrist.
        action_embeddings: optional action embedding dict.
        use_action: whether to include action embedding.

    Returns:
        Dict[episode_index, state_action_embedding_array]
    """
    state_action = {}

    # Step 1: Build normalized visual embeddings
    visual_embs = {}
    for ep_idx, data in embeddings.items():
        phi_global = data["phi_global"]
        phi_wrist = data["phi_wrist"]

        if VISUAL_NORMALIZE:
            phi_global = _l2_normalize(phi_global)
            phi_wrist = _l2_normalize(phi_wrist)

        visual_combined = np.concatenate([phi_global, phi_wrist], axis=0)

        if VISUAL_NORMALIZE:
            visual_combined = _l2_normalize(visual_combined)

        visual_embs[ep_idx] = visual_combined

    visual_dim = len(visual_embs[next(iter(visual_embs))])
    print(f"  Visual embedding dimension: {visual_dim}")

    # Step 2: Check action embedding availability
    has_action = use_action and action_embeddings and len(action_embeddings) > 0

    if use_action and not has_action:
        warnings.warn(
            "use_action=True but no action_embeddings provided or empty. "
            "Falling back to visual-only embedding."
        )

    # Step 3: Build state-action embeddings
    for ep_idx in embeddings:
        visual_part = visual_embs[ep_idx] * VISUAL_WEIGHT

        if has_action and ep_idx in action_embeddings:
            action_emb = action_embeddings[ep_idx]
            if ACTION_NORMALIZE:
                action_emb = _l2_normalize(action_emb)
            action_part = action_emb * ACTION_WEIGHT

            combined = np.concatenate([visual_part, action_part], axis=0)
            state_action[ep_idx] = combined
        else:
            state_action[ep_idx] = visual_part

    # Step 4: Log dimensions
    sample_emb = state_action[next(iter(state_action))]
    final_dim = len(sample_emb)

    if has_action:
        action_dim = len(action_embeddings[next(iter(action_embeddings))])
        print(f"  Action embedding dimension: {action_dim}")
    else:
        print(f"  Action embedding dimension: N/A (visual-only)")

    print(f"  Final state-action embedding dimension: {final_dim}")

    return state_action


def normalize_embedding(
    embeddings: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """
    L2-normalize each episode embedding for stable distance computation.

    Args:
        embeddings: dict of episode embeddings.

    Returns:
        Dict[episode_index, normalized_embedding_array]
    """
    normalized = {}
    for ep_idx, emb in embeddings.items():
        normalized[ep_idx] = _l2_normalize(emb)
    return normalized