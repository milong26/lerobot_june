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
    Build state-action joint embedding.

    If action_embeddings is provided and use_action is True, concatenate
    visual (global+wrist) with action embedding.

    If action embedding is not available, fall back to visual-only with a warning.

    Args:
        embeddings: visual embedding dict.
        action_embeddings: optional action embedding dict.
        use_action: whether to include action embedding.

    Returns:
        Dict[episode_index, state_action_embedding_array]
    """
    state_action = {}

    if use_action and action_embeddings:
        available_action_eps = set(action_embeddings.keys())
        visual_eps = set(embeddings.keys())
        common_eps = visual_eps & available_action_eps

        if len(common_eps) < len(visual_eps):
            missing = visual_eps - available_action_eps
            warnings.warn(
                f"Action embedding missing for {len(missing)} episodes. "
                f"Falling back to visual-only for those episodes."
            )

        for ep_idx in embeddings:
            phi_global = embeddings[ep_idx]["phi_global"]
            phi_wrist = embeddings[ep_idx]["phi_wrist"]
            visual_combined = np.concatenate([phi_global, phi_wrist], axis=0)

            if ep_idx in action_embeddings:
                state_action[ep_idx] = np.concatenate(
                    [visual_combined, action_embeddings[ep_idx]], axis=0
                )
            else:
                state_action[ep_idx] = visual_combined
    else:
        if use_action and not action_embeddings:
            warnings.warn(
                "use_action=True but no action_embeddings provided. "
                "Falling back to visual-only embedding."
            )
        for ep_idx, data in embeddings.items():
            state_action[ep_idx] = np.concatenate(
                [data["phi_global"], data["phi_wrist"]], axis=0
            )

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
        norm = np.linalg.norm(emb)
        if norm < 1e-8:
            normalized[ep_idx] = emb
        else:
            normalized[ep_idx] = emb / norm
    return normalized