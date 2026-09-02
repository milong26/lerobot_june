"""
DemInf Subset Selection

Select top-K episodes by DemInf score and save subset JSON compatible
with the existing training pipeline.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from deminf.config import DemInfConfig
from deminf.utils import atomic_save_json, ensure_dir

logger = logging.getLogger("deminf")


def select_top_episodes(
    score_table: pd.DataFrame,
    target_episodes: int,
    tie_break_seed: int = 42,
) -> List[int]:
    """
    Select top-K episodes by DemInf score (descending).

    Ties are broken by episode index (ascending) for stable sorting.

    Args:
        score_table: DataFrame with 'deminf_score' and 'episode_idx' columns.
        target_episodes: Number of episodes to select.
        tie_break_seed: Seed for tie-breaking (not used if stable sort).

    Returns:
        List of selected episode indices (sorted ascending for compatibility).
    """
    if target_episodes > len(score_table):
        logger.warning(
            f"target_episodes={target_episodes} > available episodes={len(score_table)}, "
            f"selecting all available episodes"
        )
        target_episodes = len(score_table)

    # Sort by score descending, then by episode_idx ascending for stable tie-breaking
    sorted_df = score_table.sort_values(
        by=["deminf_score", "episode_idx"],
        ascending=[False, True],
    ).reset_index(drop=True)

    selected = sorted_df.head(target_episodes)["episode_idx"].tolist()

    # Sort ascending for compatibility with existing subset loaders
    selected = sorted(selected)

    logger.info(f"Selected {len(selected)} episodes: {selected[:10]}...{selected[-5:]}")

    return selected


def save_subset_json(
    selected_indices: List[int],
    score_table: pd.DataFrame,
    config: DemInfConfig,
    output_path: str | Path,
    relative_action: bool = False,
) -> Dict[str, Any]:
    """
    Save subset JSON compatible with existing training pipeline.

    Format matches random/uniform/SubZeroCore subset files:
    {
        "selected_episode_indices": [...],
        "num_episodes": K,
        "selection_method": "deminf",
        "parameters": {...}
    }

    Args:
        selected_indices: List of selected episode indices (ascending sorted).
        score_table: Full episode score DataFrame.
        config: DemInfConfig.
        output_path: Output JSON file path.
        relative_action: Whether actions are relative/delta control.

    Returns:
        Subset dictionary.
    """
    ensure_dir(Path(output_path).parent)

    subset_data = {
        "selected_episode_indices": selected_indices,
        "num_episodes": len(selected_indices),
        "selection_method": "deminf",
        "parameters": {
            "target_episodes": config.target_episodes,
            "seed": config.seed,
            "ks": list(config.ks),
            "state_latent_dim": config.state_latent_dim,
            "action_latent_dim": config.action_latent_dim,
            "vae_beta_state": config.vae_beta_state,
            "vae_beta_action": config.vae_beta_action,
            "vae_epochs": config.vae_epochs,
            "hidden_dims": config.hidden_dims,
            "relative_action": relative_action,
            "dataset_path": config.dataset_path,
            "score_aggregation": "mean",
            "ksg_mode": config.ksg_mode,
            "ksg_backend": config.ksg_backend,
            "representation": config.representation,
        },
    }

    atomic_save_json(subset_data, output_path)
    logger.info(f"Saved subset JSON to {output_path}")

    return subset_data


def save_score_rankings(
    score_table: pd.DataFrame,
    selected_indices: List[int],
    output_path: str | Path,
) -> None:
    """
    Save detailed score rankings with selection status.

    Args:
        score_table: Full episode score DataFrame.
        selected_indices: Selected episode indices.
        output_path: Output CSV file path.
    """
    df = score_table.copy()
    df["selected"] = df["episode_idx"].isin(selected_indices)

    ensure_dir(Path(output_path).parent)
    df.to_csv(str(output_path), index=False)
    logger.info(f"Saved score rankings to {output_path}")