"""
DemInf Episode Scoring

From dataset to trajectory scores:
1. Encode all timesteps through trained VAEs
2. Compute KSG local MI scores
3. Aggregate per-episode scores
4. Save score table as CSV
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from deminf.config import DemInfConfig
from deminf.ksg import ksg_local_scores
from deminf.models import BetaVAE
from deminf.utils import ensure_dir

logger = logging.getLogger("deminf")


def encode_all_timesteps(
    state_model: BetaVAE,
    action_model: BetaVAE,
    states: np.ndarray,
    actions: np.ndarray,
    config: DemInfConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode all timesteps through trained VAEs using posterior mean.

    Args:
        state_model: Trained state VAE in eval mode.
        action_model: Trained action VAE in eval mode.
        states: State vectors [N, D_s].
        actions: Action vectors [N, D_a].
        config: DemInfConfig.

    Returns:
        z_s: State latents [N, state_latent_dim].
        z_a: Action latents [N, action_latent_dim].
    """
    device = state_model.encoder.shared[0].weight.device

    state_tensor = torch.from_numpy(states).float().to(device)
    action_tensor = torch.from_numpy(actions).float().to(device)

    # Encode in batches
    z_s_list = []
    z_a_list = []

    state_model.eval()
    action_model.eval()

    with torch.no_grad():
        for start in range(0, len(states), config.score_batch_size):
            end = min(start + config.score_batch_size, len(states))
            s_batch = state_tensor[start:end]
            a_batch = action_tensor[start:end]

            # Use posterior mean (deterministic) for scoring
            z_s_batch = state_model.get_embedding(s_batch, use_mean=True)
            z_a_batch = action_model.get_embedding(a_batch, use_mean=True)

            z_s_list.append(z_s_batch.cpu().numpy())
            z_a_list.append(z_a_batch.cpu().numpy())

    z_s = np.concatenate(z_s_list, axis=0)
    z_a = np.concatenate(z_a_list, axis=0)

    logger.info(f"Encoded {len(z_s)} timesteps: z_s={z_s.shape}, z_a={z_a.shape}")

    return z_s, z_a


def save_latent_cache(
    output_dir: str | Path,
    z_s: np.ndarray,
    z_a: np.ndarray,
    episode_ids: np.ndarray,
    timestep_ids: np.ndarray,
) -> None:
    """
    Save latent embeddings to cache file.

    Args:
        output_dir: Output directory.
        z_s: State latents [N, d_s].
        z_a: Action latents [N, d_a].
        episode_ids: Episode indices [N].
        timestep_ids: Timestep indices [N].
    """
    cache_path = Path(output_dir) / "latents.npz"
    np.savez(
        str(cache_path),
        z_state=z_s,
        z_action=z_a,
        episode_ids=episode_ids,
        timestep_ids=timestep_ids,
    )
    logger.info(f"Saved latent cache to {cache_path}")


def load_latent_cache(
    output_dir: str | Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load latent embeddings from cache file.

    Args:
        output_dir: Output directory.

    Returns:
        z_s, z_a, episode_ids, timestep_ids.
    """
    cache_path = Path(output_dir) / "latents.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Latent cache not found: {cache_path}")

    data = np.load(str(cache_path))
    return data["z_state"], data["z_action"], data["episode_ids"], data["timestep_ids"]


def compute_timestep_information_scores(
    z_s: np.ndarray,
    z_a: np.ndarray,
    config: DemInfConfig,
) -> np.ndarray:
    """
    Compute KSG local information scores for all timesteps.

    Args:
        z_s: State latents [N, d_s].
        z_a: Action latents [N, d_a].
        config: DemInfConfig.

    Returns:
        Per-timestep scores [N].
    """
    logger.info(
        f"Computing KSG scores: N={len(z_s)}, ks={config.ks}, "
        f"mode={config.ksg_mode}, backend={config.ksg_backend}"
    )

    scores = ksg_local_scores(
        z_s, z_a,
        ks=config.ks,
        chunk_size=config.ksg_chunk_size or config.score_batch_size,
        mode=config.ksg_mode,
        backend=config.ksg_backend,
    )

    # Sanity check
    assert np.all(np.isfinite(scores)), "KSG scores contain NaN/Inf"

    logger.info(
        f"KSG scores: min={scores.min():.4f}, max={scores.max():.4f}, "
        f"mean={scores.mean():.4f}, std={scores.std():.4f}"
    )

    return scores


def aggregate_episode_scores(
    local_scores: np.ndarray,
    episode_ids: np.ndarray,
    aggregation: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate per-timestep scores into per-episode scores.

    DemInf default: trajectory score = arithmetic mean of all valid timestep
    information contributions within the episode.

    S(tau_e) = 1/T_e * sum_{t=1}^{T_e} I_hat(z_s_{e,t}; z_a_{e,t})

    Args:
        local_scores: Per-timestep scores [N].
        episode_ids: Episode indices [N].
        aggregation: Aggregation method. Default 'mean'.

    Returns:
        DataFrame with columns: episode_idx, deminf_score, num_steps,
        score_std, score_min, score_max, rank.
    """
    unique_episodes = np.unique(episode_ids)
    records = []

    for ep_idx in unique_episodes:
        mask = episode_ids == ep_idx
        ep_scores = local_scores[mask]

        if aggregation == "mean":
            score = float(np.mean(ep_scores))
        elif aggregation == "sum":
            score = float(np.sum(ep_scores))
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        records.append({
            "episode_idx": int(ep_idx),
            "deminf_score": score,
            "num_steps": int(len(ep_scores)),
            "score_std": float(np.std(ep_scores)),
            "score_min": float(np.min(ep_scores)),
            "score_max": float(np.max(ep_scores)),
        })

    df = pd.DataFrame(records)

    # Add rank (descending by score)
    df["rank"] = df["deminf_score"].rank(ascending=False, method="min").astype(int)

    # Sort by rank
    df = df.sort_values("rank").reset_index(drop=True)

    return df


def sanity_check_scores(df: pd.DataFrame, top_k: int = 10) -> None:
    """
    Print sanity check for episode scores.

    Args:
        df: Episode score DataFrame.
        top_k: Number of top/bottom episodes to display.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Episode Score Sanity Check")
    logger.info(f"{'='*60}")

    # Check for all NaN
    if df["deminf_score"].isna().any():
        logger.warning("WARNING: Some episodes have NaN scores!")

    # Check for all same scores
    if df["deminf_score"].nunique() == 1:
        logger.warning("WARNING: All episode scores are identical!")

    # Check for extreme outliers
    q1 = df["deminf_score"].quantile(0.25)
    q3 = df["deminf_score"].quantile(0.75)
    iqr = q3 - q1
    outliers = df[
        (df["deminf_score"] < q1 - 3 * iqr) | (df["deminf_score"] > q3 + 3 * iqr)
    ]
    if len(outliers) > 0:
        logger.warning(f"Found {len(outliers)} outlier episodes (beyond 3*IQR)")

    # Top-K
    logger.info(f"\nTop-{top_k} episodes (highest DemInf score):")
    for _, row in df.head(top_k).iterrows():
        logger.info(
            f"  Rank {row['rank']}: Episode {row['episode_idx']}, "
            f"score={row['deminf_score']:.4f}, steps={row['num_steps']}"
        )

    # Bottom-K
    logger.info(f"\nBottom-{top_k} episodes (lowest DemInf score):")
    for _, row in df.tail(top_k).iterrows():
        logger.info(
            f"  Rank {row['rank']}: Episode {row['episode_idx']}, "
            f"score={row['deminf_score']:.4f}, steps={row['num_steps']}"
        )


def score_dataset(
    state_model: BetaVAE,
    action_model: BetaVAE,
    states: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    timestep_ids: np.ndarray,
    config: DemInfConfig,
    output_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Full scoring pipeline: encode -> KSG -> aggregate.

    Args:
        state_model: Trained state VAE.
        action_model: Trained action VAE.
        states: State vectors [N, D_s].
        actions: Action vectors [N, D_a].
        episode_ids: Episode indices [N].
        timestep_ids: Timestep indices [N].
        config: DemInfConfig.
        output_dir: Directory to save outputs. If None, no files saved.

    Returns:
        Episode score DataFrame.
    """
    # Step 1: Encode all timesteps
    z_s, z_a = encode_all_timesteps(state_model, action_model, states, actions, config)

    # Save latent cache
    if output_dir is not None and config.use_latent_cache:
        save_latent_cache(output_dir, z_s, z_a, episode_ids, timestep_ids)

    # Step 2: Compute KSG scores
    local_scores = compute_timestep_information_scores(z_s, z_a, config)

    # Step 3: Aggregate per episode
    score_df = aggregate_episode_scores(local_scores, episode_ids)

    # Step 4: Sanity check
    sanity_check_scores(score_df)

    # Step 5: Save CSV
    if output_dir is not None:
        ensure_dir(output_dir)
        csv_path = Path(output_dir) / "episode_scores.csv"
        score_df.to_csv(str(csv_path), index=False)
        logger.info(f"Saved episode scores to {csv_path}")

    return score_df