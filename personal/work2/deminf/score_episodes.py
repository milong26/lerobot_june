"""
DemInf Episode Scoring

Official DemInf quality inference pipeline:
1. Encode all timesteps through trained VAEs (posterior mean)
2. Build quality batches with repeat=4, batch_size=1024, drop_remainder=True
3. Batch-local KSG scoring per official estimator
4. Filter NaN, global p1/p99 clipping, global z-score normalization
5. Mean aggregate by episode
6. Rank episodes by deminf_score (descending)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from deminf.config import DemInfConfig
from deminf.ksg import deminf_ksg_batch_scores
from deminf.models import BetaVAE
from deminf.utils import ensure_dir

logger = logging.getLogger("deminf")


def encode_all_timesteps(
    state_model: BetaVAE,
    action_model: BetaVAE,
    states: np.ndarray,
    actions: np.ndarray,
    batch_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode all timesteps through trained VAEs using posterior mean.

    Args:
        state_model: Trained state VAE in eval mode.
        action_model: Trained action VAE in eval mode.
        states: State vectors [N, D_s].
        actions: Action vectors [N, D_a].
        batch_size: Encoding batch size.

    Returns:
        z_s: State latents [N, state_latent_dim].
        z_a: Action latents [N, action_latent_dim].
    """
    device = state_model.encoder.shared[0].weight.device

    state_tensor = torch.from_numpy(states).float()
    action_tensor = torch.from_numpy(actions).float()

    z_s_list = []
    z_a_list = []

    state_model.eval()
    action_model.eval()

    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            end = min(start + batch_size, len(states))
            s_batch = state_tensor[start:end].to(device, non_blocking=True)
            a_batch = action_tensor[start:end].to(device, non_blocking=True)

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
    global_row_ids: np.ndarray,
    manifest: Dict,
) -> None:
    """Save latent embeddings to cache file with manifest."""
    cache_path = Path(output_dir) / "latents.npz"
    np.savez(
        str(cache_path),
        z_state=z_s,
        z_action=z_a,
        episode_ids=episode_ids,
        timestep_ids=timestep_ids,
        global_row_ids=global_row_ids,
    )
    manifest_path = Path(output_dir) / "latents_manifest.json"
    import json
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved latent cache to {cache_path} with manifest {manifest_path}")


def load_latent_cache(
    output_dir: str | Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load latent embeddings from cache file."""
    cache_path = Path(output_dir) / "latents.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"Latent cache not found: {cache_path}")

    data = np.load(str(cache_path))
    global_row_ids = data["global_row_ids"] if "global_row_ids" in data else data["timestep_ids"]
    return data["z_state"], data["z_action"], data["episode_ids"], data["timestep_ids"], global_row_ids


def build_quality_records(
    z_s: np.ndarray,
    z_a: np.ndarray,
    episode_ids: np.ndarray,
    timestep_ids: np.ndarray,
    global_row_ids: np.ndarray,
) -> List[Dict]:
    """
    Build quality inference transition records.

    Each record contains z_state, z_action, episode_idx, timestep_idx, global_row_idx.
    """
    records = []
    for i in range(len(z_s)):
        records.append({
            "z_state": z_s[i],
            "z_action": z_a[i],
            "episode_idx": int(episode_ids[i]),
            "timestep_idx": int(timestep_ids[i]),
            "global_row_idx": int(global_row_ids[i]),
        })
    return records


def build_official_quality_batches(
    records: List[Dict],
    config: DemInfConfig,
    base_seed: int = 42,
) -> List[Tuple[List[Dict], int, int]]:
    """
    Build quality batches mimicking official DemInf quality inference.

    1. Group by episode (terminal transitions already dropped).
    2. Apply effective_discard_fraction per episode (0.0 when quality_cache=True).
    3. Flatten all transitions.
    4. For each repeat (default 4):
       - Deterministic shuffle with seed = base_seed + repeat_id
       - Split into batches of quality_batch_size
       - Drop remainder if quality_drop_remainder=True

    Returns:
        List of (batch_records, repeat_id, batch_id).
    """
    import random

    effective_discard = config.effective_discard_fraction()

    # Group by episode
    ep_groups: Dict[int, List[Dict]] = {}
    for rec in records:
        ep_idx = rec["episode_idx"]
        if ep_idx not in ep_groups:
            ep_groups[ep_idx] = []
        ep_groups[ep_idx].append(rec)

    # Apply discard fraction per episode
    flat_records = []
    for ep_idx, ep_recs in ep_groups.items():
        if effective_discard > 0.0:
            rng = random.Random(base_seed + ep_idx)
            rng.shuffle(ep_recs)
            keep = max(int(len(ep_recs) * (1 - effective_discard)), 1)
            ep_recs = ep_recs[:keep]
        flat_records.extend(ep_recs)

    total_transitions = len(flat_records)
    bs = config.quality_batch_size

    if total_transitions < bs:
        raise ValueError(
            f"Total transitions ({total_transitions}) < quality_batch_size ({bs}). "
            f"Cannot perform official KSG batch scoring. "
            f"Reduce quality_batch_size only in smoke tests."
        )

    batches = []
    for repeat_id in range(config.quality_repeat):
        rng = random.Random(base_seed + repeat_id)
        shuffled = list(flat_records)
        rng.shuffle(shuffled)

        batch_id = 0
        for start in range(0, len(shuffled), bs):
            end = start + bs
            if config.quality_drop_remainder and end > len(shuffled):
                break
            batch = shuffled[start:end]
            if len(batch) == bs:
                batches.append((batch, repeat_id, batch_id))
                batch_id += 1

    logger.info(
        f"Built {len(batches)} quality batches: "
        f"repeat={config.quality_repeat}, batch_size={bs}, "
        f"total_transitions={total_transitions}, "
        f"effective_discard_fraction={effective_discard}"
    )

    return batches


def score_quality_batches(
    batches: List[Tuple[List[Dict], int, int]],
    ks: Tuple[int, ...] = (5, 6, 7),
) -> List[Dict]:
    """
    Score each quality batch using official DemInf KSG estimator.

    Returns list of dicts with scores and metadata.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_results = []

    for batch_records, repeat_id, batch_id in batches:
        z_s = np.stack([r["z_state"] for r in batch_records])
        z_a = np.stack([r["z_action"] for r in batch_records])

        z_s_t = torch.from_numpy(z_s).float().to(device)
        z_a_t = torch.from_numpy(z_a).float().to(device)

        scores = deminf_ksg_batch_scores(z_s_t, z_a_t, ks)

        for i, rec in enumerate(batch_records):
            all_results.append({
                "episode_idx": rec["episode_idx"],
                "timestep_idx": rec["timestep_idx"],
                "global_row_idx": rec["global_row_idx"],
                "repeat_id": repeat_id,
                "batch_id": batch_id,
                "raw_score": float(scores[i]),
            })

    return all_results


def postprocess_scores(
    results: List[Dict],
    config: DemInfConfig,
) -> pd.DataFrame:
    """
    Post-process raw timestep scores:
    1. Filter NaN
    2. Global p1/p99 clipping
    3. Global z-score normalization
    4. Mean aggregate by episode

    Returns episode score DataFrame.
    """
    df = pd.DataFrame(results)

    # Filter NaN
    nan_mask = ~np.isfinite(df["raw_score"])
    n_nan = nan_mask.sum()
    if n_nan > 0:
        logger.info(f"Filtered {n_nan} NaN timestep scores")
        df = df[~nan_mask].reset_index(drop=True)

    scores = df["raw_score"].values.astype(np.float64)

    # Global percentile clipping
    p_low = np.percentile(scores, config.score_clip_low)
    p_high = np.percentile(scores, config.score_clip_high)
    scores_clipped = np.clip(scores, p_low, p_high)
    df["clipped_score"] = scores_clipped

    logger.info(f"Score clipping: p{config.score_clip_low}={p_low:.4f}, p{config.score_clip_high}={p_high:.4f}")

    # Global z-score normalization
    score_mean = np.mean(scores_clipped)
    score_std = np.std(scores_clipped)
    if score_std < 1e-10:
        raise ValueError(
            f"Score std is near zero ({score_std}), cannot normalize. "
            f"All scores may be identical."
        )
    scores_norm = (scores_clipped - score_mean) / score_std
    df["normalized_score"] = scores_norm

    # Mean aggregate by episode
    ep_scores = df.groupby("episode_idx").agg(
        deminf_score=("normalized_score", "mean"),
        num_score_samples=("normalized_score", "count"),
        raw_mean=("raw_score", "mean"),
        raw_std=("raw_score", "std"),
    ).reset_index()

    # Rank (descending by score)
    ep_scores["rank"] = ep_scores["deminf_score"].rank(ascending=False, method="min").astype(int)
    ep_scores = ep_scores.sort_values("rank").reset_index(drop=True)

    logger.info(
        f"Episode scores: {len(ep_scores)} episodes, "
        f"total timestep scores={len(df)}, "
        f"score range=[{ep_scores['deminf_score'].min():.4f}, {ep_scores['deminf_score'].max():.4f}]"
    )

    return ep_scores, df


def score_dataset(
    state_model: BetaVAE,
    action_model: BetaVAE,
    states: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    timestep_ids: np.ndarray,
    global_row_ids: np.ndarray,
    config: DemInfConfig,
    output_dir: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full official DemInf scoring pipeline.

    Returns:
        episode_scores_df: Per-episode scores with rank.
        timestep_scores_df: Per-timestep raw/clipped/normalized scores.
    """
    # Step 1: Encode
    z_s, z_a = encode_all_timesteps(
        state_model, action_model, states, actions,
        batch_size=config.quality_batch_size,
    )

    # Step 2: Build records
    records = build_quality_records(z_s, z_a, episode_ids, timestep_ids, global_row_ids)

    # Step 3: Build quality batches
    batches = build_official_quality_batches(records, config, base_seed=config.seed)

    # Step 4: Score batches
    results = score_quality_batches(batches, ks=config.ks)

    # Step 5: Post-process
    ep_scores_df, ts_scores_df = postprocess_scores(results, config)

    # Save outputs
    if output_dir is not None:
        ensure_dir(output_dir)

        # Save timestep scores
        ts_path = Path(output_dir) / "raw_timestep_scores.csv"
        ts_scores_df.to_csv(str(ts_path), index=False)
        logger.info(f"Saved timestep scores to {ts_path}")

        # Save episode scores
        csv_path = Path(output_dir) / "episode_scores.csv"
        ep_scores_df.to_csv(str(csv_path), index=False)
        logger.info(f"Saved episode scores to {csv_path}")

    return ep_scores_df, ts_scores_df