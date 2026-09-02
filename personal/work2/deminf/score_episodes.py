"""
DemInf Episode Scoring

Official DemInf quality inference pipeline:
1. Encode all timesteps through trained VAEs (posterior mean) [via encode_all_timesteps]
2. Build quality batches with repeat=4, batch_size=1024, drop_remainder=True
3. Batch-local KSG scoring per official estimator
4. Filter NaN (NOT Inf), global p1/p99 clipping, global z-score normalization
5. Mean aggregate by episode
6. Rank episodes by deminf_score (descending)

The core scoring function is score_latents(), which operates on pre-computed
latents and is the only entry point for the official quality pipeline.
score_dataset() is kept as a compatibility wrapper that encodes then delegates
to score_latents().
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from deminf.config import DemInfConfig
from deminf.ksg import deminf_ksg_batch_scores
from deminf.models import BetaVAE
from deminf.utils import ensure_dir, validate_latent_cache_metadata

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
    """
    Save latent embeddings to cache file with comprehensive manifest.

    The manifest includes fingerprint, git_commit, checkpoint hashes,
    dataset info hash, latent shapes, and array hashes for full validation.

    Args:
        output_dir: Directory to save cache files.
        z_s: State latents [N, state_latent_dim].
        z_a: Action latents [N, action_latent_dim].
        episode_ids: Episode index per transition [N].
        timestep_ids: Timestep index per transition [N].
        global_row_ids: Global HF dataset row index per transition [N].
        manifest: Dictionary with fingerprint and metadata.
    """
    cache_path = Path(output_dir) / "latents.npz"
    np.savez(
        str(cache_path),
        z_state=z_s,
        z_action=z_a,
        episode_ids=episode_ids,
        timestep_ids=timestep_ids,
        global_row_ids=global_row_ids,
    )

    # Enhance manifest with array hashes and shapes
    enhanced_manifest = dict(manifest)
    enhanced_manifest["z_state_shape"] = list(z_s.shape)
    enhanced_manifest["z_action_shape"] = list(z_a.shape)
    enhanced_manifest["episode_ids_hash"] = _array_hash_int64(episode_ids)
    enhanced_manifest["timestep_ids_hash"] = _array_hash_int64(timestep_ids)
    enhanced_manifest["global_row_ids_hash"] = _array_hash_int64(global_row_ids)
    enhanced_manifest["num_transitions"] = len(z_s)

    manifest_path = Path(output_dir) / "latents_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(enhanced_manifest, f, indent=2)
    logger.info(f"Saved latent cache to {cache_path} with manifest {manifest_path}")


def _array_hash_int64(arr: np.ndarray) -> str:
    """SHA256 hash of an int64 array for cache validation."""
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def validate_latent_cache(
    output_dir: str | Path,
    expected_fingerprint: str,
    expected_num_transitions: Optional[int] = None,
    expected_global_row_ids: Optional[np.ndarray] = None,
    expected_z_state_shape: Optional[Tuple[int, int]] = None,
    expected_z_action_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate that a latent cache file matches the current experiment.

    Checks:
    - npz and manifest files exist
    - Fingerprint matches
    - z_state/z_action lengths match expected_num_transitions
    - episode_ids/timestep_ids/global_row_ids array lengths match
    - Latent dimensions are correct
    - No NaN/Inf in latent arrays
    - global_row_ids hash matches current data

    Args:
        output_dir: Directory containing cache files.
        expected_fingerprint: Expected cache fingerprint string.
        expected_num_transitions: Expected number of transitions (N).
        expected_global_row_ids: Current global row ids for hash comparison.
        expected_z_state_shape: Expected z_state shape (N, latent_dim).
        expected_z_action_shape: Expected z_action shape (N, latent_dim).

    Returns:
        Tuple of (is_valid, list_of_mismatch_reasons).
    """
    reasons = []
    cache_path = Path(output_dir) / "latents.npz"
    manifest_path = Path(output_dir) / "latents_manifest.json"

    if not cache_path.exists():
        reasons.append(f"Latent cache file not found: {cache_path}")
        return False, reasons

    if not manifest_path.exists():
        reasons.append(f"Latent manifest file not found: {manifest_path}")
        return False, reasons

    with open(manifest_path, "r") as f:
        cached_manifest = json.load(f)

    # Check fingerprint
    cached_fp = cached_manifest.get("fingerprint", "")
    if cached_fp != expected_fingerprint:
        reasons.append(
            f"fingerprint mismatch: cached='{cached_fp}', expected='{expected_fingerprint}'"
        )

    # Load cache data
    data = np.load(str(cache_path))
    z_s = data["z_state"]
    z_a = data["z_action"]
    ep_ids = data["episode_ids"]
    ts_ids = data["timestep_ids"]
    gr_ids = data["global_row_ids"] if "global_row_ids" in data else data["timestep_ids"]

    # Check lengths
    n = len(z_s)
    if expected_num_transitions is not None and n != expected_num_transitions:
        reasons.append(
            f"num_transitions mismatch: cache={n}, expected={expected_num_transitions}"
        )

    if len(z_a) != n:
        reasons.append(f"z_action length {len(z_a)} != z_state length {n}")
    if len(ep_ids) != n:
        reasons.append(f"episode_ids length {len(ep_ids)} != z_state length {n}")
    if len(ts_ids) != n:
        reasons.append(f"timestep_ids length {len(ts_ids)} != z_state length {n}")
    if len(gr_ids) != n:
        reasons.append(f"global_row_ids length {len(gr_ids)} != z_state length {n}")

    # Check shapes
    if expected_z_state_shape is not None and z_s.shape != expected_z_state_shape:
        reasons.append(
            f"z_state shape mismatch: cache={z_s.shape}, expected={expected_z_state_shape}"
        )
    if expected_z_action_shape is not None and z_a.shape != expected_z_action_shape:
        reasons.append(
            f"z_action shape mismatch: cache={z_a.shape}, expected={expected_z_action_shape}"
        )

    # Check for NaN/Inf
    if not np.all(np.isfinite(z_s)):
        reasons.append("z_state contains NaN or Inf")
    if not np.all(np.isfinite(z_a)):
        reasons.append("z_action contains NaN or Inf")

    # Check global_row_ids hash
    if expected_global_row_ids is not None:
        cached_hash = cached_manifest.get("global_row_ids_hash", "")
        current_hash = _array_hash_int64(expected_global_row_ids)
        if cached_hash != current_hash:
            reasons.append(
                f"global_row_ids hash mismatch: cached='{cached_hash}', current='{current_hash}'"
            )

    return (len(reasons) == 0, reasons)


def load_latent_cache(
    output_dir: str | Path,
    current_config: Optional[DemInfConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load latent embeddings from cache file.

    Before loading, validates cache metadata against current experiment config
    if current_config is provided. Raises RuntimeError on validation failure.

    Args:
        output_dir: Directory containing cache files.
        current_config: Current DemInfConfig for validation. If None, skips validation.

    Returns:
        Tuple of (z_state, z_action, episode_ids, timestep_ids, global_row_ids).
    """
    cache_path = Path(output_dir) / "latents.npz"
    manifest_path = Path(output_dir) / "latents_manifest.json"

    if not cache_path.exists():
        raise FileNotFoundError(f"Latent cache not found: {cache_path}")

    if current_config is not None:
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Latent manifest not found: {manifest_path}. "
                f"Cannot validate cache without manifest."
            )
        with open(manifest_path, "r") as f:
            metadata = json.load(f)

        is_valid, error_reason = validate_latent_cache_metadata(metadata, current_config)
        if not is_valid:
            raise RuntimeError(
                f"Latent cache validation failed: {error_reason}. "
                f"Cache does not match current experiment configuration."
            )

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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Post-process raw timestep scores:
    1. Filter NaN only (NOT Inf, matching official ~jnp.isnan(pred))
    2. Global p1/p99 clipping
    3. Global z-score normalization
    4. Mean aggregate by episode

    Official DemInf only filters NaN. If Inf values exist after NaN filtering,
    a ValueError is raised to flag degenerate latents rather than silently
    removing them.

    Returns:
        ep_scores_df: Per-episode scores with rank.
        ts_scores_df: Per-timestep raw/clipped/normalized scores.
    """
    df = pd.DataFrame(results)

    # Filter NaN only (official semantics: ~jnp.isnan(pred))
    nan_mask = np.isnan(df["raw_score"].values)
    n_nan = nan_mask.sum()
    if n_nan > 0:
        logger.info(f"Filtered {n_nan} NaN timestep scores")
        df = df[~nan_mask].reset_index(drop=True)

    scores = df["raw_score"].values.astype(np.float64)

    # Check for Inf after NaN filtering (official only filters NaN)
    inf_mask = np.isinf(scores)
    if inf_mask.any():
        n_inf = inf_mask.sum()
        raise ValueError(
            f"DemInf KSG produced {n_inf} Inf score(s); official pipeline only "
            f"filters NaN, investigate duplicate/degenerate latents"
        )

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


def save_episode_scores(
    ep_scores_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Save episode scores to CSV.

    Fields: episode_id, deminf_score, num_timestep_samples, raw_score_mean,
            normalized_score_mean, rank
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    save_df = ep_scores_df.rename(columns={"episode_idx": "episode_id"}).copy()
    save_df = save_df.rename(columns={
        "num_score_samples": "num_timestep_samples",
        "raw_mean": "raw_score_mean",
        "normalized_score": "deminf_score",
    })

    if "normalized_score_mean" not in save_df.columns:
        save_df["normalized_score_mean"] = save_df["deminf_score"]

    cols = ["episode_id", "deminf_score", "num_timestep_samples",
            "raw_score_mean", "normalized_score_mean", "rank"]
    available_cols = [c for c in cols if c in save_df.columns]
    save_df = save_df[available_cols]

    csv_path = output_dir / "episode_scores.csv"
    save_df.to_csv(str(csv_path), index=False)
    logger.info(f"Saved episode scores to {csv_path}")


def save_timestep_scores(
    ts_scores_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """
    Save timestep scores to CSV.

    Fields: episode_id, timestep_id, raw_ksg_score, normalized_score,
            repeat_id, batch_id
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    save_df = ts_scores_df.rename(columns={
        "episode_idx": "episode_id",
        "timestep_idx": "timestep_id",
        "raw_score": "raw_ksg_score",
    })

    cols = ["episode_id", "timestep_id", "raw_ksg_score", "normalized_score",
            "repeat_id", "batch_id"]
    available_cols = [c for c in cols if c in save_df.columns]
    save_df = save_df[available_cols]

    csv_path = output_dir / "timestep_scores.csv"
    save_df.to_csv(str(csv_path), index=False)
    logger.info(f"Saved timestep scores to {csv_path}")


def score_latents(
    z_s: np.ndarray,
    z_a: np.ndarray,
    episode_ids: np.ndarray,
    timestep_ids: np.ndarray,
    global_row_ids: np.ndarray,
    config: DemInfConfig,
    output_dir: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Core quality scoring function operating on pre-computed latents.

    This is the ONLY entry point for the official DemInf quality pipeline.
    It does NOT receive VAE models or raw states/actions; it works entirely
    on posterior mean latent vectors.

    Pipeline:
    1. Validate latent shapes (N, latent_dim consistency)
    2. Build quality records from latents
    3. Build official quality batches (repeat=4, shuffle, batch_size=1024, drop_remainder)
    4. Score each batch with deminf_ksg_batch_scores
    5. Post-process: filter NaN, p1/p99 clipping, global z-score, episode mean
    6. Save raw_timestep_scores.csv and episode_scores.csv

    Args:
        z_s: State latents [N, state_latent_dim].
        z_a: Action latents [N, action_latent_dim].
        episode_ids: Episode index per transition [N].
        timestep_ids: Timestep index per transition [N].
        global_row_ids: Global HF dataset row index per transition [N].
        config: DemInfConfig with quality inference parameters.
        output_dir: Optional directory to save CSV outputs.

    Returns:
        episode_scores_df: Per-episode scores with rank.
        timestep_scores_df: Per-timestep raw/clipped/normalized scores.
    """
    # Validate shapes
    N = len(z_s)
    assert len(z_a) == N, f"z_a length {len(z_a)} != z_s length {N}"
    assert len(episode_ids) == N, f"episode_ids length {len(episode_ids)} != {N}"
    assert len(timestep_ids) == N, f"timestep_ids length {len(timestep_ids)} != {N}"
    assert len(global_row_ids) == N, f"global_row_ids length {len(global_row_ids)} != {N}"
    assert z_s.ndim == 2, f"z_s should be 2D, got {z_s.ndim}"
    assert z_a.ndim == 2, f"z_a should be 2D, got {z_a.ndim}"

    # Build records
    records = build_quality_records(z_s, z_a, episode_ids, timestep_ids, global_row_ids)

    # Build quality batches
    batches = build_official_quality_batches(records, config, base_seed=config.seed)

    # Score batches
    results = score_quality_batches(batches, ks=config.ks)

    # Post-process
    ep_scores_df, ts_scores_df = postprocess_scores(results, config)

    # Save outputs
    if output_dir is not None:
        ensure_dir(output_dir)
        save_episode_scores(ep_scores_df, output_dir)
        save_timestep_scores(ts_scores_df, output_dir)

    return ep_scores_df, ts_scores_df


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
    COMPATIBILITY WRAPPER ONLY - NOT USED IN OFFICIAL run_deminf.py PIPELINE.

    This function encodes raw states/actions through VAEs then delegates to
    score_latents(). The official pipeline in run_deminf.py directly calls
    score_latents() with pre-computed latents.

    Args:
        state_model: Trained state VAE in eval mode.
        action_model: Trained action VAE in eval mode.
        states: State vectors [N, D_s].
        actions: Action vectors [N, D_a].
        episode_ids: Episode index per transition [N].
        timestep_ids: Timestep index per transition [N].
        global_row_ids: Global HF dataset row index per transition [N].
        config: DemInfConfig with quality inference parameters.
        output_dir: Optional directory to save CSV outputs.

    Returns:
        episode_scores_df: Per-episode scores with rank.
        timestep_scores_df: Per-timestep raw/clipped/normalized scores.
    """
    # Encode (compatibility step; official path uses pre-computed latents)
    z_s, z_a = encode_all_timesteps(
        state_model, action_model, states, actions,
        batch_size=config.quality_batch_size,
    )

    # Delegate to core scoring function
    return score_latents(
        z_s, z_a, episode_ids, timestep_ids, global_row_ids,
        config, output_dir,
    )