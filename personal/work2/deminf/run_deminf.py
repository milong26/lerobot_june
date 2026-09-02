#!/usr/bin/env python
"""
DemInf Episode Selection - Main Entry Point

Algorithmically and numerically validated PyTorch/LeRobot reimplementation of
the official DemInf estimator (RSS 2025, state-based).

Usage:
    python run_deminf.py \
        --dataset-path /path/to/lerobot/dataset \
        --output-dir /path/to/output \
        --target-episodes 112 \
        --seed 42 \
        --vae-steps 50000 \
        --vae-lr 1e-4 \
        --vae-batch-size 256 \
        --quality-batch-size 1024 \
        --quality-repeat 4 \
        --state-latent-dim 12 \
        --action-latent-dim 6 \
        --ks 5 6 7 \
        --state-source observation.environment_state

Pipeline:
    1. Set seed
    2. Load LeRobot dataset
    3. Build verified global episode index
    4. Drop terminal transitions
    5. Extract observation.environment_state and action
    6. Validate dimensions and relative action
    7. Fit/save DemInf normalization
    8. Train/load state VAE exactly 50000 optimizer steps
    9. Train/load action VAE exactly 50000 optimizer steps
    10. Encode posterior means
    11. Validate/reuse latent cache
    12. Construct official random repeated quality batches
    13. Batch-local official KSG score
    14. Remove NaNs, global p1/p99 clipping, global z-score
    15. Mean aggregate by episode
    16. Rank, Top-K, subset JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from deminf.config import DemInfConfig
from deminf.dataset_adapter import (
    build_episode_index_from_lerobot,
    check_relative_action,
    collect_training_arrays,
    drop_terminal_transitions,
    infer_episode_structure,
    validate_episode_index,
    DemInfNormalizer,
)
from deminf.models import BetaVAE
from deminf.score_episodes import (
    encode_all_timesteps,
    load_latent_cache,
    save_latent_cache,
    score_dataset,
)
from deminf.select_subset import save_score_rankings, save_subset_json, select_top_episodes
from deminf.train_vae import find_checkpoint, load_vae_checkpoint, train_beta_vae
from deminf.utils import (
    atomic_save_json,
    ensure_dir,
    get_device,
    init_logger,
    save_metadata,
    set_global_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="DemInf Episode Selection (Official State-Based, RSS 2025)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to LeRobot dataset directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for all results")

    # Core parameters
    parser.add_argument("--target-episodes", type=int, default=112,
                        help="Number of episodes to select")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="PyTorch device (cuda, cpu, cuda:0)")

    # VAE training
    parser.add_argument("--vae-steps", type=int, default=50000,
                        help="Number of VAE optimizer steps (official: 50000)")
    parser.add_argument("--vae-lr", type=float, default=1e-4,
                        help="VAE learning rate (official: 1e-4)")
    parser.add_argument("--vae-batch-size", type=int, default=256,
                        help="Batch size for VAE training")
    parser.add_argument("--state-latent-dim", type=int, default=12,
                        help="State VAE latent dimension (official: 12)")
    parser.add_argument("--action-latent-dim", type=int, default=6,
                        help="Action VAE latent dimension (official: 6)")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 6, 7],
                        help="K values for KSG estimator (official: 5 6 7)")
    parser.add_argument("--vae-beta-state", type=float, default=0.05,
                        help="State VAE beta coefficient (official: 0.05)")
    parser.add_argument("--vae-beta-action", type=float, default=0.05,
                        help="Action VAE beta coefficient (official: 0.05)")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay for Adam optimizer (official: 0.0)")

    # Quality inference
    parser.add_argument("--quality-batch-size", type=int, default=1024,
                        help="Batch size for quality KSG scoring (official: 1024)")
    parser.add_argument("--quality-repeat", type=int, default=4,
                        help="Number of quality repeat iterations (official: 4)")
    parser.add_argument("--quality-cache", action="store_true", default=True,
                        help="Use quality cache (forces effective_discard_fraction=0)")
    parser.add_argument("--no-quality-cache", action="store_true",
                        help="Disable quality cache")
    parser.add_argument("--quality-discard-fraction", type=float, default=0.5,
                        help="Discard fraction per episode (overridden to 0 if quality_cache=True)")
    parser.add_argument("--score-clip-low", type=float, default=1.0,
                        help="Lower percentile for score clipping")
    parser.add_argument("--score-clip-high", type=float, default=99.0,
                        help="Upper percentile for score clipping")

    # Data
    parser.add_argument("--state-source", type=str, default="observation.environment_state",
                        help="State source key (official: observation.environment_state)")
    parser.add_argument("--max-timesteps", type=int, default=None,
                        help="Max timesteps per episode (None = all)")

    # Checkpointing
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing checkpoints")
    parser.add_argument("--skip-train-if-checkpoint-exists", action="store_true", default=True,
                        help="Skip VAE training if checkpoint exists")
    parser.add_argument("--no-skip-train", action="store_true",
                        help="Override --skip-train-if-checkpoint-exists")
    parser.add_argument("--use-latent-cache", action="store_true", default=True,
                        help="Load/save latent embeddings from cache")

    # Advanced
    parser.add_argument("--ksg-backend", type=str, default="chunked",
                        choices=["chunked", "full"],
                        help="KSG backend: chunked (memory-safe) or full")
    parser.add_argument("--ksg-mode", type=str, default="deminf_rank",
                        choices=["deminf_rank", "full_mi"],
                        help="KSG scoring mode")

    return parser.parse_args()


def build_cache_fingerprint(
    dataset_path: str,
    state_source: str,
    action_key: str,
    state_dim: int,
    action_dim: int,
    state_latent_dim: int,
    action_latent_dim: int,
    hidden_dims: list,
    vae_beta_state: float,
    vae_beta_action: float,
    vae_lr: float,
    vae_steps: int,
    normalization_manifest: dict,
    state_ckpt_path: str | None,
    action_ckpt_path: str | None,
    git_commit: str,
) -> str:
    """Build a comprehensive fingerprint for latent cache validation."""
    state_ckpt_hash = ""
    if state_ckpt_path and Path(state_ckpt_path).exists():
        with open(state_ckpt_path, "rb") as f:
            state_ckpt_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    action_ckpt_hash = ""
    if action_ckpt_path and Path(action_ckpt_path).exists():
        with open(action_ckpt_path, "rb") as f:
            action_ckpt_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    info_path = Path(dataset_path) / "meta" / "info.json"
    info_hash = ""
    if info_path.exists():
        with open(info_path, "rb") as f:
            info_hash = hashlib.sha256(f.read()).hexdigest()[:16]

    key_fields = {
        "dataset_path": dataset_path,
        "info_hash": info_hash,
        "state_source": state_source,
        "action_key": action_key,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "state_latent_dim": state_latent_dim,
        "action_latent_dim": action_latent_dim,
        "hidden_dims": hidden_dims,
        "vae_beta_state": vae_beta_state,
        "vae_beta_action": vae_beta_action,
        "vae_lr": vae_lr,
        "vae_steps": vae_steps,
        "normalization_manifest": normalization_manifest,
        "state_ckpt_hash": state_ckpt_hash,
        "action_ckpt_hash": action_ckpt_hash,
        "git_commit": git_commit,
    }
    raw = json.dumps(key_fields, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def main() -> None:
    """Run the full official DemInf pipeline."""
    args = parse_args()

    # Build config
    config = DemInfConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        target_episodes=args.target_episodes,
        seed=args.seed,
        device=args.device,
        batch_size=args.vae_batch_size,
        num_workers=4,
        state_source=args.state_source,
        state_latent_dim=args.state_latent_dim,
        action_latent_dim=args.action_latent_dim,
        vae_steps=args.vae_steps,
        vae_lr=args.vae_lr,
        vae_beta_state=args.vae_beta_state,
        vae_beta_action=args.vae_beta_action,
        weight_decay=args.weight_decay,
        ks=tuple(args.ks),
        quality_batch_size=args.quality_batch_size,
        quality_repeat=args.quality_repeat,
        quality_cache=not args.no_quality_cache,
        quality_discard_fraction=args.quality_discard_fraction,
        score_clip_low=args.score_clip_low,
        score_clip_high=args.score_clip_high,
        max_timesteps_per_episode=args.max_timesteps,
        checkpoint_dir=str(Path(args.output_dir) / "checkpoints"),
        resume=args.resume,
        skip_train_if_checkpoint_exists=not args.no_skip_train,
        use_latent_cache=args.use_latent_cache,
        ksg_backend=args.ksg_backend,
        ksg_mode=args.ksg_mode,
    )

    if args.no_skip_train:
        config.skip_train_if_checkpoint_exists = False

    config.validate()
    set_global_seed(config.seed)

    logger = init_logger(config.output_dir)
    logger.info("=" * 60)
    logger.info("DemInf Episode Selection (Official State-Based, RSS 2025)")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config.dataset_path}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Target episodes: {config.target_episodes}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"Device: {config.device}")

    save_metadata(config.output_dir, config)

    # =========================================================================
    # Step 1: Read LeRobot dataset structure
    # =========================================================================
    logger.info("\n--- Step 1: Reading dataset structure ---")
    structure = infer_episode_structure(config.dataset_path)
    logger.info(f"Total episodes: {structure['total_episodes']}")
    logger.info(f"Total frames: {structure['total_frames']}")

    repo_id = str(Path(config.dataset_path).name)

    # =========================================================================
    # Step 2: Build verified global episode index
    # =========================================================================
    logger.info("\n--- Step 2: Building verified global episode index ---")
    episode_map = build_episode_index_from_lerobot(config.dataset_path, repo_id)
    validate_episode_index(config.dataset_path, repo_id, episode_map)

    total_episodes = len(episode_map)
    if config.target_episodes > total_episodes:
        logger.warning(
            f"target_episodes={config.target_episodes} > total_episodes={total_episodes}"
        )

    # =========================================================================
    # Step 3: Drop terminal transitions
    # =========================================================================
    logger.info("\n--- Step 3: Dropping terminal transitions ---")
    episode_map = drop_terminal_transitions(episode_map)

    # =========================================================================
    # Step 4: Extract state and action arrays
    # =========================================================================
    logger.info("\n--- Step 4: Extracting state/action arrays ---")
    states, actions, episode_ids, timestep_ids, global_row_ids = collect_training_arrays(
        dataset_root=config.dataset_path,
        repo_id=repo_id,
        episode_map=episode_map,
        state_source=config.state_source,
        action_key=config.action_key,
        max_timesteps_per_episode=config.max_timesteps_per_episode,
    )

    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    logger.info(f"state_source={config.state_source}, state_dim={state_dim}")
    logger.info(f"action_dim={action_dim}")
    logger.info(f"Total timesteps: {len(states)}")

    assert state_dim == 39, f"Expected state_dim=39, got {state_dim}"
    assert action_dim == 4, f"Expected action_dim=4, got {action_dim}"

    # =========================================================================
    # Step 5: Check relative action
    # =========================================================================
    logger.info("\n--- Step 5: Checking action type ---")
    relative_action, action_evidence = check_relative_action(config.dataset_path, repo_id)
    logger.info(f"Relative action: {relative_action} (evidence: {action_evidence})")

    # =========================================================================
    # Step 6: Fit and save DemInf normalization
    # =========================================================================
    logger.info("\n--- Step 6: Fitting DemInf normalization ---")
    normalizer = DemInfNormalizer(state_dim=state_dim, action_dim=action_dim)
    normalizer.fit(states, actions)

    norm_stats_path = Path(config.output_dir) / "normalization_stats.npz"
    normalizer.save(str(norm_stats_path))

    states = normalizer.normalize_state(states)
    actions = normalizer.normalize_action(actions)
    logger.info(f"Normalized states: mean~{states.mean():.4f}, std~{states.std():.4f}")
    logger.info(f"Normalized actions: mean~{actions.mean():.4f}, std~{actions.std():.4f}")

    normalization_manifest = normalizer.get_manifest()

    # =========================================================================
    # Step 7: Train or load state VAE
    # =========================================================================
    logger.info("\n--- Step 7: Training/loading state VAE ---")
    device = get_device(config.device)

    state_ckpt_path = find_checkpoint(config.checkpoint_dir, "state", config.checkpoint_step)
    action_ckpt_path = find_checkpoint(config.checkpoint_dir, "action", config.checkpoint_step)

    # Check latent cache
    latent_cache_path = Path(config.output_dir) / "latents.npz"
    manifest_path = Path(config.output_dir) / "latents_manifest.json"

    cache_fingerprint = build_cache_fingerprint(
        dataset_path=config.dataset_path,
        state_source=config.state_source,
        action_key=config.action_key,
        state_dim=state_dim,
        action_dim=action_dim,
        state_latent_dim=config.state_latent_dim,
        action_latent_dim=config.action_latent_dim,
        hidden_dims=config.hidden_dims,
        vae_beta_state=config.vae_beta_state,
        vae_beta_action=config.vae_beta_action,
        vae_lr=config.vae_lr,
        vae_steps=config.vae_steps,
        normalization_manifest=normalization_manifest,
        state_ckpt_path=state_ckpt_path,
        action_ckpt_path=action_ckpt_path,
        git_commit="",
    )

    use_cache = False
    if config.use_latent_cache and latent_cache_path.exists() and manifest_path.exists():
        with open(manifest_path, "r") as f:
            cached_manifest = json.load(f)
        cached_fp = cached_manifest.get("fingerprint", "")
        if cached_fp == cache_fingerprint:
            use_cache = True
            logger.info(f"Latent cache fingerprint matches, reusing cache")
        else:
            logger.info(
                f"Latent cache fingerprint mismatch (cached={cached_fp}, current={cache_fingerprint}), "
                f"will re-encode"
            )

    # Train state VAE if needed
    if not state_ckpt_path or not config.skip_train_if_checkpoint_exists:
        state_model, state_log = train_beta_vae(
            data=states,
            input_dim=state_dim,
            latent_dim=config.state_latent_dim,
            config=config,
            name="state",
            normalization_stats=normalization_manifest,
        )
    else:
        logger.info(f"Loading state VAE from {state_ckpt_path}")
        ckpt = load_vae_checkpoint(state_ckpt_path, device)
        state_model = BetaVAE(state_dim, config.state_latent_dim, config.hidden_dims).to(device)
        state_model.load_state_dict(ckpt["model_state_dict"])
        state_model.eval()

    # =========================================================================
    # Step 8: Train or load action VAE
    # =========================================================================
    logger.info("\n--- Step 8: Training/loading action VAE ---")

    if not action_ckpt_path or not config.skip_train_if_checkpoint_exists:
        action_model, action_log = train_beta_vae(
            data=actions,
            input_dim=action_dim,
            latent_dim=config.action_latent_dim,
            config=config,
            name="action",
            normalization_stats=normalization_manifest,
        )
    else:
        logger.info(f"Loading action VAE from {action_ckpt_path}")
        ckpt = load_vae_checkpoint(action_ckpt_path, device)
        action_model = BetaVAE(action_dim, config.action_latent_dim, config.hidden_dims).to(device)
        action_model.load_state_dict(ckpt["model_state_dict"])
        action_model.eval()

    # =========================================================================
    # Step 9: Encode all timesteps
    # =========================================================================
    if not use_cache:
        logger.info("\n--- Step 9: Encoding all timesteps ---")
        z_s, z_a = encode_all_timesteps(
            state_model, action_model, states, actions,
            batch_size=config.quality_batch_size,
        )

        # Save latent cache with manifest
        cache_manifest = {
            "fingerprint": cache_fingerprint,
            "dataset_path": config.dataset_path,
            "state_source": config.state_source,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "state_latent_dim": config.state_latent_dim,
            "action_latent_dim": config.action_latent_dim,
            "normalization_manifest": normalization_manifest,
            "num_transitions": len(z_s),
        }
        save_latent_cache(
            config.output_dir, z_s, z_a, episode_ids, timestep_ids, global_row_ids,
            cache_manifest,
        )
    else:
        z_s, z_a, episode_ids, timestep_ids, global_row_ids = load_latent_cache(config.output_dir)
        logger.info(f"Loaded latent cache: z_s={z_s.shape}, z_a={z_a.shape}")

    # =========================================================================
    # Step 10-16: Official quality inference scoring
    # =========================================================================
    logger.info("\n--- Step 10-16: Official quality inference scoring ---")
    ep_scores_df, ts_scores_df = score_dataset(
        state_model=state_model,
        action_model=action_model,
        states=states,
        actions=actions,
        episode_ids=episode_ids,
        timestep_ids=timestep_ids,
        global_row_ids=global_row_ids,
        config=config,
        output_dir=config.output_dir,
    )

    # =========================================================================
    # Step 17: Select top episodes
    # =========================================================================
    logger.info("\n--- Step 17: Selecting top episodes ---")
    selected_indices = select_top_episodes(
        ep_scores_df, config.target_episodes, tie_break_seed=config.seed
    )

    # =========================================================================
    # Step 18: Save outputs
    # =========================================================================
    logger.info("\n--- Step 18: Saving outputs ---")

    subset_filename = f"deminf_{config.target_episodes}_seed{config.seed}.json"
    subset_path = Path(config.output_dir) / "subsets" / subset_filename
    subset_data = save_subset_json(
        selected_indices, ep_scores_df, config, subset_path,
        relative_action=relative_action,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    rankings_path = Path(config.output_dir) / "score_rankings.csv"
    save_score_rankings(ep_scores_df, selected_indices, rankings_path)

    config_path = Path(config.output_dir) / "config.json"
    atomic_save_json(vars(config), config_path)

    # =========================================================================
    # Summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("DemInf Selection Summary")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config.dataset_path}")
    logger.info(f"Total episodes: {total_episodes}")
    logger.info(f"Total timesteps: {len(states)}")
    logger.info(f"State source: {config.state_source}")
    logger.info(f"State dimension: {state_dim}")
    logger.info(f"Action dimension: {action_dim}")
    logger.info(f"State latent dim: {config.state_latent_dim}")
    logger.info(f"Action latent dim: {config.action_latent_dim}")
    logger.info(f"VAE steps: {config.vae_steps}")
    logger.info(f"VAE lr: {config.vae_lr}")
    logger.info(f"VAE beta (state/action): {config.vae_beta_state}/{config.vae_beta_action}")
    logger.info(f"KSG ks: {config.ks}")
    logger.info(f"Quality batch size: {config.quality_batch_size}")
    logger.info(f"Quality repeat: {config.quality_repeat}")
    logger.info(f"Effective discard fraction: {config.effective_discard_fraction()}")
    logger.info(f"Relative action: {relative_action}")
    logger.info(f"Selected episodes: {len(selected_indices)}")
    logger.info(f"Top episode indices: {selected_indices[:10]}...")
    logger.info(f"\nOutput files:")
    logger.info(f"  Subset JSON: {subset_path}")
    logger.info(f"  Episode scores: {config.output_dir}/episode_scores.csv")
    logger.info(f"  Timestep scores: {config.output_dir}/raw_timestep_scores.csv")
    logger.info(f"  Score rankings: {rankings_path}")
    logger.info(f"  Config: {config_path}")
    logger.info(f"  Normalization stats: {norm_stats_path}")
    logger.info(f"  Latent cache: {latent_cache_path}")
    logger.info(f"  VAE checkpoints: {config.checkpoint_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()