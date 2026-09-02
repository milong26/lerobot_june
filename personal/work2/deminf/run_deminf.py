#!/usr/bin/env python
"""
DemInf Episode Selection - Main Entry Point

Faithful reimplementation of DemInf (Demonstration Information) core algorithm
in the current LeRobot/PyTorch pipeline.

Usage:
    python run_deminf.py \
        --dataset-path /path/to/lerobot/dataset \
        --output-dir /path/to/output \
        --target-episodes 112 \
        --seed 42 \
        --device cuda \
        --vae-epochs 100 \
        --batch-size 256 \
        --score-batch-size 1024 \
        --state-latent-dim 12 \
        --action-latent-dim 6 \
        --ks 5 6 7 \
        --resume

Pipeline:
    1. Set seed
    2. Read LeRobot dataset
    3. Build episode index
    4. Extract per-timestep state/action
    5. Check relative action
    6. Compute and save normalization statistics
    7. Normalize state/action
    8. Train or load state VAE
    9. Train or load action VAE
    10. Batch encode all timesteps
    11. KSG local scoring
    12. Episode mean aggregation
    13. Episode ranking
    14. Select top K
    15. Output files
    16. Print summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from deminf.config import DemInfConfig
from deminf.dataset_adapter import (
    build_episode_index,
    check_relative_action,
    collect_training_arrays,
    compute_normalization_stats,
    infer_episode_structure,
    normalize_array,
)
from deminf.models import BetaVAE
from deminf.score_episodes import load_latent_cache, save_latent_cache, score_dataset
from deminf.select_subset import save_score_rankings, save_subset_json, select_top_episodes
from deminf.train_vae import load_vae_checkpoint, train_beta_vae
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
        description="DemInf Episode Selection",
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
    parser.add_argument("--vae-epochs", type=int, default=100,
                        help="Number of VAE training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for VAE training")
    parser.add_argument("--score-batch-size", type=int, default=1024,
                        help="Batch size for KSG scoring")
    parser.add_argument("--state-latent-dim", type=int, default=12,
                        help="State VAE latent dimension")
    parser.add_argument("--action-latent-dim", type=int, default=6,
                        help="Action VAE latent dimension")
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 6, 7],
                        help="K values for KSG estimator")
    parser.add_argument("--vae-lr", type=float, default=1e-3,
                        help="VAE learning rate")
    parser.add_argument("--vae-beta-state", type=float, default=0.05,
                        help="State VAE beta coefficient")
    parser.add_argument("--vae-beta-action", type=float, default=0.05,
                        help="Action VAE beta coefficient")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="Weight decay for Adam optimizer")

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
    parser.add_argument("--max-timesteps", type=int, default=None,
                        help="Max timesteps per episode (None = all)")

    return parser.parse_args()


def main() -> None:
    """Run the full DemInf pipeline."""
    args = parse_args()

    # Build config
    config = DemInfConfig(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        target_episodes=args.target_episodes,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=4,
        state_latent_dim=args.state_latent_dim,
        action_latent_dim=args.action_latent_dim,
        vae_epochs=args.vae_epochs,
        vae_lr=args.vae_lr,
        vae_beta_state=args.vae_beta_state,
        vae_beta_action=args.vae_beta_action,
        weight_decay=args.weight_decay,
        ks=tuple(args.ks),
        score_batch_size=args.score_batch_size,
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

    # Validate config
    config.validate()

    # Set seed
    set_global_seed(config.seed)

    # Initialize logger
    logger = init_logger(config.output_dir)
    logger.info("=" * 60)
    logger.info("DemInf Episode Selection")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config.dataset_path}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Target episodes: {config.target_episodes}")
    logger.info(f"Seed: {config.seed}")
    logger.info(f"Device: {config.device}")

    # Save metadata
    save_metadata(config.output_dir, config)

    # =========================================================================
    # Step 1: Read LeRobot dataset structure
    # =========================================================================
    logger.info("\n--- Step 1: Reading dataset structure ---")
    structure = infer_episode_structure(config.dataset_path)
    logger.info(f"Total episodes: {structure['total_episodes']}")
    logger.info(f"Total frames: {structure['total_frames']}")
    logger.info(f"FPS: {structure['fps']}")
    logger.info(f"State keys: {structure['state_keys']}")
    logger.info(f"State shape: {structure['state_shape']}")
    logger.info(f"Action shape: {structure['action_shape']}")

    state_keys = structure["state_keys"]
    if not state_keys:
        state_keys = ["observation.state"]
        logger.info(f"Using default state key: {state_keys}")

    # =========================================================================
    # Step 2: Build episode index
    # =========================================================================
    logger.info("\n--- Step 2: Building episode index ---")
    episode_map = build_episode_index(config.dataset_path)
    total_episodes = len(episode_map)

    if config.target_episodes > total_episodes:
        logger.warning(
            f"target_episodes={config.target_episodes} > total_episodes={total_episodes}, "
            f"will select all available episodes"
        )

    # =========================================================================
    # Step 3: Extract per-timestep state/action
    # =========================================================================
    logger.info("\n--- Step 3: Extracting state/action arrays ---")
    states, actions, episode_ids, timestep_ids = collect_training_arrays(
        dataset_root=config.dataset_path,
        episode_map=episode_map,
        state_keys=state_keys,
        action_key=config.action_key,
        max_timesteps_per_episode=config.max_timesteps_per_episode,
    )

    state_dim = states.shape[1]
    action_dim = actions.shape[1]
    logger.info(f"State dimension: {state_dim}")
    logger.info(f"Action dimension: {action_dim}")
    logger.info(f"Total timesteps: {len(states)}")

    # =========================================================================
    # Step 4: Check relative action
    # =========================================================================
    logger.info("\n--- Step 4: Checking action type ---")
    relative_action = check_relative_action(config.dataset_path)
    if config.relative_action is not None:
        relative_action = config.relative_action
    logger.info(f"Relative action: {relative_action}")

    # =========================================================================
    # Step 5: Compute and save normalization statistics
    # =========================================================================
    logger.info("\n--- Step 5: Computing normalization statistics ---")
    state_stats = compute_normalization_stats(states)
    action_stats = compute_normalization_stats(actions)

    norm_stats_path = Path(config.output_dir) / "normalization_stats.npz"
    np.savez(
        str(norm_stats_path),
        state_mean=state_stats["mean"],
        state_std=state_stats["std"],
        action_mean=action_stats["mean"],
        action_std=action_stats["std"],
    )
    logger.info(f"Saved normalization stats to {norm_stats_path}")
    logger.info(f"State mean: {state_stats['mean'][:4]}... (first 4 dims)")
    logger.info(f"State std: {state_stats['std'][:4]}... (first 4 dims)")
    logger.info(f"Action mean: {action_stats['mean']}")
    logger.info(f"Action std: {action_stats['std']}")

    # =========================================================================
    # Step 6: Normalize state/action
    # =========================================================================
    logger.info("\n--- Step 6: Normalizing state/action ---")
    if config.normalize_state:
        states = normalize_array(states, state_stats)
        logger.info(f"Normalized states: mean~{states.mean():.4f}, std~{states.std():.4f}")
    if config.normalize_action:
        actions = normalize_array(actions, action_stats)
        logger.info(f"Normalized actions: mean~{actions.mean():.4f}, std~{actions.std():.4f}")

    # =========================================================================
    # Step 7: Train or load state VAE
    # =========================================================================
    logger.info("\n--- Step 7: Training/loading state VAE ---")
    device = get_device(config.device)

    # Check for latent cache
    latent_cache_path = Path(config.output_dir) / "latents.npz"
    use_cache = config.use_latent_cache and latent_cache_path.exists()

    if use_cache:
        logger.info(f"Loading latent cache from {latent_cache_path}")
        z_s_cached, z_a_cached, ep_ids_cached, ts_ids_cached = load_latent_cache(config.output_dir)
        logger.info(f"Loaded cached latents: z_s={z_s_cached.shape}, z_a={z_a_cached.shape}")
        # We still need VAE models for the checkpoint, but can skip encoding
        # Create dummy models for compatibility
        state_model = BetaVAE(state_dim, config.state_latent_dim, config.hidden_dims).to(device)
        action_model = BetaVAE(action_dim, config.action_latent_dim, config.hidden_dims).to(device)
    else:
        state_model, state_log = train_beta_vae(
            data=states,
            input_dim=state_dim,
            latent_dim=config.state_latent_dim,
            config=config,
            name="state",
            normalization_stats=state_stats,
        )
        logger.info(f"State VAE training complete")

    # =========================================================================
    # Step 8: Train or load action VAE
    # =========================================================================
    logger.info("\n--- Step 8: Training/loading action VAE ---")

    if not use_cache:
        action_model, action_log = train_beta_vae(
            data=actions,
            input_dim=action_dim,
            latent_dim=config.action_latent_dim,
            config=config,
            name="action",
            normalization_stats=action_stats,
        )
        logger.info(f"Action VAE training complete")

    # =========================================================================
    # Step 9: Batch encode all timesteps (if not using cache)
    # =========================================================================
    if not use_cache:
        logger.info("\n--- Step 9: Encoding all timesteps ---")
        from deminf.score_episodes import encode_all_timesteps

        z_s, z_a = encode_all_timesteps(
            state_model, action_model, states, actions, config
        )
        save_latent_cache(config.output_dir, z_s, z_a, episode_ids, timestep_ids)
    else:
        z_s = z_s_cached
        z_a = z_a_cached
        episode_ids = ep_ids_cached
        timestep_ids = ts_ids_cached

    # =========================================================================
    # Step 10: KSG local scoring
    # =========================================================================
    logger.info("\n--- Step 10: KSG local scoring ---")
    from deminf.score_episodes import compute_timestep_information_scores

    local_scores = compute_timestep_information_scores(z_s, z_a, config)

    # =========================================================================
    # Step 11: Episode mean aggregation
    # =========================================================================
    logger.info("\n--- Step 11: Episode aggregation ---")
    from deminf.score_episodes import aggregate_episode_scores, sanity_check_scores

    score_df = aggregate_episode_scores(local_scores, episode_ids)
    sanity_check_scores(score_df)

    # Save CSV
    csv_path = Path(config.output_dir) / "episode_scores.csv"
    score_df.to_csv(str(csv_path), index=False)
    logger.info(f"Saved episode scores to {csv_path}")

    # =========================================================================
    # Step 12: Select top K episodes
    # =========================================================================
    logger.info("\n--- Step 12: Selecting top episodes ---")
    selected_indices = select_top_episodes(
        score_df, config.target_episodes, tie_break_seed=config.seed
    )

    # =========================================================================
    # Step 13: Save outputs
    # =========================================================================
    logger.info("\n--- Step 13: Saving outputs ---")

    # Subset JSON
    subset_filename = f"deminf_{config.target_episodes}_seed{config.seed}.json"
    subset_path = Path(config.output_dir) / "subsets" / subset_filename
    subset_data = save_subset_json(
        selected_indices, score_df, config, subset_path,
        relative_action=relative_action,
    )

    # Score rankings
    rankings_path = Path(config.output_dir) / "score_rankings.csv"
    save_score_rankings(score_df, selected_indices, rankings_path)

    # Config
    config_path = Path(config.output_dir) / "config.json"
    atomic_save_json(vars(config), config_path)

    # =========================================================================
    # Step 14: Print summary
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("DemInf Selection Summary")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config.dataset_path}")
    logger.info(f"Total episodes: {total_episodes}")
    logger.info(f"Total timesteps: {len(states)}")
    logger.info(f"State dimension: {state_dim}")
    logger.info(f"Action dimension: {action_dim}")
    logger.info(f"State latent dim: {config.state_latent_dim}")
    logger.info(f"Action latent dim: {config.action_latent_dim}")
    logger.info(f"KSG ks: {config.ks}")
    logger.info(f"KSG mode: {config.ksg_mode}")
    logger.info(f"Relative action: {relative_action}")
    logger.info(f"Selected episodes: {len(selected_indices)}")
    logger.info(f"Top episode indices: {selected_indices[:10]}...")
    logger.info(f"\nOutput files:")
    logger.info(f"  Subset JSON: {subset_path}")
    logger.info(f"  Episode scores: {csv_path}")
    logger.info(f"  Score rankings: {rankings_path}")
    logger.info(f"  Config: {config_path}")
    logger.info(f"  Normalization stats: {norm_stats_path}")
    logger.info(f"  Latent cache: {latent_cache_path}")
    logger.info(f"  VAE checkpoints: {config.checkpoint_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()