#!/usr/bin/env python
"""
Main experiment runner for Ours methods.
Supports stages: pool, embed, select, train_eval, search_budget, all

Usage:
    python run_experiment.py --stage all
    python run_experiment.py --stage pool
    python run_experiment.py --stage embed
    python run_experiment.py --stage select --strategy sic --target-size 20
    python run_experiment.py --stage train-eval --subset-file ours_sic_b0_fps_size_20_subset.json
    python run_experiment.py --stage search-budget
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Set paths
BASE_DIR = Path(__file__).parent
POOL_DIR = BASE_DIR / "pool"
PCA_DIR = BASE_DIR / "pca"
SUBSETS_DIR = BASE_DIR / "subsets"
EXPERIMENTS_DIR = BASE_DIR / "experiments"
EVAL_DIR = BASE_DIR / "eval"
RESULTS_DIR = BASE_DIR / "results"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# LeRobot project root
LEROBOT_ROOT = Path(__file__).parent.parent.parent.parent

# Dataset info
DATASET_REPO_ID = "lerobot/metaworld_pick_place"
DATASET_ROOT = str(BASE_DIR.parent / "dataset")

# GPU settings (GPU 3 and 4 are unavailable)
AVAILABLE_GPUS = ["0", "1", "2"]  # Only use GPUs 0, 1, 2


def stage_pool():
    """Stage 1: Build candidate pool."""
    print("\n" + "=" * 60)
    print("STAGE 1: Building candidate pool")
    print("=" * 60)

    from build_pool import build_pool
    obj_pos, goal_pos, ep_indices, stats = build_pool()
    return obj_pos, goal_pos, ep_indices, stats


def stage_embed(num_keyframes=5, pca_dim=32, pooling="concat", force=False):
    """Stage 2: Extract VLM embeddings with temporal keyframe sampling."""
    print("\n" + "=" * 60)
    print(f"STAGE 2: Extracting VLM embeddings (K={num_keyframes}, pool={pooling}, PCA={pca_dim})")
    print("=" * 60)

    from extract_embeddings import load_embeddings
    raw_emb, pca_emb, episode_indices, pca_model = load_embeddings(
        num_keyframes=num_keyframes,
        n_components=pca_dim,
        pooling=pooling,
        force=force,
    )
    return raw_emb, pca_emb, episode_indices, pca_model


def stage_select(strategy="sic", target_size=20, b0_method="fps", b0_grid_size=3,
                 num_keyframes=5, pca_dim=32, pooling="concat"):
    """Stage 3: Select subset using Ours strategy."""
    print("\n" + "=" * 60)
    print(f"STAGE 3: Selecting subset (strategy={strategy}, target={target_size}, pool={pooling})")
    print("=" * 60)

    # Load VLM embeddings
    from extract_embeddings import load_embeddings
    raw_emb, pca_emb, episode_indices, _ = load_embeddings(
        num_keyframes=num_keyframes,
        n_components=pca_dim,
        pooling=pooling,
    )

    # Use PCA embeddings for selection (N, pca_dim)
    features = pca_emb

    # Load obj positions for B0
    import pandas as pd
    df = pd.read_csv(POOL_DIR / "episode_metadata.csv")
    obj_positions = df[["obj_x", "obj_y"]].values

    # Initialize B0
    from selection_strategies import B0Initializer, OursSelector

    if b0_method == "uniform_grid":
        b0_indices = B0Initializer.uniform_grid(obj_positions, grid_size=b0_grid_size)
    elif b0_method == "quantile_grid":
        b0_indices = B0Initializer.quantile_grid(obj_positions, grid_size=b0_grid_size)
    elif b0_method == "random":
        b0_indices = B0Initializer.random(obj_positions, n=b0_grid_size * b0_grid_size)
    elif b0_method == "fps":
        b0_indices = B0Initializer.fps(obj_positions, n=b0_grid_size * b0_grid_size)
    else:
        b0_indices = None

    print(f"B0: {len(b0_indices)} episodes via {b0_method}")

    # Select
    selector = OursSelector(features, obj_positions, episode_indices, strategy=strategy)
    selected = selector.select(target_size, b0_indices=b0_indices)

    # Save
    method_name = f"select_{strategy}_k{num_keyframes}_{pooling}_ts{target_size}"
    config = {
        "strategy": strategy,
        "b0_method": b0_method,
        "b0_grid_size": b0_grid_size,
        "target_size": target_size,
        "num_keyframes": num_keyframes,
        "pooling": pooling,
        "pca_dim": pca_dim,
    }
    result = selector.save_selection(method_name, config)

    print(f"\nSelected {len(selected)} episodes:")
    print(f"  Indices: {selected[:20]}{'...' if len(selected) > 20 else ''}")

    return selected, result


def stage_train_eval(subset_file, gpu_id=0, num_steps=30000, eval_interval=5000, eval_episodes=50, seed=42):
    """Stage 4: Train VLA on selected subset and evaluate."""
    print("\n" + "=" * 60)
    print(f"STAGE 4: Training and Evaluation")
    print(f"  Subset: {subset_file}")
    print(f"  GPU: {gpu_id}")
    print(f"  Steps: {num_steps}")
    print("=" * 60)

    # Load subset
    subset_path = SUBSETS_DIR / subset_file
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_path}")
    
    with open(subset_path) as f:
        subset_data = json.load(f)
    
    selected_indices = subset_data["selected_episode_indices"]
    print(f"Training on {len(selected_indices)} episodes: {selected_indices[:10]}...")

    # Create output directory
    exp_name = subset_path.stem
    output_dir = CHECKPOINTS_DIR / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build episode indices string for lerobot-train
    episodes_str = "[" + ",".join(str(i) for i in selected_indices) + "]"

    # Build training command
    cmd = [
        "lerobot-train",
        f"--policy.type=smolvla",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        f"--dataset.root={DATASET_ROOT}",
        f"--dataset.episodes={episodes_str}",
        f"--dataset.eval_split=0.0",  # We'll do manual eval
        f"--policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        f"--policy.freeze_vision_encoder=True",
        f"--policy.train_expert_only=True",
        f"--policy.train_state_proj=True",
        f"--policy.optimizer_lr=1e-4",
        f"--steps={num_steps}",
        f"--eval.n_episodes={eval_episodes}",
        f"--eval.batch_size=1",
        f"--eval.freq={eval_interval}",
        f"--seed={seed}",
        f"--output_dir={output_dir}",
        f"--device=cuda:{gpu_id}",
        "--wandb.enable=False",
    ]

    # Add MetaWorld environment config
    cmd.extend([
        "--env.type=metaworld",
        "--env.task=pick-place-v3",
    ])

    print(f"\nRunning training command:")
    print(" ".join(cmd))
    print("\n" + "-" * 60)

    # Set GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Run training
    start_time = time.time()
    result = subprocess.run(cmd, env=env, cwd=str(LEROBOT_ROOT))
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"\nTraining failed with return code {result.returncode}")
        return None

    print(f"\nTraining completed in {elapsed:.1f}s")

    # Save experiment record
    exp_record = {
        "exp_name": exp_name,
        "subset_file": subset_file,
        "num_episodes": len(selected_indices),
        "selected_indices": selected_indices,
        "num_steps": num_steps,
        "eval_episodes": eval_episodes,
        "seed": seed,
        "gpu_id": gpu_id,
        "output_dir": str(output_dir),
        "elapsed_time": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record_path = RESULTS_DIR / f"{exp_name}_record.json"
    with open(record_path, "w") as f:
        json.dump(exp_record, f, indent=2)

    print(f"Experiment record saved to {record_path}")
    return exp_record


def stage_search_budget(strategies=None, b0_methods=None, target_sizes=None, 
                        num_keyframes=5, pca_dim=32, gpu_id=0, 
                        num_steps=30000, eval_episodes=50):
    """Stage 5: Search for minimum effective subset size."""
    print("\n" + "=" * 60)
    print("STAGE 5: Budget Search")
    print("=" * 60)

    if strategies is None:
        strategies = ["sic", "coverage_greedy", "fps_embedding"]
    if b0_methods is None:
        b0_methods = ["fps", "uniform_grid"]
    if target_sizes is None:
        target_sizes = [50, 100, 150, 200, 300, 400, 500]

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for strategy in strategies:
        for b0_method in b0_methods:
            for target_size in target_sizes:
                b0_size = min(9, target_size // 2)
                if b0_size < 4:
                    b0_size = 4

                print(f"\n{'='*40}")
                print(f"Testing: strategy={strategy}, b0={b0_method}, size={target_size}")
                print(f"{'='*40}")

                # Select subset
                try:
                    selected, _ = stage_select(
                        strategy=strategy,
                        target_size=target_size,
                        b0_method=b0_method,
                        b0_grid_size=3,
                        num_keyframes=num_keyframes,
                        pca_dim=pca_dim,
                    )
                except Exception as e:
                    print(f"  Selection failed: {e}")
                    continue

                # Train and eval
                subset_file = f"search_{strategy}_b0_{b0_method}_size_{target_size}_subset.json"
                try:
                    record = stage_train_eval(
                        subset_file=subset_file,
                        gpu_id=gpu_id,
                        num_steps=num_steps,
                        eval_episodes=eval_episodes,
                    )
                except Exception as e:
                    print(f"  Training failed: {e}")
                    continue

                if record:
                    all_results.append({
                        "strategy": strategy,
                        "b0_method": b0_method,
                        "subset_size": target_size,
                        "record": record,
                    })

    # Save all results
    results_path = EXPERIMENTS_DIR / "budget_search_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Budget Search Complete")
    print(f"{'='*60}")
    print(f"Total experiments: {len(all_results)}")
    print(f"Results saved to {results_path}")

    return all_results


def stage_all():
    """Run all stages."""
    obj_pos, goal_pos, ep_indices, stats = stage_pool()
    raw_emb, pca_emb, ep_indices, pca_model = stage_embed(num_keyframes=5, pca_dim=32)
    selected, result = stage_select(
        strategy="sic", target_size=20, b0_method="fps", b0_grid_size=3
    )
    return selected


def main():
    parser = argparse.ArgumentParser(description="Ours experiment runner")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["pool", "embed", "select", "train-eval", "search-budget", "all"],
                        help="Which stage to run")
    parser.add_argument("--strategy", type=str, default="sic",
                        choices=["sic", "coverage_greedy", "fps_embedding", "undercovered"])
    parser.add_argument("--target-size", type=int, default=20)
    parser.add_argument("--b0-method", type=str, default="fps",
                        choices=["fps", "uniform_grid", "quantile_grid", "random"])
    parser.add_argument("--b0-grid-size", type=int, default=3)
    parser.add_argument("--num-keyframes", type=int, default=5, choices=[3, 5, 7, 9],
                        help="Number of temporal keyframes (default: 5)")
    parser.add_argument("--pooling", type=str, default="concat", choices=["max", "mean", "concat"],
                        help="Pooling method for keyframe embeddings (default: concat)")
    parser.add_argument("--pca-dim", type=int, default=32,
                        help="PCA dimension (default: 32)")
    parser.add_argument("--force-embed", action="store_true",
                        help="Force re-extract VLM embeddings")
    parser.add_argument("--subset-file", type=str, default=None,
                        help="Subset JSON file for train-eval stage")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="GPU ID to use (0, 1, or 2)")
    parser.add_argument("--num-steps", type=int, default=30000,
                        help="Number of training steps")
    parser.add_argument("--eval-interval", type=int, default=5000,
                        help="Evaluation interval in steps")
    parser.add_argument("--eval-episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print(f"Ours Experiment Runner")
    print(f"Stage: {args.stage}")
    print(f"Strategy: {args.strategy}")
    print(f"Target size: {args.target_size}")
    print(f"B0 method: {args.b0_method}")
    print(f"Num keyframes: {args.num_keyframes}")
    print(f"Pooling: {args.pooling}")
    print(f"PCA dim: {args.pca_dim}")

    if args.stage == "pool":
        stage_pool()
    elif args.stage == "embed":
        stage_embed(num_keyframes=args.num_keyframes, pca_dim=args.pca_dim, pooling=args.pooling, force=args.force_embed)
    elif args.stage == "select":
        stage_select(
            strategy=args.strategy,
            target_size=args.target_size,
            b0_method=args.b0_method,
            b0_grid_size=args.b0_grid_size,
            num_keyframes=args.num_keyframes,
            pooling=args.pooling,
            pca_dim=args.pca_dim,
        )
    elif args.stage == "train-eval":
        if args.subset_file is None:
            print("Error: --subset-file is required for train-eval stage")
            sys.exit(1)
        stage_train_eval(
            subset_file=args.subset_file,
            gpu_id=args.gpu_id,
            num_steps=args.num_steps,
            eval_interval=args.eval_interval,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
        )
    elif args.stage == "search-budget":
        stage_search_budget(
            num_keyframes=args.num_keyframes,
            pca_dim=args.pca_dim,
            gpu_id=args.gpu_id,
            num_steps=args.num_steps,
            eval_episodes=args.eval_episodes,
        )
    elif args.stage == "all":
        stage_all()


if __name__ == "__main__":
    main()