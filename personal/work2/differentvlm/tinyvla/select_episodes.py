#!/usr/bin/env python
"""
Select episodes from LeRobot dataset using different strategies.

Supported methods:
    - random: Random episode selection
    - grid_uniform: Uniform spacing across episodes
    - our_v5: Adaptive coverage selection based on V5 embeddings
    - deminf: Mutual information-based episode selection (DemInf)

Usage:
    python select_episodes.py --method random --num-episodes 100 --seed 42 \
        --dataset-root /path/to/dataset --output-dir /path/to/output
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def load_episode_count(dataset_root: str) -> int:
    """Load episode count from LeRobotDataset."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"Loading LeRobotDataset from {dataset_root}")
    dataset = LeRobotDataset(repo_id="1/2", root=dataset_root)
    total_episodes = dataset.num_episodes
    print(f"Total episodes: {total_episodes}")
    return total_episodes


def select_random(
    dataset_root: str,
    num_episodes: int,
    seed: int,
    output_dir: str,
) -> list[int]:
    """
    Select episodes randomly using the official script.
    
    Calls: personal/work2/duibi/train_and_eval_scripts/select_random_episodes.py
    """
    print(f"\n{'='*60}")
    print(f"Random Episode Selection")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_root}")
    print(f"Num episodes: {num_episodes}")
    print(f"Seed: {seed}")
    sys.stdout.flush()

    stage_start = time.time()

    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    WORK2_ROOT = Path(__file__).resolve().parents[3]

    select_script = WORK2_ROOT / "duibi" / "train_and_eval_scripts" / "select_random_episodes.py"
    if not select_script.exists():
        raise FileNotFoundError(f"Selection script not found: {select_script}")

    cmd = [
        sys.executable, str(select_script),
        "--num-episodes", str(num_episodes),
        "--seed", str(seed),
        "--dataset-root", dataset_root,
        "--output-dir", output_dir,
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    sys.stdout.flush()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        raise RuntimeError(f"Episode selection failed with exit code {result.returncode}")

    subset_file = Path(output_dir) / f"random_{num_episodes}_seed{seed}.json"
    if not subset_file.exists():
        raise FileNotFoundError(f"Subset file not generated: {subset_file}")

    # Load the selected episodes
    with open(subset_file, "r") as f:
        subset_data = json.load(f)
    
    selected = subset_data["selected_episode_indices"]

    stage_time = time.time() - stage_start
    print(f"\nRandom selection complete in {stage_time:.1f}s")
    print(f"Selected {len(selected)} episodes")
    print(f"Selected: {selected[:10]}{'...' if len(selected) > 10 else ''}")
    sys.stdout.flush()

    return selected


def select_grid_uniform(
    dataset_root: str,
    num_episodes: int,
    seed: int,
    output_dir: str,
) -> list[int]:
    """
    Select episodes using grid_uniform algorithm with factorized rand_vec-group joint-region uniform distribution.
    
    Calls: personal/work2/duibi/train_and_eval_scripts/select_grid_uniform.py
    """
    print(f"\n{'='*60}")
    print(f"Grid Uniform Episode Selection")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_root}")
    print(f"Num episodes: {num_episodes}")
    print(f"Seed: {seed}")
    sys.stdout.flush()

    stage_start = time.time()

    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    WORK2_ROOT = Path(__file__).resolve().parents[3]

    select_script = WORK2_ROOT / "duibi" / "train_and_eval_scripts" / "select_grid_uniform.py"
    if not select_script.exists():
        raise FileNotFoundError(f"Selection script not found: {select_script}")

    cmd = [
        sys.executable, str(select_script),
        "--num-episodes", str(num_episodes),
        "--seed", str(seed),
        "--dataset-root", dataset_root,
        "--output-dir", output_dir,
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    sys.stdout.flush()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        raise RuntimeError(f"Episode selection failed with exit code {result.returncode}")

    subset_file = Path(output_dir) / f"grid_uniform_{num_episodes}_seed{seed}.json"
    if not subset_file.exists():
        raise FileNotFoundError(f"Subset file not generated: {subset_file}")

    # Load the selected episodes
    with open(subset_file, "r") as f:
        subset_data = json.load(f)
    
    selected = subset_data["selected_episode_indices"]

    stage_time = time.time() - stage_start
    print(f"\nGrid uniform selection complete in {stage_time:.1f}s")
    print(f"Selected {len(selected)} episodes")
    print(f"Selected: {selected[:10]}{'...' if len(selected) > 10 else ''}")
    sys.stdout.flush()

    return selected


def select_our_v5(
    dataset_root: str,
    num_episodes: int,
    seed: int,
    output_dir: str,
    gpu_id: int = 0,
) -> list[int]:
    """
    Select episodes using our_v5 adaptive coverage algorithm.
    
    This method calls the external select_our_v5.py script which requires:
    - V5 visual embeddings (phi_global + phi_wrist)
    - Action descriptors
    - episode_initial_states.json with rand_vec
    
    Returns the selected episode indices.
    """
    print(f"\n{'='*60}")
    print(f"Our V5 Episode Selection")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_root}")
    print(f"Num episodes: {num_episodes}")
    print(f"Seed: {seed}")
    sys.stdout.flush()

    stage_start = time.time()

    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    WORK2_ROOT = Path(__file__).resolve().parents[3]

    select_script = WORK2_ROOT / "our_v5" / "select_our_v5.py"
    if not select_script.exists():
        raise FileNotFoundError(f"Selection script not found: {select_script}")

    dataset_dir = Path(dataset_root)
    v5_output_dir = Path(output_dir).parent / f"our_v5_{num_episodes}_seed{seed}"

    # Auto-extract V5 embeddings if they don't exist
    print(f"\nEnsuring V5 embeddings exist...")
    from embedding_utils.ensure_embeddings_v5 import ensure_v5_embeddings_exist as _ensure_v5
    from embedding_utils.config import DEFAULT_PCA_DIM

    visual_dir, action_dir = _ensure_v5(
        dataset_root=dataset_root,
        dataset_name=dataset_dir.name,
        gpu_id=gpu_id,
        pca_dim=DEFAULT_PCA_DIM,
    )
    visual_embedding_dir = str(visual_dir)
    action_descriptor_dir = str(action_dir)
    print(f"Visual embeddings: {visual_embedding_dir}")
    print(f"Action descriptors: {action_descriptor_dir}")
    sys.stdout.flush()

    if not (dataset_dir / "episode_initial_states.json").exists():
        raise FileNotFoundError(
            f"Dataset directory missing episode_initial_states.json: {dataset_dir}\n"
            f"our_v5 requires rand_vec for each episode."
        )

    cmd = [
        sys.executable, str(select_script),
        "--visual-embedding-dir", str(visual_embedding_dir),
        "--action-descriptor-dir", str(action_descriptor_dir),
        "--dataset-dir", str(dataset_dir),
        "--output-dir", str(v5_output_dir),
        "--num-selected", str(num_episodes),
        "--seed", str(seed),
        "--visual-weight", "0.5",
        "--action-weight", "0.5",
        "--region-ratio", "0.1",
        "--min-regions", "16",
        "--max-regions", "128",
        "--b0-region-ratio", "0.2",
        "--coverage-weight", "0.5",
        "--region-visual-weight", "0.3",
        "--region-action-weight", "0.2",
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    sys.stdout.flush()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        raise RuntimeError(f"Episode selection failed with exit code {result.returncode}")

    v5_subset = v5_output_dir / "subsets" / f"our_v5_{num_episodes}_seed{seed}.json"
    if not v5_subset.exists():
        raise FileNotFoundError(f"Subset file not generated: {v5_subset}")

    # Load the selected episodes
    with open(v5_subset, "r") as f:
        subset_data = json.load(f)
    
    selected = subset_data["selected_episodes"]

    stage_time = time.time() - stage_start
    print(f"\nOur V5 selection complete in {stage_time:.1f}s")
    print(f"Selected {len(selected)} episodes")
    print(f"Selected: {selected[:10]}{'...' if len(selected) > 10 else ''}")
    sys.stdout.flush()

    return selected


def select_deminf(
    dataset_root: str,
    num_episodes: int,
    seed: int,
    output_dir: str,
) -> list[int]:
    """
    Select episodes using DemInf mutual information algorithm.
    
    This method calls the external run_deminf.py script which:
    1. Trains state and action VAEs
    2. Encodes all timesteps
    3. Scores episodes by mutual information
    4. Selects top-K episodes
    
    Returns the selected episode indices.
    """
    print(f"\n{'='*60}")
    print(f"DemInf Episode Selection")
    print(f"{'='*60}")
    print(f"Dataset: {dataset_root}")
    print(f"Num episodes: {num_episodes}")
    print(f"Seed: {seed}")
    sys.stdout.flush()

    stage_start = time.time()

    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    WORK2_ROOT = Path(__file__).resolve().parents[3]

    select_script = WORK2_ROOT / "deminf" / "run_deminf.py"
    if not select_script.exists():
        raise FileNotFoundError(f"Selection script not found: {select_script}")

    deminf_output_dir = Path(output_dir).parent / f"deminf_{num_episodes}_seed{seed}"

    cmd = [
        sys.executable, str(select_script),
        "--dataset-path", dataset_root,
        "--output-dir", str(deminf_output_dir),
        "--target-episodes", str(num_episodes),
        "--seed", str(seed),
        "--vae-steps", "50000",
        "--vae-lr", "1e-4",
        "--vae-batch-size", "256",
        "--state-latent-dim", "12",
        "--action-latent-dim", "6",
        "--ks", "5", "6", "7",
        "--state-source", "observation.environment_state",
        "--quality-batch-size", "1024",
        "--quality-repeat", "4",
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    sys.stdout.flush()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        raise RuntimeError(f"Episode selection failed with exit code {result.returncode}")

    deminf_subset = deminf_output_dir / "subsets" / f"deminf_{num_episodes}_seed{seed}.json"
    if not deminf_subset.exists():
        raise FileNotFoundError(f"Subset file not generated: {deminf_subset}")

    # Load the selected episodes
    with open(deminf_subset, "r") as f:
        subset_data = json.load(f)
    
    selected = subset_data["selected_episodes"]

    stage_time = time.time() - stage_start
    print(f"\nDemInf selection complete in {stage_time:.1f}s")
    print(f"Selected {len(selected)} episodes")
    print(f"Selected: {selected[:10]}{'...' if len(selected) > 10 else ''}")
    sys.stdout.flush()

    return selected


METHODS = {
    "random": select_random,
    "grid_uniform": select_grid_uniform,
    "our_v5": select_our_v5,
    "deminf": select_deminf,
}


def main():
    parser = argparse.ArgumentParser(description="Select episodes from LeRobot dataset")
    parser.add_argument("--method", type=str, default="random", choices=METHODS.keys())
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID for our_v5 embedding extraction")
    args = parser.parse_args()

    # All methods now call external scripts uniformly
    if args.method == "random":
        selected = select_random(
            dataset_root=args.dataset_root,
            num_episodes=args.num_episodes,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    elif args.method == "grid_uniform":
        selected = select_grid_uniform(
            dataset_root=args.dataset_root,
            num_episodes=args.num_episodes,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    elif args.method == "our_v5":
        selected = select_our_v5(
            dataset_root=args.dataset_root,
            num_episodes=args.num_episodes,
            seed=args.seed,
            output_dir=args.output_dir,
            gpu_id=args.gpu_id,
        )
    elif args.method == "deminf":
        selected = select_deminf(
            dataset_root=args.dataset_root,
            num_episodes=args.num_episodes,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    total_episodes = load_episode_count(args.dataset_root)

    # Save to output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "selected_episodes.json"
    with open(output_file, "w") as f:
        json.dump({
            "method": args.method,
            "seed": args.seed,
            "total_episodes": total_episodes,
            "selected_episodes": selected,
            "num_selected": len(selected),
        }, f, indent=2)

    print(f"\nSaved {len(selected)} episode indices to {output_file}")
    return str(output_file)


if __name__ == "__main__":
    main()