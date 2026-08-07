#!/usr/bin/env python
"""
SCRIPT 1: Generate a large candidate pool and apply selection strategies.

Step 1: Generate large candidate pool (500 demos with grid + random positions)
Step 2: Extract frozen VLM embeddings for all demos
Step 3: Apply each selection strategy to choose subsets from the pool
Step 4: Print summary
"""

import os
import sys
import json
import time
import shutil
import numpy as np
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from personal.work5.config import CFG
from personal.work5.data_collection.metaworld_collector import (
    collect_demo, generate_grid_positions, generate_random_positions
)
from personal.work5.data_collection.lerobot_writer import (
    create_dataset, add_episode_to_dataset, finalize_dataset
)
from personal.work5.data_collection.strategies import (
    k_center_select, fps_select, sic_noise_select
)
from personal.work5.sic.embeddings import (
    load_frozen_vlm, extract_all_episode_embeddings,
    extract_episode_embeddings, get_episode_mean_embedding
)
from personal.work5.sic.anchor import build_anchor_reference, compute_sic_score


def step1_generate_candidate_pool(n_total=500):
    """Generate large candidate pool with both grid and random positions."""
    print("\n" + "=" * 80)
    print(f"STEP 1: Generating large candidate pool ({n_total} demos)")
    print("=" * 80)

    # Generate grid positions (B0) + random positions
    grid_positions = generate_grid_positions(CFG.grid_n_per_axis, CFG.task_name)
    n_random = n_total - len(grid_positions)
    random_positions = generate_random_positions(n_random, CFG.task_name, seed=123)

    all_positions = grid_positions + random_positions
    print(f"Grid positions: {len(grid_positions)}, Random positions: {len(random_positions)}")
    print(f"Total positions: {len(all_positions)}")

    all_frames = []
    all_infos = []
    all_positions_list = []
    b0_indices = list(range(len(grid_positions)))  # First 25 are B0

    for idx, (obj_pos, goal_pos) in enumerate(all_positions):
        print(f"  Collecting demo {idx + 1}/{len(all_positions)}...")
        frames, info = collect_demo(
            obj_pos, goal_pos, CFG.task_name, CFG.max_steps, CFG.image_size, seed=42 + idx
        )
        if frames is not None and info is not None:
            all_frames.append(frames)
            all_infos.append(info)
            all_positions_list.append((obj_pos, goal_pos))
            print(f"    Success: {info['success']}, Frames: {info['num_frames']}")
        else:
            print(f"    Failed or rejected")

    print(f"\nCandidate pool collected: {len(all_frames)} successful demos")

    # Save candidate pool
    output_dir = CFG.datasets_dir / 'candidate_pool'
    if output_dir.exists():
        shutil.rmtree(output_dir)

    dataset = create_dataset("work5/candidate_pool", str(output_dir), CFG.fps, CFG.image_size)
    for frames in all_frames:
        add_episode_to_dataset(dataset, frames, task=CFG.task_name)

    finalize_dataset(dataset, output_dir, all_infos, CFG.task_name, 'candidate_pool')
    print(f"  Saved candidate pool with {len(all_frames)} episodes")

    return all_frames, all_infos, all_positions_list, b0_indices


def step2_extract_embeddings(all_frames, all_infos, b0_indices):
    """Extract frozen VLM embeddings for all demos in candidate pool."""
    print("\n" + "=" * 80)
    print("STEP 2: Extracting frozen VLM embeddings for all demos")
    print("=" * 80)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    pool_dataset = LeRobotDataset("work5/candidate_pool", root=str(CFG.datasets_dir / 'candidate_pool'))
    model, processor = load_frozen_vlm(CFG.vlm_model_id, CFG.device)

    all_embeddings = extract_all_episode_embeddings(
        model, processor, pool_dataset, CFG.global_cam_key, CFG.device,
        cache_path=str(CFG.work_dir / "cache" / "pool_embeddings.pkl")
    )

    all_mean_embeddings = {}
    for ep_idx, ep_embs in all_embeddings.items():
        all_mean_embeddings[ep_idx] = get_episode_mean_embedding(ep_embs)

    all_mean_array = np.array([all_mean_embeddings[k] for k in sorted(all_mean_embeddings.keys())])
    print(f"All mean embeddings shape: {all_mean_array.shape}")

    # Extract B0 embeddings (first 25 demos)
    b0_mean_array = all_mean_array[b0_indices]
    print(f"B0 mean embeddings shape: {b0_mean_array.shape}")

    return model, processor, all_mean_array, b0_mean_array


def step3_apply_strategies(model, processor, all_mean_array, b0_mean_array,
                           all_frames, all_infos, b0_indices):
    """Apply each selection strategy to choose subsets from the candidate pool."""
    print("\n" + "=" * 80)
    print("STEP 3: Applying selection strategies")
    print("=" * 80)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # Get B0 frames and infos
    b0_frames = [all_frames[i] for i in b0_indices]
    b0_infos = [all_infos[i] for i in b0_indices]

    # Get candidate pool (excluding B0)
    candidate_indices = [i for i in range(len(all_frames)) if i not in b0_indices]
    candidate_frames = [all_frames[i] for i in candidate_indices]
    candidate_infos = [all_infos[i] for i in candidate_indices]
    candidate_mean_array = all_mean_array[candidate_indices]

    # Uniform-B0 strategy (baseline)
    print("\n--- Uniform-B0 Strategy (Baseline) ---")
    output_dir = CFG.datasets_dir / 'uniform_b0'
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dataset = create_dataset("work5/uniform_b0", str(output_dir), CFG.fps, CFG.image_size)
    for frames in b0_frames:
        add_episode_to_dataset(dataset, frames, task=CFG.task_name)
    finalize_dataset(dataset, output_dir, b0_infos, CFG.task_name, 'uniform_b0')
    print(f"  Uniform-B0 dataset: {len(b0_frames)} episodes")

    # K-Center strategy
    print("\n--- K-Center Strategy ---")
    n_select = CFG.total_budget - CFG.b0_demos  # 50
    kcenter_indices = k_center_select(b0_mean_array, candidate_mean_array, n_select)
    kcenter_frames = [candidate_frames[i] for i in kcenter_indices]
    kcenter_infos = [candidate_infos[i] for i in kcenter_indices]

    output_dir = CFG.datasets_dir / 'kcenter'
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dataset = create_dataset("work5/kcenter", str(output_dir), CFG.fps, CFG.image_size)
    for frames in b0_frames:
        add_episode_to_dataset(dataset, frames, task=CFG.task_name)
    for frames in kcenter_frames:
        add_episode_to_dataset(dataset, frames, task=CFG.task_name)
    all_kcenter_infos = b0_infos + kcenter_infos
    finalize_dataset(dataset, output_dir, all_kcenter_infos, CFG.task_name, 'kcenter')
    print(f"  K-Center dataset: {len(all_kcenter_infos)} episodes")

    # FPS strategy
    print("\n--- FPS Strategy ---")
    fps_indices = fps_select(b0_mean_array, candidate_mean_array, n_select)
    fps_frames = [candidate_frames[i] for i in fps_indices]
    fps_infos = [candidate_infos[i] for i in fps_indices]

    output_dir = CFG.datasets_dir / 'fps'
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dataset = create_dataset("work5/fps", str(output_dir), CFG.fps, CFG.image_size)
    for frames in b0_frames:
        add_episode_to_dataset(dataset, frames, task=CFG.task_name)
    for frames in fps_frames:
        add_episode_to_dataset(dataset, frames, task=CFG.task_name)
    all_fps_infos = b0_infos + fps_infos
    finalize_dataset(dataset, output_dir, all_fps_infos, CFG.task_name, 'fps')
    print(f"  FPS dataset: {len(all_fps_infos)} episodes")

    # SIC-Noise strategy
    print("\n--- SIC-Noise Strategy (Ours) ---")
    undercovered_indices, sic_scores = sic_noise_select(
        b0_mean_array, CFG.n_undercovered, CFG.n_noise_per_undercovered)

    print(f"Most under-covered B0 indices: {undercovered_indices}")

    # Generate noise-augmented demos
    noise_frames = []
    noise_infos = []

    for uc_idx in undercovered_indices:
        orig_obj = b0_infos[uc_idx]['obj_init_pos']
        orig_goal = b0_infos[uc_idx]['goal_pose']

        for n_idx in range(CFG.n_noise_per_undercovered):
            noise_obj = orig_obj.copy() + np.random.normal(0, CFG.noise_sigma_pos, 3)
            noise_obj[2] = orig_obj[2]

            frames, info = collect_demo(
                noise_obj, orig_goal, CFG.task_name, CFG.max_steps, CFG.image_size,
                seed=300 + uc_idx * 10 + n_idx
            )
            if frames is not None and info is not None:
                noise_frames.append(frames)
                noise_infos.append(info)

    output_dir = CFG.datasets_dir / 'sic_noise'
    if output_dir.exists():
        shutil.rmtree(output_dir)
    dataset = create_dataset("work5/sic_noise", str(output_dir), CFG.fps, CFG.image_size)
    for frames in b0_frames:
        add_episode_to_dataset(dataset, frames, task=CFG.task_name)
    for frames in noise_frames:
        add_episode_to_dataset(dataset, frames, task=CFG.task_name)
    all_sic_infos = b0_infos + noise_infos
    finalize_dataset(dataset, output_dir, all_sic_infos, CFG.task_name, 'sic_noise')
    print(f"SIC-Noise dataset: {len(all_sic_infos)} episodes")

    return sic_scores


def step4_print_summary(sic_scores, b0_mean_array):
    """Print summary of all datasets."""
    print("\n" + "=" * 80)
    print("STEP 4: Dataset Summary")
    print("=" * 80)

    from sklearn.decomposition import PCA

    strategies = ['uniform_b0', 'kcenter', 'fps', 'sic_noise', 'candidate_pool']
    for strategy in strategies:
        ds_path = CFG.datasets_dir / strategy
        if ds_path.exists():
            info_file = ds_path / "meta" / "info.json"
            if info_file.exists():
                with open(info_file) as f:
                    info = json.load(f)
                n_episodes = info.get('total_episodes', 'unknown')
                print(f"  {strategy}: {n_episodes} demos")

    pca = PCA(n_components=32)
    pca.fit(b0_mean_array)
    print(f"\nPCA Variance Explained (d={CFG.d_pca}): {pca.explained_variance_ratio_.sum()*100:.1f}%")

    print(f"\nSIC Scores for B0:")
    print(f"  Mean: {sic_scores.mean():.4f}")
    print(f"  Min: {sic_scores.min():.4f}")
    print(f"  Max: {sic_scores.max():.4f}")

    print("\nDatasets ready. Run: python personal/work5/run_experiments.py")


if __name__ == "__main__":
    CFG.setup()

    start_time = time.time()

    all_frames, all_infos, all_positions, b0_indices = step1_generate_candidate_pool(100)
    model, processor, all_mean_array, b0_mean_array = step2_extract_embeddings(
        all_frames, all_infos, b0_indices)
    sic_scores = step3_apply_strategies(
        model, processor, all_mean_array, b0_mean_array,
        all_frames, all_infos, b0_indices)
    step4_print_summary(sic_scores, b0_mean_array)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time/60:.1f} minutes")