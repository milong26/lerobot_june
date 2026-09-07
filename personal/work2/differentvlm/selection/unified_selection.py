"""
Unified Selection Module for DifferentVLM

Supports multiple episode selection methods:
- v5: VLM embedding + action descriptor based selection (default)
- grid_uniform: Factorized rand_vec-group uniform distribution
- random: Random episode selection

This module provides a unified interface for all selection methods.
"""

import sys
import json
import subprocess
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from differentvlm.configs.vlm_config import VLMExperimentConfig


def run_selection(cfg: VLMExperimentConfig, embedding_dir: str = None) -> str:
    """
    Run episode selection using the specified method.
    
    Args:
        cfg: experiment configuration
        embedding_dir: VLM embedding directory (only used for v5 mode)
    
    Returns:
        Path to the subset JSON file
    """
    selection_mode = cfg.selection_mode
    
    if selection_mode == "v5":
        if embedding_dir is None:
            raise ValueError("embedding_dir is required for v5 selection mode")
        from differentvlm.selection.select_v5_wrapper import run_v5_selection
        return run_v5_selection(cfg, embedding_dir)
    elif selection_mode == "grid_uniform":
        return run_grid_uniform_selection(cfg)
    elif selection_mode == "random":
        return run_random_selection(cfg)
    else:
        raise ValueError(f"Unknown selection mode: {selection_mode}. Supported: v5, grid_uniform, random")


def run_grid_uniform_selection(cfg: VLMExperimentConfig) -> str:
    """
    Run grid_uniform episode selection.
    Uses the duibi/select_grid_uniform.py script.
    """
    print(f"\n{'='*60}")
    print(f"Running Grid Uniform Episode Selection")
    print(f"{'='*60}")
    print(f"Dataset: {cfg.dataset_name}")
    print(f"Num episodes: {cfg.selection_num_episodes}")
    print(f"Seed: {cfg.selection_seed}")
    sys.stdout.flush()

    subset_file = Path(cfg.results_dir) / f"grid_uniform_{cfg.selection_num_episodes}_seed{cfg.selection_seed}.json"

    if subset_file.exists():
        print(f"\n[CACHE HIT] Subset file already exists: {subset_file}")
        sys.stdout.flush()
        return str(subset_file)

    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    select_script = PROJECT_ROOT / "personal" / "work2" / "duibi" / "train_and_eval_scripts" / "select_grid_uniform.py"

    if not select_script.exists():
        raise FileNotFoundError(f"Selection script not found: {select_script}")

    cmd = [
        sys.executable, str(select_script),
        "--num-episodes", str(cfg.selection_num_episodes),
        "--seed", str(cfg.selection_seed),
        "--dataset-root", cfg.dataset_root,
        "--output-dir", cfg.results_dir,
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    sys.stdout.flush()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        raise RuntimeError(f"Grid uniform selection failed with exit code {result.returncode}")

    if not subset_file.exists():
        raise FileNotFoundError(f"Subset file not generated: {subset_file}")

    print(f"\nSelection complete. Subset file: {subset_file}")
    sys.stdout.flush()
    return str(subset_file)


def run_random_selection(cfg: VLMExperimentConfig) -> str:
    """
    Run random episode selection.
    Uses the duibi/select_random_episodes.py script.
    """
    print(f"\n{'='*60}")
    print(f"Running Random Episode Selection")
    print(f"{'='*60}")
    print(f"Dataset: {cfg.dataset_name}")
    print(f"Num episodes: {cfg.selection_num_episodes}")
    print(f"Seed: {cfg.selection_seed}")
    sys.stdout.flush()

    subset_file = Path(cfg.results_dir) / f"random_{cfg.selection_num_episodes}_seed{cfg.selection_seed}.json"

    if subset_file.exists():
        print(f"\n[CACHE HIT] Subset file already exists: {subset_file}")
        sys.stdout.flush()
        return str(subset_file)

    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    select_script = PROJECT_ROOT / "personal" / "work2" / "duibi" / "train_and_eval_scripts" / "select_random_episodes.py"

    if not select_script.exists():
        raise FileNotFoundError(f"Selection script not found: {select_script}")

    cmd = [
        sys.executable, str(select_script),
        "--num-episodes", str(cfg.selection_num_episodes),
        "--seed", str(cfg.selection_seed),
        "--dataset-root", cfg.dataset_root,
        "--output-dir", cfg.results_dir,
    ]

    print(f"\nRunning: {' '.join(cmd)}")
    sys.stdout.flush()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        raise RuntimeError(f"Random selection failed with exit code {result.returncode}")

    if not subset_file.exists():
        raise FileNotFoundError(f"Subset file not generated: {subset_file}")

    print(f"\nSelection complete. Subset file: {subset_file}")
    sys.stdout.flush()
    return str(subset_file)