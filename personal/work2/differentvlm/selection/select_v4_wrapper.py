"""
V4 Selection Wrapper for DifferentVLM

Calls the existing our_v4 selection logic with differentvlm-generated embeddings.
Does NOT copy or reimplement selection algorithm.
"""

import sys
import json
import subprocess
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from differentvlm.configs.vlm_config import VLMExperimentConfig


def run_v4_selection(cfg: VLMExperimentConfig) -> str:
    """
    Run v4 selection using existing our_v4 logic.
    Flow: VLM embedding read -> visual coverage calculation -> v4 episode selection -> save selected_episode.json
    Returns the subset file path.
    """
    print(f"\n{'='*60}")
    print(f"Running V4 Episode Selection")
    print(f"{'='*60}")
    print(f"Embedding dir: {cfg.embedding_cache_dir}")
    print(f"Dataset root: {cfg.dataset_root}")
    print(f"Output dir: {cfg.results_dir}")
    print(f"Num episodes: {cfg.selection_num_episodes}")
    print(f"Seed: {cfg.selection_seed}")
    print(f"{'='*60}")

    output_dir = Path(cfg.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "subsets").mkdir(parents=True, exist_ok=True)
    (output_dir / "results").mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[4]
    our_v4_root = project_root / "personal" / "work2" / "our_v4"
    select_script = our_v4_root / "experiments" / "select_episodes_v4.py"

    if not select_script.exists():
        raise FileNotFoundError(f"V4 selection script not found: {select_script}")

    cmd = [
        sys.executable, str(select_script),
        "--dataset-root", cfg.dataset_root,
        "--embedding-dir", cfg.embedding_cache_dir,
        "--output-dir", str(cfg.results_dir),
        "--num-selected", str(cfg.selection_num_episodes),
        "--seed", str(cfg.selection_seed),
    ]

    print(f"Running: {' '.join(cmd)}")
    sys.stdout.flush()

    result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[4]))

    if result.returncode != 0:
        raise RuntimeError(f"V4 selection failed with exit code {result.returncode}")

    subset_file = output_dir / "subsets" / f"dynamicgrid_v4_{cfg.selection_num_episodes}_seed{cfg.selection_seed}.json"
    if not subset_file.exists():
        raise FileNotFoundError(f"Subset file not generated: {subset_file}")

    print(f"V4 selection complete. Subset file: {subset_file}")
    sys.stdout.flush()
    return str(subset_file)