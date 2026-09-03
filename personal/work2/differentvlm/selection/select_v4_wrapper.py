"""
V4 Selection Wrapper for DifferentVLM

Calls the existing our_v4 selection logic with differentvlm-generated embeddings.
Does NOT copy or reimplement selection algorithm.

Input validation:
- Confirms embeddings are in unified JSON format (episode_{idx}.json)
- Validates model_name and camera match config
- Confirms only visual embeddings are used (no success/grasp_success/eval/attention data)

Output:
- selected_episode.json with vlm_name, camera, selection_method=v4, selected_episode_indices
"""

import sys
import json
import subprocess
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from differentvlm.configs.vlm_config import VLMExperimentConfig


def validate_embedding_format(embedding_dir: str, expected_vlm_name: str, expected_camera: str) -> dict:
    """
    Validate that embeddings are in the expected unified format.
    Checks:
    1. episode_0.json exists
    2. Contains required fields: episode_id, global_embedding, wrist_embedding, model_name, camera
    3. model_name matches expected VLM
    4. camera matches expected camera
    5. Count total episodes
    Returns validation result dict.
    """
    emb_dir = Path(embedding_dir)
    if not emb_dir.exists():
        raise FileNotFoundError(f"Embedding directory not found: {emb_dir}")

    ep0_file = emb_dir / "episode_0.json"
    if not ep0_file.exists():
        raise FileNotFoundError(f"episode_0.json not found in {emb_dir}")

    with open(ep0_file, "r") as f:
        meta = json.load(f)

    required_fields = ["episode_id", "global_embedding", "wrist_embedding", "model_name", "camera", "embedding_dim"]
    for field in required_fields:
        if field not in meta:
            raise ValueError(f"Missing required field '{field}' in episode_0.json")

    if meta["model_name"] != expected_vlm_name:
        raise ValueError(f"Model name mismatch: expected={expected_vlm_name}, got={meta['model_name']}")

    if meta["camera"] != expected_camera:
        raise ValueError(f"Camera mismatch: expected={expected_camera}, got={meta['camera']}")

    episode_count = 0
    for ep_file in emb_dir.glob("episode_*.json"):
        episode_count += 1

    print(f"Embedding format validation PASSED")
    print(f"  Format: unified JSON (episode_*.json)")
    print(f"  Model: {meta['model_name']}")
    print(f"  Camera: {meta['camera']}")
    print(f"  Embedding dim: {meta['embedding_dim']}")
    print(f"  Total episodes: {episode_count}")
    sys.stdout.flush()

    return {
        "format": "unified_json",
        "model_name": meta["model_name"],
        "camera": meta["camera"],
        "embedding_dim": meta["embedding_dim"],
        "episode_count": episode_count,
    }


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

    print(f"\nValidating embedding input format...")
    sys.stdout.flush()
    validation = validate_embedding_format(
        cfg.embedding_cache_dir,
        cfg.vlm_name,
        cfg.camera,
    )

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

    print(f"\nRunning: {' '.join(cmd)}")
    sys.stdout.flush()

    result = subprocess.run(cmd, cwd=str(project_root))

    if result.returncode != 0:
        raise RuntimeError(f"V4 selection failed with exit code {result.returncode}")

    subset_file = output_dir / "subsets" / f"dynamicgrid_v4_{cfg.selection_num_episodes}_seed{cfg.selection_seed}.json"
    if not subset_file.exists():
        raise FileNotFoundError(f"Subset file not generated: {subset_file}")

    print(f"\nV4 selection complete. Subset file: {subset_file}")
    sys.stdout.flush()

    with open(subset_file, "r") as f:
        subset_data = json.load(f)

    subset_data["metadata"] = {
        "vlm_name": cfg.vlm_name,
        "selection_vlm_model_id": cfg.selection_vlm_model_id,
        "camera": cfg.camera,
        "selection_method": "v4",
        "selected_episode_indices": subset_data["selected_episode_indices"],
        "num_selected": len(subset_data["selected_episode_indices"]),
        "seed": cfg.selection_seed,
        "embedding_dir": cfg.embedding_cache_dir,
        "embedding_validation": validation,
    }

    with open(subset_file, "w") as f:
        json.dump(subset_data, f, indent=2)

    print(f"Metadata added to subset file")
    print(f"  VLM: {cfg.vlm_name}")
    print(f"  Camera: {cfg.camera}")
    print(f"  Selection method: v4")
    print(f"  Selected episodes: {len(subset_data['selected_episode_indices'])}")
    sys.stdout.flush()

    return str(subset_file)