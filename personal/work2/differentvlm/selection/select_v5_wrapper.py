"""
V5 Selection Wrapper for DifferentVLM

Calls the existing our_v5 selection logic with differentvlm-generated embeddings.
Does NOT copy or reimplement selection algorithm.

Input validation:
- Confirms embeddings are in unified JSON format (episode_{idx}.json)
- Validates model_name and camera match config
- Converts JSON embeddings to .npy format for our_v5

Output:
- selected_episode.json with vlm_name, camera, selection_method=v5, selected_episode_indices
"""

import sys
import json
import subprocess
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from differentvlm.configs.vlm_config import VLMExperimentConfig


def convert_embeddings_to_npy(embedding_dir: str, output_dir: str) -> str:
    """
    Convert differentvlm unified JSON embeddings to our_v5 .npy format.
    
    Input:  episode_{idx}.json with {episode_id, global_embedding, wrist_embedding, ...}
    Output: ({idx}).npy with {phi_global, phi_wrist, episode_index}
    
    Returns the output directory path.
    """
    emb_dir = Path(embedding_dir)
    npy_dir = Path(output_dir)
    npy_dir.mkdir(parents=True, exist_ok=True)
    
    converted_count = 0
    for json_file in sorted(emb_dir.glob("episode_*.json")):
        with open(json_file, "r") as f:
            data = json.load(f)
        
        ep_idx = data["episode_id"]
        phi_global = np.array(data["global_embedding"], dtype=np.float32)
        phi_wrist = np.array(data["wrist_embedding"], dtype=np.float32)
        
        # our_v5 expects: episode_index, phi_global, phi_wrist
        npy_data = {
            "episode_index": ep_idx,
            "phi_global": phi_global,
            "phi_wrist": phi_wrist,
        }
        npy_file = npy_dir / f"({ep_idx}).npy"
        np.save(npy_file, npy_data)
        converted_count += 1
    
    print(f"Converted {converted_count} embeddings from JSON to .npy format")
    print(f"  Source: {emb_dir}")
    print(f"  Target: {npy_dir}")
    sys.stdout.flush()
    
    return str(npy_dir)


def validate_embedding_format(embedding_dir: str, expected_vlm_name: str, expected_camera: str) -> dict:
    """
    Validate that embeddings are in the expected unified format.
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
    print(f"  VLM: {meta['model_name']}")
    print(f"  Camera: {meta['camera']}")
    print(f"  Episodes: {episode_count}")
    sys.stdout.flush()

    return {"valid": True, "episodes": episode_count, "model_name": meta["model_name"], "camera": meta["camera"]}


def run_v5_selection(cfg: VLMExperimentConfig, embedding_dir: str) -> str:
    """
    Run V5 episode selection with differentvlm-generated embeddings.
    Reuses existing our_v5 selection entry point.
    Returns the subset file path.
    """
    print(f"\n{'='*60}")
    print(f"Running V5 Episode Selection")
    print(f"{'='*60}")

    # Validate embedding format
    validation = validate_embedding_format(
        embedding_dir,
        expected_vlm_name=cfg.vlm_name,
        expected_camera=cfg.camera,
    )

    print(f"Selected episodes: {cfg.selection_num_episodes}")
    print(f"Selection seed: {cfg.selection_seed}")
    print(f"{'='*60}")
    sys.stdout.flush()

    # Convert embeddings to .npy format for our_v5
    npy_dir = Path(embedding_dir) / "npy_for_selection"
    npy_dir = convert_embeddings_to_npy(embedding_dir, str(npy_dir))

    # our_v5 also requires action descriptors
    # For now, we generate dummy action descriptors (zeros) since differentvlm
    # doesn't compute action descriptors. This allows our_v5 to run but with
    # action_weight effectively having no effect.
    action_descriptor_dir = Path(embedding_dir) / "action_descriptors"
    action_descriptor_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate dummy action descriptors for all episodes
    episode_count = validation["episodes"]
    for ep_idx in range(episode_count):
        # Dummy 64-dim action descriptor (zeros)
        dummy_descriptor = np.zeros(64, dtype=np.float32)
        npy_data = {
            "episode_index": ep_idx,
            "action_descriptor": dummy_descriptor,
        }
        npy_file = action_descriptor_dir / f"({ep_idx}).npy"
        np.save(npy_file, npy_data)
    
    print(f"Generated {episode_count} dummy action descriptors (zeros)")
    print(f"  Note: differentvlm doesn't compute action descriptors, so action_weight has no effect")
    print(f"  Directory: {action_descriptor_dir}")
    sys.stdout.flush()

    # Get dataset directory for rand_vec loading
    dataset_dir = Path(cfg.dataset_root)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not (dataset_dir / "episode_initial_states.json").exists():
        raise FileNotFoundError(
            f"Dataset directory missing episode_initial_states.json: {dataset_dir}\n"
            f"Please ensure the dataset contains episode_initial_states.json with rand_vec for each episode."
        )

    # Call our_v5 selection script
    v5_script = Path(__file__).resolve().parents[3] / "our_v5" / "select_our_v5.py"
    if not v5_script.exists():
        raise FileNotFoundError(f"our_v5 selection script not found: {v5_script}")

    subset_file = Path(cfg.results_dir) / "selected_episode.json"
    output_dir = Path(cfg.results_dir) / "v5_selection_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", str(v5_script),
        "--visual-embedding-dir", str(npy_dir),
        "--action-descriptor-dir", str(action_descriptor_dir),
        "--dataset-dir", str(dataset_dir),
        "--output-dir", str(output_dir),
        "--num-selected", str(cfg.selection_num_episodes),
        "--seed", str(cfg.selection_seed),
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

    print(f"\nRunning our_v5 selection: {v5_script}")
    print(f"Output: {subset_file}")
    sys.stdout.flush()

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).resolve().parents[4]),
    )

    if result.returncode != 0:
        print(f"WARNING: Selection exited with code {result.returncode}")
        print(result.stdout.decode())
        sys.stdout.flush()

    # our_v5 saves to output_dir/subsets/our_v5_{num}_seed{seed}.json
    # We need to copy/rename it to the expected subset_file location
    subset_candidates = list((output_dir / "subsets").glob(f"our_v5_*.json"))
    if subset_candidates:
        import shutil
        subset_source = subset_candidates[0]
        shutil.copy(subset_source, subset_file)
        print(f"\nSelection complete.")
        print(f"Subset file: {subset_file}")
    else:
        raise FileNotFoundError(f"Selection did not produce output file in {output_dir / 'subsets'}")

    sys.stdout.flush()
    return str(subset_file)