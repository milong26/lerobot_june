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
import time
import numpy as np
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from differentvlm.configs.vlm_config import VLMExperimentConfig


def extract_action_descriptors(dataset_root: str, output_dir: str) -> str:
    """
    Extract real action descriptors from LeRobot dataset.
    
    Reads action sequences from each episode, resamples to fixed length,
    computes statistics, and saves as .npy files for V5 selection.
    
    Args:
        dataset_root: path to LeRobotDataset root directory
        output_dir: output directory for action descriptor .npy files
    
    Returns:
        output_dir path string
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Check if action descriptors already exist
    existing_files = list(output_path.glob("(*).npy"))
    if existing_files:
        print(f"[CACHE HIT] Found {len(existing_files)} existing action descriptors")
        print(f"  Directory: {output_path}")
        print(f"  Reusing cached action descriptors. Skipping extraction.")
        sys.stdout.flush()
        return str(output_path)
    
    print(f"\n{'='*60}")
    print(f"Extracting Action Descriptors from Dataset")
    print(f"{'='*60}")
    print(f"Dataset root: {dataset_root}")
    print(f"Output dir: {output_path}")
    sys.stdout.flush()
    
    # Load dataset
    start_time = time.time()
    print(f"Loading dataset...")
    dataset = LeRobotDataset(
        repo_id="work2/metaworld",
        root=dataset_root
    )
    num_episodes = len(dataset.meta.episodes)
    print(f"Dataset loaded: {num_episodes} episodes")
    sys.stdout.flush()
    
    # Find action key
    features = dataset.meta.features
    action_key = None
    for key in features.keys():
        if "action" in key.lower():
            action_key = key
            break
    if action_key is None:
        raise ValueError(f"No action feature found in dataset. Available: {list(features.keys())}")
    print(f"Action key: {action_key}")
    sys.stdout.flush()
    
    # Extract action descriptors for all episodes
    V5_ACTION_STEPS = 16
    extracted_count = 0
    
    for ep_idx in range(num_episodes):
        ep_start = time.time()
        
        # Get episode frame range
        from_idx = dataset.meta.episodes["dataset_from_index"][ep_idx]
        to_idx = dataset.meta.episodes["dataset_to_index"][ep_idx]
        
        # Extract action sequence
        actions = []
        for frame_idx in range(from_idx, to_idx):
            frame = dataset[frame_idx]
            actions.append(frame[action_key])
        actions = np.array(actions)
        
        # Resample to fixed length
        T = actions.shape[0]
        if T != V5_ACTION_STEPS:
            original_indices = np.linspace(0, T - 1, T)
            new_indices = np.linspace(0, T - 1, V5_ACTION_STEPS)
            resampled_actions = np.array([
                np.interp(new_indices, original_indices, actions[:, d])
                for d in range(actions.shape[1])
            ]).T
        else:
            resampled_actions = actions
        
        # Compute action statistics
        mean_action = np.mean(actions, axis=0)
        std_action = np.std(actions, axis=0)
        delta_actions = np.diff(actions, axis=0)
        delta_action_mean = np.mean(delta_actions, axis=0)
        delta_action_std = np.std(delta_actions, axis=0)
        velocity_mean = np.mean(np.abs(delta_actions), axis=0)
        trajectory_length = np.array([float(np.sum(np.linalg.norm(delta_actions, axis=1)))])
        
        # Concatenate into descriptor
        descriptor = np.concatenate([
            resampled_actions.flatten(),
            mean_action,
            std_action,
            delta_action_mean,
            delta_action_std,
            velocity_mean,
            trajectory_length,
        ]).astype(np.float32)
        
        # Save to .npy file
        npy_file = output_path / f"({ep_idx}).npy"
        np.save(npy_file, {
            "episode_index": ep_idx,
            "action_descriptor": descriptor,
        }, allow_pickle=True)
        
        extracted_count += 1
        elapsed = time.time() - ep_start
        
        if extracted_count % 50 == 0 or extracted_count == num_episodes:
            progress = extracted_count / num_episodes * 100
            print(f"  Extracted {extracted_count}/{num_episodes} episodes ({progress:.1f}%), "
                  f"descriptor shape={descriptor.shape}, last episode time={elapsed:.2f}s")
            sys.stdout.flush()
    
    total_time = time.time() - start_time
    print(f"\nAction descriptor extraction complete: {extracted_count} episodes in {total_time:.1f}s")
    print(f"  Output directory: {output_path}")
    sys.stdout.flush()
    
    return str(output_path)


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

    # Extract real action descriptors from dataset (not dummy zeros)
    action_descriptor_dir = Path(embedding_dir) / "action_descriptors"
    action_descriptor_dir = extract_action_descriptors(cfg.dataset_root, str(action_descriptor_dir))

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
    v5_script = Path(__file__).resolve().parents[2] / "our_v5" / "select_our_v5.py"
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