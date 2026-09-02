"""
Shared embedding cache operations.

Handles cache directory resolution, validation, legacy discovery,
and migration to the shared cache system.
"""

import json
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import (
    SHARED_EMBEDDING_ROOT,
    MODEL_NAME,
    PROMPT_TEXT,
    TOKEN_POOLING,
    GLOBAL_FRAME_RULE,
    WRIST_START_RATIO,
    WRIST_END_RATIO,
    TEMPORAL_POOLING,
    DEFAULT_PCA_DIM,
    EXTRACTOR_VERSION,
    build_extraction_method_name,
)


def normalize_dataset_name(dataset_name: str) -> str:
    """
    Normalize dataset name to include pick_place_ prefix if not already present.
    
    Examples:
        corner -> pick_place_corner
        corner2 -> pick_place_corner2
        pick_place_corner -> pick_place_corner (unchanged)
    """
    if dataset_name.startswith("pick_place_"):
        return dataset_name
    return f"pick_place_{dataset_name}"


def get_shared_embedding_dir(dataset_name: str, pca_dim: int = DEFAULT_PCA_DIM) -> Path:
    """
    Get the canonical shared embedding directory for a dataset and extraction method.
    
    Returns unique path like:
    /data/.../shared_embeddings/pick_place_corner/smolvlm2-500m_last-hidden-tokenmean_global-first5_wrist-20to70_temporal-mean_pca32_v1/
    """
    normalized = normalize_dataset_name(dataset_name)
    method_name = build_extraction_method_name(pca_dim)
    return SHARED_EMBEDDING_ROOT / normalized / method_name


def get_dataset_episode_indices(dataset_root: str) -> List[int]:
    """
    Get the real episode indices for a dataset.
    
    Priority:
    1. Read from dataset_root/episode_initial_states.json
    2. Fall back to LeRobotDataset metadata
    
    Never assumes a fixed number of episodes.
    """
    meta_file = Path(dataset_root) / "episode_initial_states.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        episodes = meta.get("episodes", [])
        return [ep.get("episode_index", i) for i, ep in enumerate(episodes)]
    
    # Fall back to LeRobotDataset metadata
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset(
        repo_id="work2/metaworld_pick_place",
        root=str(dataset_root)
    )
    return list(range(dataset.num_episodes))


def build_expected_metadata(
    dataset_root: str,
    dataset_name: str,
    pca_dim: int,
    episode_indices: List[int],
    source: Optional[str] = None,
) -> Dict:
    """Build metadata dictionary for the shared embedding cache."""
    return {
        "dataset_name": dataset_name,
        "dataset_root": dataset_root,
        "num_episodes": len(episode_indices),
        "episode_indices": episode_indices,
        "model_name": MODEL_NAME,
        "prompt_text": PROMPT_TEXT,
        "token_pooling": TOKEN_POOLING,
        "global_frame_rule": GLOBAL_FRAME_RULE,
        "wrist_start_ratio": WRIST_START_RATIO,
        "wrist_end_ratio": WRIST_END_RATIO,
        "temporal_pooling": TEMPORAL_POOLING,
        "pca_dim": pca_dim,
        "extractor_version": EXTRACTOR_VERSION,
        "extraction_method_name": build_extraction_method_name(pca_dim),
        "source": source or "generated",
    }


def load_cache_metadata(cache_dir: Path) -> Optional[Dict]:
    """Load metadata.json from a cache directory."""
    meta_file = cache_dir / "metadata.json"
    if not meta_file.exists():
        return None
    with open(meta_file) as f:
        return json.load(f)


def write_cache_metadata(cache_dir: Path, metadata: Dict) -> None:
    """Write metadata.json to a cache directory."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def validate_embedding_file(path: Path, expected_episode_index: int) -> Tuple[bool, List[str]]:
    """
    Validate a single embedding .npy file.
    
    Checks:
    - File can be loaded with np.load(allow_pickle=True).item()
    - Contains phi_global and phi_wrist
    - Both are non-empty 1D numpy arrays with finite values
    - Filename matches expected episode index pattern: ({index}).npy
    """
    issues = []
    
    expected_filename = f"({expected_episode_index}).npy"
    if path.name != expected_filename:
        issues.append(f"Filename mismatch: expected '{expected_filename}', got '{path.name}'")
    
    if not path.exists():
        issues.append(f"File does not exist: {path}")
        return False, issues
    
    try:
        data = np.load(str(path), allow_pickle=True).item()
    except Exception as e:
        issues.append(f"Failed to load {path}: {e}")
        return False, issues
    
    if "phi_global" not in data:
        issues.append(f"Missing 'phi_global' in {path.name}")
    if "phi_wrist" not in data:
        issues.append(f"Missing 'phi_wrist' in {path.name}")
    
    if "phi_global" in data:
        pg = data["phi_global"]
        if not isinstance(pg, np.ndarray) or pg.ndim != 1 or len(pg) == 0:
            issues.append(f"phi_global in {path.name} is not a non-empty 1D array")
        elif not np.all(np.isfinite(pg)):
            issues.append(f"phi_global in {path.name} contains non-finite values")
    
    if "phi_wrist" in data:
        pw = data["phi_wrist"]
        if not isinstance(pw, np.ndarray) or pw.ndim != 1 or len(pw) == 0:
            issues.append(f"phi_wrist in {path.name} is not a non-empty 1D array")
        elif not np.all(np.isfinite(pw)):
            issues.append(f"phi_wrist in {path.name} contains non-finite values")
    
    return len(issues) == 0, issues


def validate_shared_cache(
    cache_dir: Path,
    dataset_root: str,
    dataset_name: str,
    pca_dim: int = DEFAULT_PCA_DIM,
    strict: bool = True,
) -> Dict:
    """
    Validate a shared embedding cache directory.
    
    Checks:
    1. metadata.json exists and matches current configuration
    2. Every expected episode index has a corresponding ({index}).npy file
    3. Each .npy file passes validate_embedding_file
    4. PCA model files exist (pca_models/pca_global_{pca_dim}.joblib and pca_wrist_{pca_dim}.joblib)
    
    Returns:
        {"valid": bool, "issues": [...], "cache_dir": str, "num_expected": int, "num_found": int}
    """
    issues = []
    
    if not cache_dir.exists():
        return {
            "valid": False,
            "issues": [f"Cache directory does not exist: {cache_dir}"],
            "cache_dir": str(cache_dir),
            "num_expected": 0,
            "num_found": 0,
        }
    
    # Check metadata
    metadata = load_cache_metadata(cache_dir)
    if metadata is None:
        issues.append("metadata.json not found in cache directory")
    else:
        expected_method = build_extraction_method_name(pca_dim)
        checks = {
            "dataset_name": dataset_name,
            "model_name": MODEL_NAME,
            "prompt_text": PROMPT_TEXT,
            "token_pooling": TOKEN_POOLING,
            "global_frame_rule": GLOBAL_FRAME_RULE,
            "wrist_start_ratio": WRIST_START_RATIO,
            "wrist_end_ratio": WRIST_END_RATIO,
            "temporal_pooling": TEMPORAL_POOLING,
            "pca_dim": pca_dim,
            "extractor_version": EXTRACTOR_VERSION,
            "extraction_method_name": expected_method,
        }
        for key, expected_val in checks.items():
            actual_val = metadata.get(key)
            if actual_val != expected_val:
                issues.append(f"Metadata mismatch for '{key}': expected {expected_val!r}, got {actual_val!r}")
    
    # Get expected episode indices
    episode_indices = get_dataset_episode_indices(dataset_root)
    num_expected = len(episode_indices)
    
    # Check each episode file exists
    num_found = 0
    for ep_idx in episode_indices:
        ep_file = cache_dir / f"({ep_idx}).npy"
        if ep_file.exists():
            valid, file_issues = validate_embedding_file(ep_file, ep_idx)
            if valid:
                num_found += 1
            else:
                issues.extend(file_issues)
        else:
            issues.append(f"Missing embedding file for episode {ep_idx}: ({ep_idx}).npy")
    
    # Check PCA model files
    pca_dir = cache_dir / "pca_models"
    pca_global_file = pca_dir / f"pca_global_{pca_dim}.joblib"
    pca_wrist_file = pca_dir / f"pca_wrist_{pca_dim}.joblib"
    
    if not pca_global_file.exists():
        issues.append(f"Missing PCA model: {pca_global_file}")
    if not pca_wrist_file.exists():
        issues.append(f"Missing PCA model: {pca_wrist_file}")
    
    return {
        "valid": len(issues) == 0 and num_found == num_expected,
        "issues": issues,
        "cache_dir": str(cache_dir),
        "num_expected": num_expected,
        "num_found": num_found,
    }


def find_legacy_embedding_candidates(
    dataset_name: str,
    num_episodes: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[Path]:
    """
    Find potential legacy embedding directories in duibi experiment directories.
    
    Searches for embeddings in:
    - ours_112_seed42_{dataset_name}/embeddings (highest priority)
    - ours_v3_no_action_*_{dataset_name}/embeddings
    - ours_v4_*_{dataset_name}/embeddings
    - ours_v2_*_{dataset_name}/embeddings
    - subzerocore_*_{dataset_name}/embeddings
    - Any other duibi/**{dataset_name}/embeddings
    
    Returns list of candidate directories (does not validate them).
    """
    duibi_root = Path("/data/zhonglinye/jun/lerobot/personal/work2/duibi")
    candidates = []
    
    if not duibi_root.exists():
        return candidates
    
    normalized = normalize_dataset_name(dataset_name)
    short_name = normalized.replace("pick_place_", "")
    
    # Priority 1: ours original (complete cache)
    ours_pattern = f"ours_112_seed42_{short_name}"
    ours_dir = duibi_root / ours_pattern / "embeddings"
    if ours_dir.exists():
        candidates.append(ours_dir)
    
    # Priority 2: v3, v4, v2, subzerocore
    search_patterns = [
        f"ours_v3_no_action_*_{short_name}/embeddings",
        f"ours_v4_*_{short_name}/embeddings",
        f"ours_v2_*_{short_name}/embeddings",
        f"subzerocore_*_{short_name}/embeddings",
    ]
    
    for pattern in search_patterns:
        for d in sorted(duibi_root.glob(pattern)):
            if d.exists() and d not in candidates:
                candidates.append(d)
    
    # Priority 3: any other duibi/**{dataset_name}/embeddings
    for d in sorted(duibi_root.glob(f"**/{short_name}*/embeddings")):
        if d.exists() and d not in candidates:
            candidates.append(d)
    
    return candidates


def validate_legacy_cache(
    source_dir: Path,
    dataset_root: str,
    pca_dim: int = DEFAULT_PCA_DIM,
) -> Dict:
    """
    Validate a legacy cache directory (which may not have metadata.json).
    
    Checks:
    - All expected episode index .npy files exist
    - Each file has valid phi_global and phi_wrist
    - PCA model files exist
    
    Does NOT require metadata.json (legacy dirs typically don't have it).
    """
    issues = []
    
    if not source_dir.exists():
        return {
            "valid": False,
            "issues": [f"Legacy directory does not exist: {source_dir}"],
        }
    
    episode_indices = get_dataset_episode_indices(dataset_root)
    
    for ep_idx in episode_indices:
        ep_file = source_dir / f"({ep_idx}).npy"
        if ep_file.exists():
            valid, file_issues = validate_embedding_file(ep_file, ep_idx)
            if not valid:
                issues.extend(file_issues)
        else:
            issues.append(f"Missing embedding file for episode {ep_idx}: ({ep_idx}).npy")
    
    # Check PCA models
    pca_dir = source_dir / "pca_models"
    pca_global = pca_dir / f"pca_global_{pca_dim}.joblib"
    pca_wrist = pca_dir / f"pca_wrist_{pca_dim}.joblib"
    
    if not pca_global.exists():
        issues.append(f"Missing PCA model: {pca_global}")
    if not pca_wrist.exists():
        issues.append(f"Missing PCA model: {pca_wrist}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }


def copy_legacy_cache_to_shared(
    source_dir: Path,
    target_dir: Path,
    dataset_root: str,
    dataset_name: str,
    pca_dim: int = DEFAULT_PCA_DIM,
) -> Path:
    """
    Copy a legacy cache to the shared cache directory.
    
    Uses copy semantics (never deletes source). Creates a temporary building
    directory first, validates, then atomically renames to target.
    
    If target already exists and is valid, returns immediately.
    If target exists but is invalid, moves it to _invalid_backups first.
    """
    if target_dir.exists():
        validation = validate_shared_cache(target_dir, dataset_root, dataset_name, pca_dim)
        if validation["valid"]:
            print(f"CACHE_ALREADY_VALID: {target_dir}")
            return target_dir
    
    episode_indices = get_dataset_episode_indices(dataset_root)
    method_name = build_extraction_method_name(pca_dim)
    
    # Create temporary building directory
    pid = os.getpid()
    building_dir = Path(str(target_dir) + f".building.{pid}")
    building_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Copy all episode .npy files
        for ep_idx in episode_indices:
            src_file = source_dir / f"({ep_idx}).npy"
            if src_file.exists():
                dst_file = building_dir / f"({ep_idx}).npy"
                shutil.copy2(str(src_file), str(dst_file))
        
        # Copy PCA models
        pca_dir = building_dir / "pca_models"
        pca_dir.mkdir(parents=True, exist_ok=True)
        
        src_pca_dir = source_dir / "pca_models"
        pca_global_src = src_pca_dir / f"pca_global_{pca_dim}.joblib"
        pca_wrist_src = src_pca_dir / f"pca_wrist_{pca_dim}.joblib"
        
        if pca_global_src.exists():
            shutil.copy2(str(pca_global_src), str(pca_dir / f"pca_global_{pca_dim}.joblib"))
        if pca_wrist_src.exists():
            shutil.copy2(str(pca_wrist_src), str(pca_dir / f"pca_wrist_{pca_dim}.joblib"))
        
        # Write metadata
        metadata = build_expected_metadata(
            dataset_root, dataset_name, pca_dim, episode_indices,
            source=f"legacy_copy:{source_dir}"
        )
        write_cache_metadata(building_dir, metadata)
        
        # Validate
        validation = validate_shared_cache(building_dir, dataset_root, dataset_name, pca_dim)
        if not validation["valid"]:
            print(f"COPY_VALIDATION_FAILED: {validation['issues']}")
            raise RuntimeError(f"Validation failed after copy: {validation['issues']}")
        
        print(f"COPY_VALIDATION_PASSED")
        
        # If target exists but is invalid, move it to backups
        if target_dir.exists():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_root = SHARED_EMBEDDING_ROOT / "_invalid_backups"
            backup_root.mkdir(parents=True, exist_ok=True)
            backup_name = f"{dataset_name}_{method_name}_{timestamp}"
            backup_dir = backup_root / backup_name
            shutil.move(str(target_dir), str(backup_dir))
            print(f"Moved invalid target to backup: {backup_dir}")
        
        # Atomically rename building to target
        shutil.move(str(building_dir), str(target_dir))
        print(f"MIGRATED_SHARED_CACHE: {target_dir}")
        
        return target_dir
        
    except Exception:
        # Clean up building dir on failure, never touch source
        if building_dir.exists():
            shutil.rmtree(building_dir)
        raise