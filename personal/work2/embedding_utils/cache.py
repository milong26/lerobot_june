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

# Legacy source types that are known to use the same SmolVLM extraction method
AUTO_MIGRATION_ALLOWED_TYPES = {
    "ours_v1",
    "ours_v3",
    "ours_v4",
    "ours_v2",
}

# Old method name variants that should be checked for migration to new canonical name
OLD_METHOD_NAME_VARIANTS = [
    "smolvlm2-500m_last-hidden-token-mean_global-first5_wrist-20to70_temporal-mean_pca{pca_dim}_v1",
]

KNOWN_LEGACY_SOURCE_TYPES = {
    "ours_v1",
    "ours_v2",
    "ours_v3",
    "ours_v4",
    "subzerocore",
    "manual",
    "previous_shared_cache",
}


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
    """
    Build metadata dictionary for the shared embedding cache.
    
    dataset_name is stored in normalized (canonical) form.
    dataset_realpath stores the resolved absolute path of dataset_root.
    """
    canonical_name = normalize_dataset_name(dataset_name)
    return {
        "dataset_name": canonical_name,
        "dataset_root": dataset_root,
        "dataset_realpath": str(Path(dataset_root).resolve()),
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


def validate_previous_shared_cache(
    cache_dir: Path,
    dataset_root: str,
    dataset_name: str,
    pca_dim: int = DEFAULT_PCA_DIM,
    allowed_old_method_names: Optional[set] = None,
) -> Dict:
    """
    Validate an old shared cache directory that used a previous method name variant.
    
    This is identical to validate_shared_cache in all metadata checks, except:
    - metadata["extraction_method_name"] can equal any value in allowed_old_method_names
      instead of the current canonical method name.
    
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

    expected_episode_indices = get_dataset_episode_indices(dataset_root)
    num_expected = len(expected_episode_indices)
    canonical_name = normalize_dataset_name(dataset_name)

    metadata = load_cache_metadata(cache_dir)
    if metadata is None:
        issues.append("metadata.json not found in cache directory")
    else:
        if "num_episodes" not in metadata:
            issues.append("Metadata missing required field: num_episodes")
        else:
            meta_num = metadata["num_episodes"]
            if meta_num != num_expected:
                issues.append(f"Metadata num_episodes mismatch: expected {num_expected}, got {meta_num}")

        if "episode_indices" not in metadata:
            issues.append("Metadata missing required field: episode_indices")
        else:
            meta_episodes = metadata["episode_indices"]
            if meta_episodes != expected_episode_indices:
                issues.append(
                    f"Metadata episode_indices mismatch: "
                    f"expected {expected_episode_indices}, got {meta_episodes}"
                )

        if "dataset_name" not in metadata:
            issues.append("Metadata missing required field: dataset_name")
        else:
            if metadata["dataset_name"] != canonical_name:
                issues.append(f"Metadata mismatch for 'dataset_name': expected {canonical_name!r}, got {metadata['dataset_name']!r}")

        if "dataset_realpath" not in metadata:
            issues.append("Metadata missing required field: dataset_realpath")
        else:
            current_realpath = str(Path(dataset_root).resolve())
            if metadata["dataset_realpath"] != current_realpath:
                issues.append(
                    f"Dataset realpath mismatch: "
                    f"cache has '{metadata['dataset_realpath']}', current is '{current_realpath}'"
                )

        required_fields = {
            "model_name": MODEL_NAME,
            "prompt_text": PROMPT_TEXT,
            "token_pooling": TOKEN_POOLING,
            "global_frame_rule": GLOBAL_FRAME_RULE,
            "wrist_start_ratio": WRIST_START_RATIO,
            "wrist_end_ratio": WRIST_END_RATIO,
            "temporal_pooling": TEMPORAL_POOLING,
            "pca_dim": pca_dim,
            "extractor_version": EXTRACTOR_VERSION,
        }
        for key, expected_val in required_fields.items():
            if key not in metadata:
                issues.append(f"Metadata missing required field: {key}")
            else:
                if metadata[key] != expected_val:
                    issues.append(f"Metadata mismatch for '{key}': expected {expected_val!r}, got {metadata[key]!r}")

        if "extraction_method_name" not in metadata:
            issues.append("Metadata missing required field: extraction_method_name")
        else:
            actual_method = metadata["extraction_method_name"]
            if allowed_old_method_names is not None:
                if actual_method not in allowed_old_method_names:
                    issues.append(
                        f"Metadata extraction_method_name '{actual_method}' "
                        f"is not in allowed old method names: {allowed_old_method_names}"
                    )
            else:
                expected_method = build_extraction_method_name(pca_dim)
                if actual_method != expected_method:
                    issues.append(f"Metadata mismatch for 'extraction_method_name': expected {expected_method!r}, got {actual_method!r}")

    num_found = 0
    for ep_idx in expected_episode_indices:
        ep_file = cache_dir / f"({ep_idx}).npy"
        if ep_file.exists():
            valid, file_issues = validate_embedding_file(ep_file, ep_idx)
            if valid:
                num_found += 1
            else:
                issues.extend(file_issues)
        else:
            issues.append(f"Missing embedding file for episode {ep_idx}: ({ep_idx}).npy")

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


def classify_legacy_dir_name(name: str, dataset_name: str) -> Optional[str]:
    """
    Classify a legacy directory basename into a source_type.
    
    Pure function that does not access the filesystem.
    
    Returns one of: "ours_v1", "ours_v2", "ours_v3", "ours_v4", "subzerocore", or None.
    """
    normalized = normalize_dataset_name(dataset_name)
    short_name = normalized.replace("pick_place_", "")

    if name.startswith("ours_v2_") and name.endswith(f"_{short_name}"):
        return "ours_v2"
    if name.startswith("ours_v3_") and name.endswith(f"_{short_name}"):
        return "ours_v3"
    if name.startswith("ours_v4_") and name.endswith(f"_{short_name}"):
        return "ours_v4"
    if name.startswith("subzerocore_") and name.endswith(f"_{short_name}"):
        return "subzerocore"

    if name.startswith("ours_") and name.endswith(f"_{short_name}"):
        parts = name.split("_")
        if len(parts) >= 4 and parts[0] == "ours":
            second_part = parts[1]
            if second_part.isdigit():
                return "ours_v1"

    return None


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
    1. metadata.json exists and ALL required fields are present with correct values
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
    
    expected_episode_indices = get_dataset_episode_indices(dataset_root)
    num_expected = len(expected_episode_indices)
    canonical_name = normalize_dataset_name(dataset_name)
    
    metadata = load_cache_metadata(cache_dir)
    if metadata is None:
        issues.append("metadata.json not found in cache directory")
    else:
        expected_method = build_extraction_method_name(pca_dim)
        
        if "num_episodes" not in metadata:
            issues.append("Metadata missing required field: num_episodes")
        else:
            meta_num = metadata["num_episodes"]
            if meta_num != num_expected:
                issues.append(f"Metadata num_episodes mismatch: expected {num_expected}, got {meta_num}")
        
        if "episode_indices" not in metadata:
            issues.append("Metadata missing required field: episode_indices")
        else:
            meta_episodes = metadata["episode_indices"]
            if meta_episodes != expected_episode_indices:
                issues.append(
                    f"Metadata episode_indices mismatch: "
                    f"expected {expected_episode_indices}, got {meta_episodes}"
                )
        
        if "dataset_name" not in metadata:
            issues.append("Metadata missing required field: dataset_name")
        else:
            if metadata["dataset_name"] != canonical_name:
                issues.append(f"Metadata mismatch for 'dataset_name': expected {canonical_name!r}, got {metadata['dataset_name']!r}")
        
        if "dataset_realpath" not in metadata:
            issues.append("Metadata missing required field: dataset_realpath")
        else:
            current_realpath = str(Path(dataset_root).resolve())
            if metadata["dataset_realpath"] != current_realpath:
                issues.append(
                    f"Dataset realpath mismatch: "
                    f"cache has '{metadata['dataset_realpath']}', current is '{current_realpath}'"
                )
        
        required_fields = {
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
        for key, expected_val in required_fields.items():
            if key not in metadata:
                issues.append(f"Metadata missing required field: {key}")
            else:
                actual_val = metadata[key]
                if actual_val != expected_val:
                    issues.append(f"Metadata mismatch for '{key}': expected {expected_val!r}, got {actual_val!r}")
    
    num_found = 0
    for ep_idx in expected_episode_indices:
        ep_file = cache_dir / f"({ep_idx}).npy"
        if ep_file.exists():
            valid, file_issues = validate_embedding_file(ep_file, ep_idx)
            if valid:
                num_found += 1
            else:
                issues.extend(file_issues)
        else:
            issues.append(f"Missing embedding file for episode {ep_idx}: ({ep_idx}).npy")
    
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
) -> List[Dict]:
    """
    Find potential legacy embedding directories in duibi experiment directories.
    
    Returns list of dicts with structure:
        {"path": Path(...), "source_type": "ours_v1"}
    
    Priority: ours_v1 -> ours_v3 -> ours_v4 -> ours_v2 -> known subzerocore
    Generic duibi/**/{dataset}*/embeddings is NOT included in auto-migration candidates.
    """
    duibi_root = Path("/data/zhonglinye/jun/lerobot/personal/work2/duibi")
    candidates = []
    seen = set()
    
    if not duibi_root.exists():
        return candidates
    
    normalized = normalize_dataset_name(dataset_name)
    short_name = normalized.replace("pick_place_", "")
    
    v1_prefixes_to_exclude = ("ours_v2_", "ours_v3_", "ours_v3_no_action_", "ours_v4_")
    
    # Priority 1: ours original (ours_112_seed42_{dataset_name}/embeddings)
    # Only match true V1 dirs: ours_{number}_seed*_{short_name}
    for d in sorted(duibi_root.glob(f"ours_*_seed*_{short_name}")):
        if d.name.startswith(v1_prefixes_to_exclude):
            continue
        parts = d.name.split("_")
        if len(parts) >= 4 and parts[0] == "ours" and parts[1].isdigit():
            emb_dir = d / "embeddings"
            if emb_dir.exists() and emb_dir not in seen:
                seen.add(emb_dir)
                candidates.append({"path": emb_dir, "source_type": "ours_v1"})
    
    # Priority 2: ours_v3_no_action
    for d in sorted(duibi_root.glob(f"ours_v3_no_action_*_{short_name}")):
        emb_dir = d / "embeddings"
        if emb_dir.exists() and emb_dir not in seen:
            seen.add(emb_dir)
            candidates.append({"path": emb_dir, "source_type": "ours_v3"})
    
    # Priority 3: ours_v4
    for d in sorted(duibi_root.glob(f"ours_v4_*_{short_name}")):
        emb_dir = d / "embeddings"
        if emb_dir.exists() and emb_dir not in seen:
            seen.add(emb_dir)
            candidates.append({"path": emb_dir, "source_type": "ours_v4"})
    
    # Priority 4: ours_v2
    for d in sorted(duibi_root.glob(f"ours_v2_*_{short_name}")):
        emb_dir = d / "embeddings"
        if emb_dir.exists() and emb_dir not in seen:
            seen.add(emb_dir)
            candidates.append({"path": emb_dir, "source_type": "ours_v2"})
    
    # Priority 5: subzerocore (only if we can verify it uses the same extraction method)
    # These are lower priority because SubZeroCore may have its own embedding logic
    for d in sorted(duibi_root.glob(f"subzerocore_*_{short_name}")):
        emb_dir = d / "embeddings"
        if emb_dir.exists() and emb_dir not in seen:
            seen.add(emb_dir)
            candidates.append({"path": emb_dir, "source_type": "subzerocore"})
    
    return candidates


def validate_legacy_cache(
    source_dir: Path,
    dataset_root: str,
    pca_dim: int = DEFAULT_PCA_DIM,
    source_type: Optional[str] = None,
) -> Dict:
    """
    Validate a legacy cache directory (which may not have metadata.json).
    
    Checks:
    - source_type is a known valid type (if provided)
    - All expected episode index .npy files exist
    - Each file has valid phi_global and phi_wrist
    - PCA model files exist
    
    source_type helps distinguish known-same-method caches from unknown ones.
    For source_type="manual", only file/PCA completeness is checked.
    """
    issues = []

    if source_type is not None and source_type not in KNOWN_LEGACY_SOURCE_TYPES:
        issues.append(f"Unknown legacy source_type: {source_type}")
        return {
            "valid": False,
            "issues": issues,
        }
    
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
    source_type: Optional[str] = None,
    source_validation_kind: str = "legacy",
    allowed_old_method_names: Optional[set] = None,
) -> Path:
    """
    Copy a legacy cache to the shared cache directory.
    
    Uses copy semantics (never deletes source). Creates a temporary building
    directory first, validates, then atomically renames to target.
    
    If target already exists and is valid, returns immediately.
    If target exists but is invalid, moves it to _invalid_backups first.
    
    source_validation_kind:
        "legacy" - calls validate_legacy_cache before copy
        "previous_shared" - calls validate_previous_shared_cache before copy
    """
    if target_dir.exists():
        validation = validate_shared_cache(target_dir, dataset_root, dataset_name, pca_dim)
        if validation["valid"]:
            print(f"CACHE_ALREADY_VALID: {target_dir}")
            return target_dir
    
    if source_validation_kind == "previous_shared":
        src_validation = validate_previous_shared_cache(
            source_dir, dataset_root, dataset_name, pca_dim,
            allowed_old_method_names=allowed_old_method_names,
        )
    else:
        src_validation = validate_legacy_cache(source_dir, dataset_root, pca_dim, source_type=source_type)
    
    if not src_validation["valid"]:
        raise RuntimeError(f"Source validation failed: {src_validation['issues']}")
    
    episode_indices = get_dataset_episode_indices(dataset_root)
    method_name = build_extraction_method_name(pca_dim)
    
    pid = os.getpid()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    building_dir_name = f"{target_dir.name}.building.{timestamp}.{pid}"
    building_dir = Path(str(target_dir.parent) + "/" + building_dir_name)
    
    if building_dir.exists():
        timestamp2 = time.strftime("%Y%m%d_%H%M%S_%f")
        building_dir_name = f"{target_dir.name}.building.{timestamp2}.{pid}"
        building_dir = Path(str(target_dir.parent) + "/" + building_dir_name)
    
    building_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        for ep_idx in episode_indices:
            src_file = source_dir / f"({ep_idx}).npy"
            if src_file.exists():
                dst_file = building_dir / f"({ep_idx}).npy"
                shutil.copy2(str(src_file), str(dst_file))
        
        pca_dir = building_dir / "pca_models"
        pca_dir.mkdir(parents=True, exist_ok=True)
        
        src_pca_dir = source_dir / "pca_models"
        pca_global_src = src_pca_dir / f"pca_global_{pca_dim}.joblib"
        pca_wrist_src = src_pca_dir / f"pca_wrist_{pca_dim}.joblib"
        
        if pca_global_src.exists():
            shutil.copy2(str(pca_global_src), str(pca_dir / f"pca_global_{pca_dim}.joblib"))
        if pca_wrist_src.exists():
            shutil.copy2(str(pca_wrist_src), str(pca_dir / f"pca_wrist_{pca_dim}.joblib"))
        
        metadata = build_expected_metadata(
            dataset_root, dataset_name, pca_dim, episode_indices,
            source=f"legacy_copy:{source_dir}"
        )
        write_cache_metadata(building_dir, metadata)
        
        validation = validate_shared_cache(building_dir, dataset_root, dataset_name, pca_dim)
        if not validation["valid"]:
            print(f"COPY_VALIDATION_FAILED: {validation['issues']}")
            print(f"BUILDING_DIR_PRESERVED: {building_dir}")
            raise RuntimeError(f"Validation failed after copy: {validation['issues']}")
        
        print(f"COPY_VALIDATION_PASSED")
        
        if target_dir.exists():
            timestamp_move = time.strftime("%Y%m%d_%H%M%S")
            backup_root = SHARED_EMBEDDING_ROOT / "_invalid_backups"
            backup_root.mkdir(parents=True, exist_ok=True)
            backup_name = f"{normalize_dataset_name(dataset_name)}_{method_name}_{timestamp_move}"
            backup_dir = backup_root / backup_name
            shutil.move(str(target_dir), str(backup_dir))
            print(f"Moved invalid target to backup: {backup_dir}")
        
        shutil.move(str(building_dir), str(target_dir))
        print(f"MIGRATED_SHARED_CACHE: {target_dir}")
        
        return target_dir
        
    except Exception as e:
        if building_dir.exists():
            print(f"BUILDING_DIR_PRESERVED: {building_dir}")
        raise


def find_previous_shared_cache_candidates(
    dataset_name: str,
    pca_dim: int = DEFAULT_PCA_DIM,
) -> List[Dict]:
    """
    Find old shared cache directories that used a previous method name variant.
    
    Returns list of dicts with structure:
        {"path": Path(...), "old_method_name": "smolvlm2-500m_last-hidden-token-mean_..."}
    """
    normalized = normalize_dataset_name(dataset_name)
    dataset_dir = SHARED_EMBEDDING_ROOT / normalized
    
    if not dataset_dir.exists():
        return []
    
    candidates = []
    
    for old_pattern in OLD_METHOD_NAME_VARIANTS:
        old_method = old_pattern.format(pca_dim=pca_dim)
        old_dir = dataset_dir / old_method
        if old_dir.exists():
            candidates.append({"path": old_dir, "old_method_name": old_method})
    
    return candidates