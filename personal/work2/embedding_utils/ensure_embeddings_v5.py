#!/usr/bin/env python
"""
V5 embedding cache entry point.

Ensures the V5 vision-action shared embedding cache exists and is valid.
This is a standalone script that does NOT modify ensure_embeddings.py.

Usage:
    python ensure_embeddings_v5.py \
        --dataset-root /data/.../dataset_view/pick_place_corner \
        --dataset-name corner \
        --gpu-id 0 \
        --pca-dim 32 \
        --path-file /path/to/output/shared_embedding_path.txt
"""

import sys
import os
import json
import argparse
import time
import numpy as np
from pathlib import Path

WORK2_ROOT = Path(__file__).resolve().parent.parent
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))

from embedding_utils.config_v5 import (
    V5_SHARED_EMBEDDING_ROOT,
    V5_PCA_DIM,
    V5_MODEL_NAME,
    V5_PROMPT_TEXT,
    V5_FRAME_SAMPLE_COUNT,
    V5_TEMPORAL_ACTION_STEPS,
    V5_EMBEDDING_VERSION,
    build_v5_extraction_method_name,
)
from embedding_utils.cache import (
    normalize_dataset_name,
    get_dataset_episode_indices,
    load_cache_metadata,
    write_cache_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Ensure V5 shared embedding cache exists")
    parser.add_argument("--dataset-root", type=str, required=True,
                       help="Dataset root directory")
    parser.add_argument("--dataset-name", type=str, required=True,
                       help="Dataset name (e.g., corner, corner2)")
    parser.add_argument("--gpu-id", type=int, default=0,
                       help="GPU ID for embedding extraction")
    parser.add_argument("--pca-dim", type=int, default=V5_PCA_DIM,
                       help=f"PCA dimension (default: {V5_PCA_DIM})")
    parser.add_argument("--path-file", type=str, default=None,
                       help="Output file to write the canonical shared embedding path")
    return parser.parse_args()


def _validate_v5_embedding_file(path: Path, expected_episode_index: int) -> tuple:
    """
    Validate a single V5 embedding .npy file.
    
    V5 uses 'episode_embedding' key instead of 'phi_global'/'phi_wrist'.
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

    if "episode_embedding" not in data:
        issues.append(f"Missing 'episode_embedding' in {path.name}")

    if "episode_embedding" in data:
        pe = data["episode_embedding"]
        if not isinstance(pe, np.ndarray) or pe.ndim != 1 or len(pe) == 0:
            issues.append(f"episode_embedding in {path.name} is not a non-empty 1D array")
        elif not np.all(np.isfinite(pe)):
            issues.append(f"episode_embedding in {path.name} contains non-finite values")

    return len(issues) == 0, issues


def _validate_v5_cache(
    cache_dir: Path,
    dataset_root: str,
    dataset_name: str,
    pca_dim: int,
) -> dict:
    """
    Validate a V5 shared embedding cache directory.
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
        expected_method = build_v5_extraction_method_name(pca_dim)

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
            "model_name": V5_MODEL_NAME,
            "prompt_text": V5_PROMPT_TEXT,
            "frame_sample_count": V5_FRAME_SAMPLE_COUNT,
            "temporal_action_steps": V5_TEMPORAL_ACTION_STEPS,
            "pca_dim": pca_dim,
            "extractor_version": V5_EMBEDDING_VERSION,
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
            valid, file_issues = _validate_v5_embedding_file(ep_file, ep_idx)
            if valid:
                num_found += 1
            else:
                issues.extend(file_issues)
        else:
            issues.append(f"Missing embedding file for episode {ep_idx}: ({ep_idx}).npy")

    pca_dir = cache_dir / "pca_models"
    pca_v5_file = pca_dir / f"pca_v5_{pca_dim}.joblib"

    if not pca_v5_file.exists():
        issues.append(f"Missing PCA model: {pca_v5_file}")

    return {
        "valid": len(issues) == 0 and num_found == num_expected,
        "issues": issues,
        "cache_dir": str(cache_dir),
        "num_expected": num_expected,
        "num_found": num_found,
    }


def _build_v5_metadata(
    dataset_root: str,
    dataset_name: str,
    pca_dim: int,
    episode_indices: list,
    source: str = "generated",
) -> dict:
    """Build V5-specific metadata dictionary."""
    canonical_name = normalize_dataset_name(dataset_name)
    return {
        "dataset_name": canonical_name,
        "dataset_root": dataset_root,
        "dataset_realpath": str(Path(dataset_root).resolve()),
        "num_episodes": len(episode_indices),
        "episode_indices": episode_indices,
        "model_name": V5_MODEL_NAME,
        "prompt_text": V5_PROMPT_TEXT,
        "frame_sample_count": V5_FRAME_SAMPLE_COUNT,
        "temporal_action_steps": V5_TEMPORAL_ACTION_STEPS,
        "pca_dim": pca_dim,
        "extractor_version": V5_EMBEDDING_VERSION,
        "extraction_method_name": build_v5_extraction_method_name(pca_dim),
        "source": source,
    }


def ensure_v5_embeddings(
    dataset_root: str,
    dataset_name: str,
    gpu_id: int,
    pca_dim: int = V5_PCA_DIM,
) -> Path:
    """
    Ensure V5 vision-action embedding cache exists and is valid.
    
    Returns the canonical shared embedding directory path.
    """
    if pca_dim is None:
        pca_dim = V5_PCA_DIM

    normalized = normalize_dataset_name(dataset_name)
    method_name = build_v5_extraction_method_name(pca_dim)
    target_dir = V5_SHARED_EMBEDDING_ROOT / normalized / method_name

    print(f"V5 Dataset: {dataset_name}")
    print(f"V5 Extraction method: {method_name}")
    print(f"V5 Shared cache path: {target_dir}")

    # Check if V5 cache already exists and is valid
    validation = _validate_v5_cache(target_dir, dataset_root, dataset_name, pca_dim)
    if validation["valid"]:
        print(f"FOUND_V5_SHARED_CACHE: {target_dir}")
        print(f"  Episodes: {validation['num_found']}/{validation['num_expected']}")
        return target_dir

    print(f"V5 Cache status: INCOMPLETE or NOT FOUND")
    print(f"  Issues: {len(validation['issues'])}")
    for issue in validation["issues"][:3]:
        print(f"    - {issue}")

    # Generate new V5 embeddings
    print(f"\nGENERATING_V5_EMBEDDINGS")
    print(f"  GPU: {gpu_id}")
    print(f"  Dataset: {dataset_name}")
    print(f"  PCA dim: {pca_dim}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from our_v5.extract_embeddings_v5 import process_v5_embeddings

    episode_indices = get_dataset_episode_indices(dataset_root)
    print(f"  Expected episodes: {len(episode_indices)}")

    pid = os.getpid()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    building_dir_name = f"{target_dir.name}.building.{timestamp}.{pid}"
    building_dir = Path(str(target_dir.parent) + "/" + building_dir_name)

    if building_dir.exists():
        timestamp2 = time.strftime("%Y%m%d_%H%M%S_%f")
        building_dir_name = f"{target_dir.name}.building.{timestamp2}.{pid}"
        building_dir = Path(str(target_dir.parent) + "/" + building_dir_name)

    print(f"  Temporary path: {building_dir}")

    try:
        process_v5_embeddings(
            dataset_dir=Path(dataset_root),
            output_dir=building_dir,
            n_components=pca_dim,
            device="cuda",
        )

        # Build V5-specific metadata
        metadata = _build_v5_metadata(
            dataset_root, dataset_name, pca_dim, episode_indices,
            source="generated"
        )
        write_cache_metadata(building_dir, metadata)

        # Validate
        validation = _validate_v5_cache(building_dir, dataset_root, dataset_name, pca_dim)
        if not validation["valid"]:
            print(f"V5_GENERATION_VALIDATION_FAILED:")
            for issue in validation["issues"]:
                print(f"  - {issue}")
            print(f"Building directory preserved for debugging: {building_dir}")
            raise RuntimeError(f"V5 generation validation failed: {validation['issues']}")

        print(f"V5_GENERATION_VALIDATION_PASSED")

        # Move building to canonical target
        if target_dir.exists():
            import shutil
            timestamp_move = time.strftime("%Y%m%d_%H%M%S")
            backup_root = V5_SHARED_EMBEDDING_ROOT / "_invalid_backups"
            backup_root.mkdir(parents=True, exist_ok=True)
            backup_name = f"{normalized}_{method_name}_{timestamp_move}"
            backup_dir = backup_root / backup_name
            shutil.move(str(target_dir), str(backup_dir))
            print(f"Moved invalid target to backup: {backup_dir}")

        import shutil
        shutil.move(str(building_dir), str(target_dir))

        print(f"V5_SHARED_CACHE_READY: {target_dir}")
        print(f"  Episodes: {validation['num_found']}/{validation['num_expected']}")

        return target_dir

    except Exception as e:
        print(f"V5_GENERATION_FAILED: {e}")
        print(f"Building directory preserved for debugging: {building_dir}")
        raise


def main():
    args = parse_args()
    
    # Set GPU before any torch import
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    shared_dir = ensure_v5_embeddings(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        gpu_id=args.gpu_id,
        pca_dim=args.pca_dim,
    )
    
    # Write path file if requested
    if args.path_file:
        path_file = Path(args.path_file)
        path_file.parent.mkdir(parents=True, exist_ok=True)
        with open(path_file, "w") as f:
            f.write(str(shared_dir.resolve()))
        print(f"Path written to: {path_file}")


if __name__ == "__main__":
    main()