#!/usr/bin/env python
"""
V5 embedding cache entry point.

Ensures existing visual embeddings and action descriptors are available for V5 selection.
V5 does NOT generate new vision-action fused embeddings.
It reuses existing global+wrist visual embeddings and adds action descriptor as auxiliary signal.

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
    V5_EMBEDDING_VERSION,
    V5_ACTION_DESCRIPTOR_VERSION,
    V5_ACTION_STEPS,
    V5_ACTION_FEATURES,
    V5_ACTION_DESCRIPTOR_OUTPUT_DIR,
    build_v5_action_descriptor_name,
)
from embedding_utils.cache import (
    normalize_dataset_name,
    get_dataset_episode_indices,
    load_cache_metadata,
    write_cache_metadata,
)
from embedding_utils.config import (
    DEFAULT_PCA_DIM,
    build_extraction_method_name,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Ensure V5 visual embeddings and action descriptors exist")
    parser.add_argument("--dataset-root", type=str, required=True,
                       help="Dataset root directory")
    parser.add_argument("--dataset-name", type=str, required=True,
                       help="Dataset name (e.g., corner, corner2)")
    parser.add_argument("--gpu-id", type=int, default=0,
                       help="GPU ID for embedding extraction")
    parser.add_argument("--pca-dim", type=int, default=DEFAULT_PCA_DIM,
                       help=f"PCA dimension (default: {DEFAULT_PCA_DIM})")
    parser.add_argument("--path-file", type=str, default=None,
                       help="Output file to write the canonical shared embedding path")
    return parser.parse_args()


def _validate_visual_embedding_file(path: Path, expected_episode_index: int) -> tuple:
    """
    Validate a single visual embedding .npy file (phi_global + phi_wrist format).
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


def _validate_visual_cache(
    cache_dir: Path,
    dataset_root: str,
    dataset_name: str,
    pca_dim: int,
) -> dict:
    """
    Validate existing visual embedding cache directory (V1-V4 format).
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

    num_found = 0
    for ep_idx in expected_episode_indices:
        ep_file = cache_dir / f"({ep_idx}).npy"
        if ep_file.exists():
            valid, file_issues = _validate_visual_embedding_file(ep_file, ep_idx)
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


def _validate_action_descriptor_file(path: Path, expected_episode_index: int) -> tuple:
    """
    Validate a single action descriptor .npy file.
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

    if "action_descriptor" not in data:
        issues.append(f"Missing 'action_descriptor' in {path.name}")

    if "action_descriptor" in data:
        ad = data["action_descriptor"]
        if not isinstance(ad, np.ndarray) or ad.ndim != 1 or len(ad) == 0:
            issues.append(f"action_descriptor in {path.name} is not a non-empty 1D array")
        elif not np.all(np.isfinite(ad)):
            issues.append(f"action_descriptor in {path.name} contains non-finite values")

    return len(issues) == 0, issues


def _validate_action_descriptor_cache(
    cache_dir: Path,
    dataset_root: str,
    dataset_name: str,
) -> dict:
    """
    Validate action descriptor cache directory.
    """
    issues = []

    if not cache_dir.exists():
        return {
            "valid": False,
            "issues": [f"Action descriptor cache directory does not exist: {cache_dir}"],
            "cache_dir": str(cache_dir),
            "num_expected": 0,
            "num_found": 0,
        }

    expected_episode_indices = get_dataset_episode_indices(dataset_root)
    num_expected = len(expected_episode_indices)
    canonical_name = normalize_dataset_name(dataset_name)

    metadata = load_cache_metadata(cache_dir)
    if metadata is None:
        issues.append("metadata.json not found in action descriptor cache directory")
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

    num_found = 0
    for ep_idx in expected_episode_indices:
        ep_file = cache_dir / f"({ep_idx}).npy"
        if ep_file.exists():
            valid, file_issues = _validate_action_descriptor_file(ep_file, ep_idx)
            if valid:
                num_found += 1
            else:
                issues.extend(file_issues)
        else:
            issues.append(f"Missing action descriptor file for episode {ep_idx}: ({ep_idx}).npy")

    return {
        "valid": len(issues) == 0 and num_found == num_expected,
        "issues": issues,
        "cache_dir": str(cache_dir),
        "num_expected": num_expected,
        "num_found": num_found,
    }


def ensure_action_descriptors_exist(
    dataset_root: str,
    dataset_name: str,
    num_steps: int = V5_ACTION_STEPS,
) -> Path:
    """
    Check if action descriptor cache exists and is valid.
    If not, generate it.

    Returns the action descriptor cache directory path.
    """
    normalized = normalize_dataset_name(dataset_name)
    method_name = build_v5_action_descriptor_name(num_steps=num_steps)
    target_dir = V5_ACTION_DESCRIPTOR_OUTPUT_DIR / normalized / method_name

    print(f"V5 Action Descriptor method: {method_name}")
    print(f"V5 Action Descriptor cache path: {target_dir}")

    validation = _validate_action_descriptor_cache(target_dir, dataset_root, dataset_name)
    if validation["valid"]:
        print(f"FOUND_ACTION_DESCRIPTOR_CACHE: {target_dir}")
        print(f"  Episodes: {validation['num_found']}/{validation['num_expected']}")
        return target_dir

    print(f"Action Descriptor Cache status: INCOMPLETE or NOT FOUND")
    print(f"  Issues: {len(validation['issues'])}")
    for issue in validation["issues"][:3]:
        print(f"    - {issue}")

    print(f"\nGENERATING_ACTION_DESCRIPTORS")
    print(f"  Dataset: {dataset_name}")
    print(f"  Action steps: {num_steps}")

    from embedding_utils.action_descriptor import extract_all_action_descriptors, save_action_descriptors

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
        descriptors = extract_all_action_descriptors(dataset_root, building_dir)

        save_action_descriptors(descriptors, building_dir, dataset_root, dataset_name)

        validation = _validate_action_descriptor_cache(building_dir, dataset_root, dataset_name)
        if not validation["valid"]:
            print(f"V5_ACTION_DESCRIPTOR_GENERATION_VALIDATION_FAILED:")
            for issue in validation["issues"]:
                print(f"  - {issue}")
            print(f"Building directory preserved for debugging: {building_dir}")
            raise RuntimeError(f"Action descriptor generation validation failed: {validation['issues']}")

        print(f"V5_ACTION_DESCRIPTOR_GENERATION_VALIDATION_PASSED")

        if target_dir.exists():
            import shutil
            timestamp_move = time.strftime("%Y%m%d_%H%M%S")
            backup_root = V5_ACTION_DESCRIPTOR_OUTPUT_DIR / "_invalid_backups"
            backup_root.mkdir(parents=True, exist_ok=True)
            backup_name = f"{normalized}_{method_name}_{timestamp_move}"
            backup_dir = backup_root / backup_name
            shutil.move(str(target_dir), str(backup_dir))
            print(f"Moved invalid target to backup: {backup_dir}")

        import shutil
        shutil.move(str(building_dir), str(target_dir))

        print(f"V5_ACTION_DESCRIPTOR_CACHE_READY: {target_dir}")
        print(f"  Episodes: {validation['num_found']}/{validation['num_expected']}")

        return target_dir

    except Exception as e:
        print(f"V5_ACTION_DESCRIPTOR_GENERATION_FAILED: {e}")
        print(f"Building directory preserved for debugging: {building_dir}")
        raise


def ensure_v5_embeddings_exist(
    dataset_root: str,
    dataset_name: str,
    gpu_id: int,
    pca_dim: int = DEFAULT_PCA_DIM,
    action_steps: int = V5_ACTION_STEPS,
) -> tuple:
    """
    Ensure both visual embeddings and action descriptors are available for V5 selection.

    Execution order:
    1. Check existing visual embedding cache (reuse V1-V4 embeddings)
    2. Check/generate action descriptor cache
    3. Return both paths

    Returns:
        (visual_embedding_dir, action_descriptor_dir) tuple of Path objects
    """
    if pca_dim is None:
        pca_dim = DEFAULT_PCA_DIM

    normalized = normalize_dataset_name(dataset_name)
    method_name = build_extraction_method_name(pca_dim)
    visual_dir = V5_SHARED_EMBEDDING_ROOT / normalized / method_name

    print(f"V5 Dataset: {dataset_name}")
    print(f"V5 Visual embedding method: {method_name}")
    print(f"V5 Visual cache path: {visual_dir}")

    validation = _validate_visual_cache(visual_dir, dataset_root, dataset_name, pca_dim)
    if validation["valid"]:
        print(f"FOUND_VISUAL_EMBEDDING_CACHE: {visual_dir}")
        print(f"  Episodes: {validation['num_found']}/{validation['num_expected']}")
    else:
        print(f"Visual Embedding Cache status: INCOMPLETE or NOT FOUND")
        print(f"  Issues: {len(validation['issues'])}")
        for issue in validation["issues"][:3]:
            print(f"    - {issue}")
        print(f"\nGenerating visual embeddings via ensure_embeddings.py flow...")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        from embedding_utils.ensure_embeddings import ensure_shared_embeddings
        visual_dir = ensure_shared_embeddings(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            gpu_id=gpu_id,
            pca_dim=pca_dim,
            allow_legacy_migration=True,
        )
        print(f"Visual embedding cache ready: {visual_dir}")

    action_descriptor_dir = ensure_action_descriptors_exist(
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        num_steps=action_steps,
    )

    return visual_dir, action_descriptor_dir


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    visual_dir, action_descriptor_dir = ensure_v5_embeddings_exist(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        gpu_id=args.gpu_id,
        pca_dim=args.pca_dim,
    )

    if args.path_file:
        path_file = Path(args.path_file)
        path_file.parent.mkdir(parents=True, exist_ok=True)
        with open(path_file, "w") as f:
            json.dump({
                "visual_embedding_dir": str(visual_dir.resolve()),
                "action_descriptor_dir": str(action_descriptor_dir.resolve()),
            }, f, indent=2)
        print(f"Path written to: {path_file}")


if __name__ == "__main__":
    main()