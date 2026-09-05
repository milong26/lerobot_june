#!/usr/bin/env python
"""
Unified embedding cache entry point.

All experiments (V2, V3, V4, SubZeroCore, etc.) call this script to ensure
the shared embedding cache exists and is valid. It handles:
1. Checking if shared cache already exists and is valid
2. Migrating from legacy caches if available
3. Generating new embeddings if no cache exists

Usage:
    python ensure_embeddings.py \
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
from pathlib import Path

# Set GPU before any other imports that might use CUDA
if "--gpu-id" in sys.argv:
    idx = sys.argv.index("--gpu-id")
    if idx + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[idx + 1]

WORK2_ROOT = Path(__file__).resolve().parent.parent
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))

from embedding_utils.config import DEFAULT_PCA_DIM, build_extraction_method_name
from embedding_utils.cache import (
    get_shared_embedding_dir,
    get_dataset_episode_indices,
    build_expected_metadata,
    write_cache_metadata,
    validate_shared_cache,
    validate_previous_shared_cache,
    find_legacy_embedding_candidates,
    validate_legacy_cache,
    copy_legacy_cache_to_shared,
    find_previous_shared_cache_candidates,
    AUTO_MIGRATION_ALLOWED_TYPES,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Ensure shared embedding cache exists")
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
    parser.add_argument("--no-legacy-migration", action="store_true",
                       help="Skip legacy cache migration")
    return parser.parse_args()


def ensure_shared_embeddings(
    dataset_root: str,
    dataset_name: str,
    gpu_id: int,
    pca_dim: int = DEFAULT_PCA_DIM,
    allow_legacy_migration: bool = True,
) -> Path:
    """
    Ensure the shared embedding cache exists and is valid.
    
    Priority:
    1. Return existing valid shared cache
    2. Migrate from legacy cache if available (only known same-method sources)
    3. Migrate from old shared cache with previous method name variant
    4. Generate new embeddings from scratch
    
    Returns the canonical shared embedding directory path.
    """
    target_dir = get_shared_embedding_dir(dataset_name, pca_dim)
    method_name = build_extraction_method_name(pca_dim)
    
    print(f"Dataset: {dataset_name}")
    print(f"Extraction method: {method_name}")
    print(f"Shared cache path: {target_dir}")
    
    # Step 1: Check if shared cache already exists and is valid
    validation = validate_shared_cache(target_dir, dataset_root, dataset_name, pca_dim)
    if validation["valid"]:
        print(f"FOUND_SHARED_CACHE: {target_dir}")
        print(f"  Episodes: {validation['num_found']}/{validation['num_expected']}")
        return target_dir
    
    print(f"Cache status: INCOMPLETE or NOT FOUND")
    print(f"  Issues: {len(validation['issues'])}")
    for issue in validation["issues"][:3]:
        print(f"    - {issue}")
    if len(validation["issues"]) > 3:
        print(f"    ... and {len(validation['issues']) - 3} more")
    
    # Step 2: Try legacy migration (only known same-method sources)
    if allow_legacy_migration:
        print(f"\nSearching for legacy embedding candidates...")
        candidates = find_legacy_embedding_candidates(dataset_name)
        
        for candidate_info in candidates:
            candidate = candidate_info["path"]
            source_type = candidate_info["source_type"]
            print(f"  Checking: {candidate} (source_type={source_type})")
            
            if source_type not in AUTO_MIGRATION_ALLOWED_TYPES:
                print(f"  -> SKIP_UNKNOWN_LEGACY_SOURCE (not in allowed auto-migration types)")
                continue
            
            legacy_validation = validate_legacy_cache(candidate, dataset_root, pca_dim, source_type=source_type)
            if legacy_validation["valid"]:
                print(f"  -> VALID")
                print(f"MIGRATING_LEGACY_CACHE: {candidate}")
                print(f"  Source: {candidate}")
                print(f"  Target: {target_dir}")
                
                try:
                    copy_legacy_cache_to_shared(
                        candidate, target_dir, dataset_root, dataset_name, pca_dim,
                        source_type=source_type,
                    )
                    print(f"COPY_VALIDATION_PASSED")
                    print(f"MIGRATED_SHARED_CACHE: {target_dir}")
                    return target_dir
                except Exception as e:
                    print(f"  Migration failed: {e}")
                    print(f"  Continuing to next candidate...")
            else:
                print(f"  -> INCOMPLETE ({len(legacy_validation['issues'])} issues)")
        
        print(f"No complete legacy cache found for auto-migration.")
    
    # Step 3: Try migrating from old shared cache with previous method name
    print(f"\nChecking for previous shared cache candidates (old method name variants)...")
    old_candidates = find_previous_shared_cache_candidates(dataset_name, pca_dim)
    
    for candidate_info in old_candidates:
        old_dir = candidate_info["path"]
        old_method_name = candidate_info["old_method_name"]
        print(f"  Checking old shared cache: {old_dir}")
        
        old_validation = validate_previous_shared_cache(
            old_dir, dataset_root, dataset_name, pca_dim,
            allowed_old_method_names={old_method_name},
        )
        if old_validation["valid"]:
            print(f"  -> VALID, migrating to new canonical name...")
            try:
                copy_legacy_cache_to_shared(
                    old_dir, target_dir, dataset_root, dataset_name, pca_dim,
                    source_type="previous_shared_cache",
                    source_validation_kind="previous_shared",
                    allowed_old_method_names={old_method_name},
                )
                new_method_name = build_extraction_method_name(pca_dim)
                print(f"MIGRATED_FROM_OLD_SHARED: {old_dir} -> {target_dir}")
                print(f"  Old method: {old_method_name}")
                print(f"  New method: {new_method_name}")
                print(f"  Original directory preserved: {old_dir}")
                return target_dir
            except Exception as e:
                print(f"  Migration from old shared cache failed: {e}")
        else:
            print(f"  -> INVALID ({len(old_validation['issues'])} issues)")
    
    # Step 4: Generate new embeddings
    print(f"\nNO_VALID_CACHE_FOUND")
    print(f"GENERATING_SHARED_EMBEDDINGS")
    print(f"  GPU: {gpu_id}")
    print(f"  Dataset: {dataset_name}")
    print(f"  PCA dim: {pca_dim}")
    
    # Set GPU before importing torch
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Import extraction logic from our/embeddings/extract_embeddings.py
    from our.embeddings.extract_embeddings import process_dataset
    
    episode_indices = get_dataset_episode_indices(dataset_root)
    print(f"  Expected episodes: {len(episode_indices)}")
    
    # Create unique temporary building directory (never reuse existing)
    pid = os.getpid()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    building_dir_name = f"{target_dir.name}.building.{timestamp}.{pid}"
    building_dir = Path(str(target_dir.parent) + "/" + building_dir_name)
    
    # If building dir already exists, generate a new unique name
    if building_dir.exists():
        timestamp2 = time.strftime("%Y%m%d_%H%M%S_%f")
        building_dir_name = f"{target_dir.name}.building.{timestamp2}.{pid}"
        building_dir = Path(str(target_dir.parent) + "/" + building_dir_name)
    
    print(f"  Temporary path: {building_dir}")
    
    try:
        # Run full extraction in the building directory
        process_dataset(
            dataset_dir=Path(dataset_root),
            output_dir=building_dir,
            n_components=pca_dim,
            device="cuda",
        )
        
        # Write metadata
        metadata = build_expected_metadata(
            dataset_root, dataset_name, pca_dim, episode_indices,
            source="generated"
        )
        write_cache_metadata(building_dir, metadata)
        
        # Validate
        validation = validate_shared_cache(building_dir, dataset_root, dataset_name, pca_dim)
        if not validation["valid"]:
            print(f"GENERATION_VALIDATION_FAILED:")
            for issue in validation["issues"]:
                print(f"  - {issue}")
            print(f"Building directory preserved for debugging: {building_dir}")
            raise RuntimeError(f"Generation validation failed: {validation['issues']}")
        
        print(f"GENERATION_VALIDATION_PASSED")
        
        # Move building to canonical target
        if target_dir.exists():
            # Target exists but was invalid, move to backup
            import shutil
            timestamp_move = time.strftime("%Y%m%d_%H%M%S")
            backup_root = target_dir.parent.parent / "_invalid_backups"
            backup_root.mkdir(parents=True, exist_ok=True)
            backup_name = f"{dataset_name}_{method_name}_{timestamp_move}"
            backup_dir = backup_root / backup_name
            shutil.move(str(target_dir), str(backup_dir))
            print(f"Moved invalid target to backup: {backup_dir}")
        
        import shutil
        shutil.move(str(building_dir), str(target_dir))
        
        print(f"SHARED_CACHE_READY: {target_dir}")
        print(f"  Episodes: {validation['num_found']}/{validation['num_expected']}")
        
        return target_dir
        
    except Exception as e:
        print(f"GENERATION_FAILED: {e}")
        print(f"Building directory preserved for debugging: {building_dir}")
        raise


def main():
    args = parse_args()
    
    shared_dir = ensure_shared_embeddings(
        dataset_root=args.dataset_root,
        dataset_name=args.dataset_name,
        gpu_id=args.gpu_id,
        pca_dim=args.pca_dim,
        allow_legacy_migration=not args.no_legacy_migration,
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