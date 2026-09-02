#!/usr/bin/env python
"""
Explicit legacy embedding migration script.

Copies (never moves/deletes) complete legacy embedding caches to the shared
canonical directory. Validates before and after copy.

Usage:
    # Migrate from a specific source directory
    python migrate_embeddings.py \
        --dataset-root /data/.../dataset_view/pick_place_corner \
        --dataset-name corner \
        --source-dir /data/.../duibi/ours_112_seed42_corner/embeddings

    # Auto-find and migrate the best available legacy cache
    python migrate_embeddings.py \
        --dataset-root /data/.../dataset_view/pick_place_corner \
        --dataset-name corner \
        --auto
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from embedding_utils.config import DEFAULT_PCA_DIM
from embedding_utils.cache import (
    get_shared_embedding_dir,
    validate_shared_cache,
    find_legacy_embedding_candidates,
    validate_legacy_cache,
    copy_legacy_cache_to_shared,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Migrate legacy embeddings to shared cache")
    parser.add_argument("--dataset-root", type=str, required=True,
                       help="Dataset root directory")
    parser.add_argument("--dataset-name", type=str, required=True,
                       help="Dataset name (e.g., corner, corner2)")
    parser.add_argument("--source-dir", type=str, default=None,
                       help="Specific legacy embedding directory to migrate from")
    parser.add_argument("--pca-dim", type=int, default=DEFAULT_PCA_DIM,
                       help=f"PCA dimension (default: {DEFAULT_PCA_DIM})")
    parser.add_argument("--auto", action="store_true",
                       help="Auto-find and migrate the best available legacy cache")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset_root
    dataset_name = args.dataset_name
    pca_dim = args.pca_dim
    
    target_dir = get_shared_embedding_dir(dataset_name, pca_dim)
    
    # Check if already valid
    validation = validate_shared_cache(target_dir, dataset_root, dataset_name, pca_dim)
    if validation["valid"]:
        print(f"CACHE_ALREADY_VALID: {target_dir}")
        print(f"  Episodes: {validation['num_found']}/{validation['num_expected']}")
        sys.exit(0)
    
    if args.source_dir:
        # Migrate from specific source
        source_dir = Path(args.source_dir)
        if not source_dir.exists():
            print(f"ERROR: Source directory does not exist: {source_dir}")
            sys.exit(1)
        
        print(f"Validating legacy cache: {source_dir}")
        legacy_validation = validate_legacy_cache(source_dir, dataset_root, pca_dim)
        if not legacy_validation["valid"]:
            print(f"ERROR: Legacy cache is incomplete:")
            for issue in legacy_validation["issues"]:
                print(f"  - {issue}")
            sys.exit(1)
        
        print(f"Legacy cache valid, copying to shared...")
        copy_legacy_cache_to_shared(source_dir, target_dir, dataset_root, dataset_name, pca_dim)
        print(f"MIGRATED_FROM: {source_dir}")
        print(f"SHARED_EMBEDDING_DIR: {target_dir}")
        
    elif args.auto:
        # Auto-find best legacy cache
        print(f"Searching for legacy embedding candidates...")
        candidates = find_legacy_embedding_candidates(dataset_name)
        
        if not candidates:
            print(f"NO_VALID_LEGACY_CACHE_FOUND")
            print(f"No legacy embedding directories found for dataset: {dataset_name}")
            sys.exit(1)
        
        migrated = False
        for candidate in candidates:
            print(f"  Checking: {candidate}")
            legacy_validation = validate_legacy_cache(candidate, dataset_root, pca_dim)
            if legacy_validation["valid"]:
                print(f"  -> VALID, migrating...")
                copy_legacy_cache_to_shared(candidate, target_dir, dataset_root, dataset_name, pca_dim)
                print(f"MIGRATED_FROM: {candidate}")
                print(f"SHARED_EMBEDDING_DIR: {target_dir}")
                migrated = True
                break
            else:
                print(f"  -> INCOMPLETE ({len(legacy_validation['issues'])} issues)")
        
        if not migrated:
            print(f"NO_VALID_LEGACY_CACHE_FOUND")
            print(f"Found {len(candidates)} candidate directories, but none are complete.")
            sys.exit(1)
    else:
        print(f"ERROR: Specify either --source-dir or --auto")
        sys.exit(1)
    
    # Final validation
    final_validation = validate_shared_cache(target_dir, dataset_root, dataset_name, pca_dim)
    if final_validation["valid"]:
        print(f"Migration complete and validated.")
        print(f"  Shared cache: {target_dir}")
        print(f"  Episodes: {final_validation['num_found']}/{final_validation['num_expected']}")
    else:
        print(f"ERROR: Final validation failed:")
        for issue in final_validation["issues"]:
            print(f"  - {issue}")
        sys.exit(1)


if __name__ == "__main__":
    main()