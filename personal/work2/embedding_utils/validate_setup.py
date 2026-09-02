#!/usr/bin/env python
"""
Static validation script for embedding_utils setup.

Checks:
1. All required modules can be imported
2. build_extraction_method_name(32) produces the exact expected string
3. get_shared_embedding_dir("corner", 32) produces the correct path suffix
4. process_dataset signature does NOT contain allow_partial_cache

This script does NOT load VLM or execute embedding extraction.
"""

import sys
import inspect
from pathlib import Path

WORK2_ROOT = Path(__file__).resolve().parent.parent
if str(WORK2_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK2_ROOT))


def check_imports():
    """Verify all required modules can be imported."""
    issues = []
    
    try:
        import embedding_utils.config
        print("  [OK] embedding_utils.config")
    except ImportError as e:
        issues.append(f"Cannot import embedding_utils.config: {e}")
        print(f"  [FAIL] embedding_utils.config: {e}")
    
    try:
        import embedding_utils.cache
        print("  [OK] embedding_utils.cache")
    except ImportError as e:
        issues.append(f"Cannot import embedding_utils.cache: {e}")
        print(f"  [FAIL] embedding_utils.cache: {e}")
    
    try:
        import our.embeddings.extract_embeddings
        print("  [OK] our.embeddings.extract_embeddings")
    except ImportError as e:
        issues.append(f"Cannot import our.embeddings.extract_embeddings: {e}")
        print(f"  [FAIL] our.embeddings.extract_embeddings: {e}")
    
    return issues


def check_method_name():
    """Verify build_extraction_method_name(32) produces the exact expected string."""
    issues = []
    
    from embedding_utils.config import build_extraction_method_name
    
    expected = "smolvlm2-500m_last-hidden-tokenmean_global-first5_wrist-20to70_temporal-mean_pca32_v1"
    actual = build_extraction_method_name(32)
    
    if actual == expected:
        print(f"  [OK] method name: {actual}")
    else:
        issues.append(f"Method name mismatch:\n    expected: {expected}\n    actual:   {actual}")
        print(f"  [FAIL] method name mismatch")
        print(f"    expected: {expected}")
        print(f"    actual:   {actual}")
    
    return issues


def check_shared_path(dataset_name="corner", pca_dim=32):
    """Verify path suffix is correct for given dataset."""
    issues = []
    
    from embedding_utils.cache import get_shared_embedding_dir
    
    expected_suffix = f"shared_embeddings/pick_place_{dataset_name}/smolvlm2-500m_last-hidden-tokenmean_global-first5_wrist-20to70_temporal-mean_pca{pca_dim}_v1"
    actual_path = get_shared_embedding_dir(dataset_name, pca_dim)
    actual_str = str(actual_path)
    
    if actual_str.endswith(expected_suffix):
        print(f"  [OK] shared path suffix matches for {dataset_name}")
    else:
        issues.append(f"Path suffix mismatch for {dataset_name}:\n    expected suffix: {expected_suffix}\n    actual path:     {actual_str}")
        print(f"  [FAIL] shared path suffix for {dataset_name}")
        print(f"    expected suffix: {expected_suffix}")
        print(f"    actual path:     {actual_str}")
    
    return issues


def check_no_partial_cache_option():
    """Verify process_dataset signature does NOT contain allow_partial_cache."""
    issues = []
    
    from our.embeddings.extract_embeddings import process_dataset
    
    sig = inspect.signature(process_dataset)
    params = list(sig.parameters.keys())
    
    if "allow_partial_cache" in params:
        issues.append(f"process_dataset still has 'allow_partial_cache' parameter: {params}")
        print(f"  [FAIL] process_dataset has allow_partial_cache parameter")
    else:
        print(f"  [OK] process_dataset signature: {params}")
    
    return issues


def main():
    all_issues = []
    
    print("=" * 60)
    print("Embedding Utils Setup Validation")
    print("=" * 60)
    
    print("\n1. Checking imports...")
    all_issues.extend(check_imports())
    
    print("\n2. Checking method name...")
    all_issues.extend(check_method_name())
    
    print("\n3. Checking shared path for corner...")
    all_issues.extend(check_shared_path("corner", 32))
    
    print("\n4. Checking shared path for corner2...")
    all_issues.extend(check_shared_path("corner2", 32))
    
    print("\n5. Checking shared path for corner3...")
    all_issues.extend(check_shared_path("corner3", 32))
    
    print("\n6. Checking no partial cache option...")
    all_issues.extend(check_no_partial_cache_option())
    
    print("\n" + "=" * 60)
    if all_issues:
        print(f"SETUP_VALIDATION_FAILED: {len(all_issues)} issue(s)")
        for issue in all_issues:
            print(f"  - {issue}")
        sys.exit(1)
    else:
        print("SETUP_VALIDATION_PASSED")
        print("All checks passed successfully.")


if __name__ == "__main__":
    main()