#!/usr/bin/env python
"""
Static validation script for embedding_utils setup.

Checks:
1. All required modules can be imported
2. build_extraction_method_name(32) produces the exact expected string
3. get_shared_embedding_dir("corner", 32) produces the correct path suffix
4. process_dataset signature does NOT contain allow_partial_cache
5. Legacy directory name classification is correct
6. Required metadata validation is strict (missing fields fail)
7. Previous method name variants are correct

This script does NOT load VLM or execute embedding extraction.
"""

import sys
import inspect
import tempfile
import json
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


def check_legacy_classification():
    """Verify classify_legacy_dir_name correctly classifies known directory names."""
    issues = []
    
    from embedding_utils.cache import classify_legacy_dir_name
    
    test_cases = [
        ("ours_112_seed42_corner", "corner", "ours_v1"),
        ("ours_v2_112_seed42_corner", "corner", "ours_v2"),
        ("ours_v3_no_action_112_seed42_corner", "corner", "ours_v3"),
        ("ours_v4_112_seed42_corner", "corner", "ours_v4"),
        ("subzerocore_112_seed42_corner", "corner", "subzerocore"),
    ]
    
    for dir_name, dataset_name, expected_type in test_cases:
        actual = classify_legacy_dir_name(dir_name, dataset_name)
        if actual == expected_type:
            print(f"  [OK] classify('{dir_name}') -> {actual}")
        else:
            issues.append(f"classify_legacy_dir_name('{dir_name}', '{dataset_name}'): expected '{expected_type}', got '{actual}'")
            print(f"  [FAIL] classify('{dir_name}') -> {actual} (expected {expected_type})")
    
    return issues


def check_required_metadata_validation():
    """Verify that validate_shared_cache fails when required metadata fields are missing."""
    issues = []
    
    from embedding_utils.cache import validate_shared_cache, write_cache_metadata, build_expected_metadata
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "test_cache"
        cache_dir.mkdir()
        
        metadata = build_expected_metadata(
            dataset_root=tmpdir,
            dataset_name="corner",
            pca_dim=32,
            episode_indices=list(range(5)),
        )
        
        for field_to_remove in ["episode_indices", "dataset_realpath", "num_episodes", "dataset_name"]:
            test_meta = {k: v for k, v in metadata.items() if k != field_to_remove}
            write_cache_metadata(cache_dir, test_meta)
            
            result = validate_shared_cache(cache_dir, tmpdir, "corner", 32)
            if result["valid"]:
                issues.append(f"validate_shared_cache passed with missing '{field_to_remove}'")
                print(f"  [FAIL] validation passed with missing '{field_to_remove}'")
            else:
                missing_issue = any("missing required field" in issue for issue in result["issues"])
                if missing_issue:
                    print(f"  [OK] validation correctly fails when '{field_to_remove}' is missing")
                else:
                    issues.append(f"validation failed for missing '{field_to_remove}' but not with 'missing required field' message: {result['issues']}")
                    print(f"  [FAIL] validation failed for wrong reason: {result['issues']}")
    
    return issues


def check_previous_method_name():
    """Verify OLD_METHOD_NAME_VARIANTS and build_extraction_method_name consistency."""
    issues = []
    
    from embedding_utils.cache import OLD_METHOD_NAME_VARIANTS
    from embedding_utils.config import build_extraction_method_name
    
    expected_old = "smolvlm2-500m_last-hidden-token-mean_global-first5_wrist-20to70_temporal-mean_pca32_v1"
    expected_new = "smolvlm2-500m_last-hidden-tokenmean_global-first5_wrist-20to70_temporal-mean_pca32_v1"
    
    actual_old = OLD_METHOD_NAME_VARIANTS[0].format(pca_dim=32)
    actual_new = build_extraction_method_name(32)
    
    if actual_old == expected_old:
        print(f"  [OK] OLD_METHOD_NAME_VARIANTS[0] (pca_dim=32): {actual_old}")
    else:
        issues.append(f"OLD_METHOD_NAME_VARIANTS mismatch:\n    expected: {expected_old}\n    actual:   {actual_old}")
        print(f"  [FAIL] OLD_METHOD_NAME_VARIANTS mismatch")
        print(f"    expected: {expected_old}")
        print(f"    actual:   {actual_old}")
    
    if actual_new == expected_new:
        print(f"  [OK] build_extraction_method_name(32): {actual_new}")
    else:
        issues.append(f"build_extraction_method_name mismatch:\n    expected: {expected_new}\n    actual:   {actual_new}")
        print(f"  [FAIL] build_extraction_method_name mismatch")
        print(f"    expected: {expected_new}")
        print(f"    actual:   {actual_new}")
    
    if actual_old != actual_new:
        print(f"  [OK] old and new method names are different (as expected)")
    else:
        issues.append("Old and new method names should be different but are the same")
        print(f"  [FAIL] old and new method names are the same")
    
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
    
    print("\n7. Checking legacy directory classification...")
    all_issues.extend(check_legacy_classification())
    
    print("\n8. Checking required metadata validation...")
    all_issues.extend(check_required_metadata_validation())
    
    print("\n9. Checking previous method name...")
    all_issues.extend(check_previous_method_name())
    
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