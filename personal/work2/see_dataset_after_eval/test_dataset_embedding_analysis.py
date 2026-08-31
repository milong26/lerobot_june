#!/usr/bin/env python
"""
Unit tests for dataset embedding analysis functions

Tests cover:
1. Correlated embeddings yield high Spearman
2. Shuffled embeddings yield Spearman near 0
3. Ridge can recover x/y from linear embeddings
4. Neighbor overlap higher for structured embeddings than random
5. Coverage metric better for farthest-point subset than clustered subset
6. Bootstrap percentile direction correct
7. Duplicate detection
8. NaN detection
9. Fixed-universe SIC uses same dbar for different subsets
10. Output JSON serializable
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from analyze_dataset_embedding import (
    compute_spearman_correlation,
    permutation_test_spearman,
    position_probe,
    neighbor_overlap_analysis,
    compute_subset_coverage,
    compute_subset_redundancy,
    random_bootstrap_analysis,
    compute_effective_rank,
    compute_embedding_statistics,
    compute_combined_embeddings,
    match_subset_to_indices,
    compute_workspace_coverage,
)
from analysis_utils import compute_fixed_universe_sic
from sic_v2 import check_embeddings_valid


def create_test_embeddings(n_episodes=50, dim=16, seed=42):
    """Create test embeddings"""
    rng = np.random.RandomState(seed)
    episode_indices = list(range(n_episodes))
    phi_globals = rng.randn(n_episodes, dim).astype(np.float32)
    phi_wrists = rng.randn(n_episodes, dim).astype(np.float32)
    return episode_indices, phi_globals, phi_wrists


def test_1_correlated_embedding_spearman():
    """Test 1: Correlated embeddings should yield high Spearman correlation"""
    print("\n=== Test 1: Correlated embedding Spearman ===")
    
    rng = np.random.RandomState(42)
    n = 100
    positions = rng.rand(n, 2)
    
    embeddings = positions + rng.randn(n, 2) * 0.05
    
    phys_dist = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
    emb_dist = np.linalg.norm(embeddings[:, np.newaxis, :] - embeddings[np.newaxis, :, :], axis=2)
    
    rho, p_value, n_pairs = compute_spearman_correlation(phys_dist, emb_dist, seed=42)
    
    assert rho > 0.8, f"Expected high correlation, got {rho:.2f}"
    assert p_value < 0.001, f"Expected significant p-value, got {p_value:.2e}"
    
    print(f"  PASS: Spearman rho={rho:.4f}, p={p_value:.2e}")
    return True


def test_2_shuffled_embedding_spearman():
    """Test 2: Shuffled embeddings should yield Spearman near 0"""
    print("\n=== Test 2: Shuffled embedding Spearman ===")
    
    rng = np.random.RandomState(42)
    n = 100
    positions = rng.rand(n, 2)
    
    shuffled_embeddings = rng.rand(n, 2)
    
    phys_dist = np.linalg.norm(positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=2)
    emb_dist = np.linalg.norm(shuffled_embeddings[:, np.newaxis, :] - shuffled_embeddings[np.newaxis, :, :], axis=2)
    
    rho, p_value, n_pairs = compute_spearman_correlation(phys_dist, emb_dist, seed=42)
    
    assert abs(rho) < 0.1, f"Expected near-zero correlation, got {rho:.2f}"
    
    print(f"  PASS: Spearman rho={rho:.4f} (near zero)")
    return True


def test_3_ridge_recovers_xy():
    """Test 3: Ridge can recover x/y from linear embeddings"""
    print("\n=== Test 3: Ridge recovers x/y ===")
    
    rng = np.random.RandomState(42)
    n = 200
    
    x = rng.rand(n, 1) * 10
    y = rng.rand(n, 1) * 5
    
    embeddings = np.concatenate([x, y], axis=1) + rng.randn(n, 2) * 0.1
    
    probe = position_probe(embeddings, np.concatenate([x, y], axis=1), n_shuffles=10, seed=42)
    
    assert probe['ridge']['R2_x'] > 0.9, f"Expected high R2_x, got {probe['ridge']['R2_x']:.4f}"
    assert probe['ridge']['R2_y'] > 0.9, f"Expected high R2_y, got {probe['ridge']['R2_y']:.4f}"
    
    print(f"  PASS: Ridge R2_x={probe['ridge']['R2_x']:.4f}, R2_y={probe['ridge']['R2_y']:.4f}")
    return True


def test_4_neighbor_overlap_structured_vs_random():
    """Test 4: Neighbor overlap higher for structured embeddings than random"""
    print("\n=== Test 4: Neighbor overlap structured vs random ===")
    
    rng = np.random.RandomState(42)
    n = 100
    
    positions = rng.rand(n, 2)
    
    structured_embeddings = positions + rng.randn(n, 2) * 0.05
    
    overlap = neighbor_overlap_analysis(structured_embeddings, positions, ks=[5, 10], seed=42)
    
    assert overlap['neighbor_overlap@5'] > overlap['random_neighbor_overlap@5'], \
        f"Structured overlap should be higher than random"
    assert overlap['neighbor_overlap@10'] > overlap['random_neighbor_overlap@10'], \
        f"Structured overlap should be higher than random"
    
    print(f"  PASS: overlap@5={overlap['neighbor_overlap@5']:.4f} > random@5={overlap['random_neighbor_overlap@5']:.4f}")
    return True


def test_5_coverage_far_vs_clustered():
    """Test 5: Coverage metric better for farthest-point subset than clustered"""
    print("\n=== Test 5: Coverage far vs clustered ===")
    
    rng = np.random.RandomState(42)
    n = 100
    dim = 2
    
    positions = rng.rand(n, dim)
    
    farthest_indices = [0, n//2, n//4, 3*n//4, n//3, 2*n//3]
    clustered_indices = [0, 1, 2, 3, 4, 5]
    
    coverage_far = compute_subset_coverage(positions, farthest_indices, positions, "far")
    coverage_clustered = compute_subset_coverage(positions, clustered_indices, positions, "clustered")
    
    assert coverage_far['mean_nearest_distance'] < coverage_clustered['mean_nearest_distance'], \
        f"Farthest should have better coverage"
    
    print(f"  PASS: far mean={coverage_far['mean_nearest_distance']:.4f} < clustered mean={coverage_clustered['mean_nearest_distance']:.4f}")
    return True


def test_6_bootstrap_percentile_direction():
    """Test 6: Bootstrap percentile direction correct"""
    print("\n=== Test 6: Bootstrap percentile direction ===")
    
    rng = np.random.RandomState(42)
    n = 50
    dim = 2
    
    embeddings = rng.rand(n, dim)
    
    good_subset = [0, n//2, n//4, 3*n//4, n//3]
    bad_subset = [0, 1, 2, 3, 4]
    
    bootstrap_good = random_bootstrap_analysis(embeddings, good_subset, n_bootstrap=100, seed=42)
    bootstrap_bad = random_bootstrap_analysis(embeddings, bad_subset, n_bootstrap=100, seed=42)
    
    assert bootstrap_good is not None, "Good subset bootstrap should not be None"
    assert bootstrap_bad is not None, "Bad subset bootstrap should not be None"
    
    good_better = bootstrap_good['mean_nearest']['better_than_random_fraction']
    bad_better = bootstrap_bad['mean_nearest']['better_than_random_fraction']
    
    assert good_better > bad_better, \
        f"Good subset should have higher better_than_random fraction"
    
    print(f"  PASS: good better_than_random={good_better:.4f} > bad better_than_random={bad_better:.4f}")
    return True


def test_7_duplicate_detection():
    """Test 7: Duplicate detection works"""
    print("\n=== Test 7: Duplicate detection ===")
    
    rng = np.random.RandomState(42)
    n = 20
    dim = 4
    
    phi = rng.randn(n, dim).astype(np.float32)
    phi[5] = phi[10]
    phi[7] = phi[12]
    
    result = check_embeddings_valid(phi, phi)
    
    assert result['stats']['n_exact_dup_global'] >= 2, \
        f"Expected at least 2 duplicates, got {result['stats']['n_exact_dup_global']}"
    assert not result['valid'], "Should be invalid due to duplicates"
    
    print(f"  PASS: Detected {result['stats']['n_exact_dup_global']} duplicates")
    return True


def test_8_nan_detection():
    """Test 8: NaN detection works"""
    print("\n=== Test 8: NaN detection ===")
    
    rng = np.random.RandomState(42)
    n = 20
    dim = 4
    
    phi = rng.randn(n, dim).astype(np.float32)
    phi[5, 2] = np.nan
    
    result = check_embeddings_valid(phi, phi)
    
    assert not result['valid'], "Should be invalid due to NaN"
    assert any("NaN" in err for err in result['errors']), "Should mention NaN in errors"
    
    print(f"  PASS: Detected NaN, errors={result['errors']}")
    return True


def test_9_fixed_universe_sic_same_dbar():
    """Test 9: Fixed-universe SIC uses same dbar for different subsets"""
    print("\n=== Test 9: Fixed-universe SIC same dbar ===")
    
    episode_indices, phi_g, phi_w = create_test_embeddings(n_episodes=30, dim=8)
    
    subset_a = episode_indices[:10]
    subset_b = episode_indices[10:20]
    
    sic_a = compute_fixed_universe_sic(subset_a, episode_indices, phi_g, phi_w)
    sic_b = compute_fixed_universe_sic(subset_b, episode_indices, phi_g, phi_w)
    
    assert abs(sic_a['dbar_global'] - sic_b['dbar_global']) < 1e-10, \
        f"dbar_global should be same: {sic_a['dbar_global']} vs {sic_b['dbar_global']}"
    assert abs(sic_a['dbar_wrist'] - sic_b['dbar_wrist']) < 1e-10, \
        f"dbar_wrist should be same: {sic_a['dbar_wrist']} vs {sic_b['dbar_wrist']}"
    
    print(f"  PASS: dbar_global={sic_a['dbar_global']:.6f}, dbar_wrist={sic_a['dbar_wrist']:.6f} (same for both subsets)")
    return True


def test_10_json_serializable():
    """Test 10: Output is JSON serializable"""
    print("\n=== Test 10: JSON serializable ===")
    
    rng = np.random.RandomState(42)
    n = 50
    dim = 4
    
    phi = rng.randn(n, dim).astype(np.float32)
    positions = rng.rand(n, 2)
    
    stats = compute_embedding_statistics(phi, "test")
    
    try:
        json_str = json.dumps(stats, default=str)
        parsed = json.loads(json_str)
        assert 'effective_rank' in parsed, "Should have effective_rank"
    except Exception as e:
        raise AssertionError(f"JSON serialization failed: {e}")
    
    print(f"  PASS: JSON serializable, effective_rank={stats['effective_rank']:.4f}")
    return True


def main():
    print("\n" + "="*60)
    print("Dataset Embedding Analysis Unit Tests")
    print("="*60)
    
    tests = [
        test_1_correlated_embedding_spearman,
        test_2_shuffled_embedding_spearman,
        test_3_ridge_recovers_xy,
        test_4_neighbor_overlap_structured_vs_random,
        test_5_coverage_far_vs_clustered,
        test_6_bootstrap_percentile_direction,
        test_7_duplicate_detection,
        test_8_nan_detection,
        test_9_fixed_universe_sic_same_dbar,
        test_10_json_serializable,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
                print(f"  FAILED: {test_fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAILED: {test_fn.__name__} - {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\nAll tests passed!")


if __name__ == "__main__":
    main()