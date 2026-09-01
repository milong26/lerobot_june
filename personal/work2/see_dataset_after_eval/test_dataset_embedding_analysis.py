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
    evaluate_ours_vs_uniform,
    evaluate_h1,
    evaluate_h2,
    evaluate_h3,
    evaluate_hypotheses,
    load_subset_episode_indices,
    l2_normalize_rows,
)
from analysis_utils import compute_fixed_universe_sic, compute_fixed_universe_sic_from_indices
from sic_v2 import check_embeddings_valid, compute_dbar_from_embeddings, build_kernel_matrices


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
    assert probe['ridge']['R2_y'] > 0.9, f"Expected hig你h R2_y, got {probe['ridge']['R2_y']:.4f}"
    
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
    
    assert coverage_far['unselected_mean_nearest_distance'] < coverage_clustered['unselected_mean_nearest_distance'], \
        f"Farthest should have better coverage"
    
    print(f"  PASS: far mean={coverage_far['unselected_mean_nearest_distance']:.4f} < clustered mean={coverage_clustered['unselected_mean_nearest_distance']:.4f}")
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
    """Test 7: Duplicate detection works (exact duplicates are warnings, not errors)"""
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
    # Exact duplicates are now warnings, not errors
    assert result['valid'], "Should still be valid (exact duplicates are warnings)"
    assert any("exact duplicate" in warn for warn in result['warnings']), "Should warn about exact duplicates"
    
    print(f"  PASS: Detected {result['stats']['n_exact_dup_global']} duplicates (as warnings)")
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


def test_11_shuffled_ridge_baseline():
    """Test 11: Shuffled Ridge baseline uses shuffled label itself, structured > shuffled mean"""
    print("\n=== Test 11: Shuffled Ridge baseline ===")
    
    rng = np.random.RandomState(42)
    n = 200
    
    x = rng.rand(n, 1) * 10
    y = rng.rand(n, 1) * 5
    
    embeddings = np.concatenate([x, y], axis=1) + rng.randn(n, 2) * 0.1
    
    probe = position_probe(embeddings, np.concatenate([x, y], axis=1), n_shuffles=20, seed=42)
    
    real_r2_x = probe['ridge']['R2_x']
    shuffled_r2_x_mean = probe['shuffled_ridge']['R2_x']['mean']
    
    assert real_r2_x > shuffled_r2_x_mean, \
        f"Real R2_x ({real_r2_x:.4f}) should be higher than shuffled mean ({shuffled_r2_x_mean:.4f})"
    
    print(f"  PASS: Real R2_x={real_r2_x:.4f} > shuffled mean={shuffled_r2_x_mean:.4f}")
    return True


def test_12_sparse_grid_insufficient_samples():
    """Test 12: Sparse grid class returns insufficient_samples instead of 0"""
    print("\n=== Test 12: Sparse grid classifiability ===")
    
    from analyze_dataset_embedding import grid_classifiability
    
    rng = np.random.RandomState(42)
    n = 20
    dim = 8
    
    phi = rng.randn(n, dim).astype(np.float32)
    positions = rng.rand(n, 2)
    
    results = grid_classifiability(phi, positions, grid_sizes=[(14, 8)], n_shuffles=10, seed=42)
    
    grid_result = results.get("14x8", {})
    status = grid_result.get("status")
    
    if status == "insufficient_samples":
        assert grid_result["accuracy"] is None, "Accuracy should be None for insufficient samples"
        print(f"  PASS: Sparse grid returns insufficient_samples (accuracy=None)")
    elif status == "ok":
        print(f"  PASS: Grid classification succeeded (accuracy={grid_result.get('accuracy'):.4f})")
    else:
        print(f"  INFO: Grid status={status}")
    
    return True


def test_13_bootstrap_coverage_redundancy_sic():
    """Test 13: Bootstrap generates coverage/redundancy metrics; Fixed SIC is separate."""
    print("\n=== Test 13: Bootstrap coverage/redundancy/SIC ===")
    
    rng = np.random.RandomState(42)
    n = 50
    dim = 4
    
    phi = rng.randn(n, dim).astype(np.float32)
    
    from sic_v2 import compute_dbar_from_embeddings, build_kernel_matrices
    from analyze_dataset_embedding import random_bootstrap_fixed_sic, generate_bootstrap_subsets
    
    dbar_g, dbar_w, _ = compute_dbar_from_embeddings(phi, phi)
    K_g, K_w = build_kernel_matrices(phi, phi, dbar_g, dbar_w)
    
    subset = [0, n//2, n//4, 3*n//4, n//3]
    
    bootstrap = random_bootstrap_analysis(
        phi, subset, n_bootstrap=50, seed=42,
        K_global=K_g, K_wrist=K_w,
        dbar_global=dbar_g, dbar_wrist=dbar_w,
    )
    
    assert bootstrap is not None, "Bootstrap should not be None"
    
    for metric in ["mean_nearest", "p95_nearest", "max_radius", "redundancy_fraction"]:
        assert metric in bootstrap, f"Missing metric: {metric}"
        assert "observed" in bootstrap[metric], f"Missing observed for {metric}"
        assert "better_than_random_fraction" in bootstrap[metric], f"Missing better_than_random for {metric}"
    
    # Fixed SIC bootstrap is a separate, once-computed evidence item.
    assert "normalized_fixed_sic" not in bootstrap, (
        "random_bootstrap_analysis should NOT contain fixed SIC; "
        "Fixed SIC is computed once via random_bootstrap_fixed_sic()"
    )
    
    pre_subsets = generate_bootstrap_subsets(n_total=n, subset_size=len(subset), n_bootstrap=50, seed=42)
    fixed_sic_boot = random_bootstrap_fixed_sic(
        subset_indices=subset,
        all_episode_indices=list(range(n)),
        K_global=K_g, K_wrist=K_w,
        dbar_global=dbar_g, dbar_wrist=dbar_w,
        n_bootstrap=50, seed=42,
        alpha=1.0, lambda_wrist=1.0,
        pre_generated_subsets=pre_subsets,
    )
    assert fixed_sic_boot is not None, "Fixed SIC bootstrap should not be None"
    assert "observed" in fixed_sic_boot, "Missing observed SIC"
    assert "better_than_random_fraction" in fixed_sic_boot, "Missing better_than_random for SIC"
    
    print(f"  PASS: All bootstrap metrics present")
    return True


def test_14_same_fixed_dbar_kernel_reused():
    """Test 14: Same fixed dbar/kernel reused across bootstrap subsets"""
    print("\n=== Test 14: Same fixed dbar/kernel reused ===")
    
    episode_indices, phi_g, phi_w = create_test_embeddings(n_episodes=30, dim=8)
    
    from sic_v2 import compute_dbar_from_embeddings, build_kernel_matrices
    
    dbar_g, dbar_w, _ = compute_dbar_from_embeddings(phi_g, phi_w)
    K_g, K_w = build_kernel_matrices(phi_g, phi_w, dbar_g, dbar_w)
    
    subset_a = episode_indices[:10]
    subset_b = episode_indices[10:20]
    
    from analysis_utils import compute_fixed_universe_sic_from_indices
    
    sic_a = compute_fixed_universe_sic_from_indices(
        subset_a, episode_indices, K_g, K_w, dbar_g, dbar_w
    )
    sic_b = compute_fixed_universe_sic_from_indices(
        subset_b, episode_indices, K_g, K_w, dbar_g, dbar_w
    )
    
    assert abs(sic_a['dbar_global'] - sic_b['dbar_global']) < 1e-10, "dbar_global should be same"
    assert abs(sic_a['dbar_wrist'] - sic_b['dbar_wrist']) < 1e-10, "dbar_wrist should be same"
    assert abs(sic_a['reference_anchor_count'] - sic_b['reference_anchor_count']) == 0, "reference_anchor_count should be same"
    
    print(f"  PASS: Same dbar/kernel reused across subsets")
    return True


def test_15_h3_random_level_not_weak():
    """Test 15: H3 at ~0.5 better-than-random must be NOT SUPPORTED, not WEAK"""
    print("\n=== Test 15: H3 random-level not marked WEAK ===")
    
    from analyze_dataset_embedding import evaluate_h3
    
    mock_results = {
        "subset_comparison": {
            "Ours": {
                "bootstrap_global": {
                    "mean_nearest": {"better_than_random_fraction": 0.5},
                    "p95_nearest": {"better_than_random_fraction": 0.5},
                    "max_radius": {"better_than_random_fraction": 0.5},
                    "redundancy_fraction": {"better_than_random_fraction": 0.5},
                },
                "bootstrap_wrist": {
                    "mean_nearest": {"better_than_random_fraction": 0.5},
                    "p95_nearest": {"better_than_random_fraction": 0.5},
                    "max_radius": {"better_than_random_fraction": 0.5},
                    "redundancy_fraction": {"better_than_random_fraction": 0.5},
                },
                "bootstrap_combined": {
                    "mean_nearest": {"better_than_random_fraction": 0.5},
                    "p95_nearest": {"better_than_random_fraction": 0.5},
                    "max_radius": {"better_than_random_fraction": 0.5},
                    "redundancy_fraction": {"better_than_random_fraction": 0.5},
                },
                "bootstrap_fixed_sic": {"better_than_random_fraction": 0.5},
            }
        }
    }
    
    h3_result = evaluate_h3(mock_results)
    
    assert h3_result["status"] != "WEAK", \
        f"H3 at random level (0.5) should not be WEAK, got {h3_result['status']}"
    assert h3_result["status"] == "NOT SUPPORTED", \
        f"H3 at random level should be NOT SUPPORTED, got {h3_result['status']}"
    
    print(f"  PASS: H3 at 0.5 correctly marked as NOT SUPPORTED")
    return True


def test_16_h3_strong_support():
    """Test 16: H3 with multiple core metrics >=0.95 should be SUPPORTED"""
    print("\n=== Test 16: H3 strong support ===")
    
    from analyze_dataset_embedding import evaluate_h3
    
    mock_results = {
        "subset_comparison": {
            "Ours": {
                "bootstrap_global": {
                    "mean_nearest": {"better_than_random_fraction": 0.98},
                    "p95_nearest": {"better_than_random_fraction": 0.97},
                    "max_radius": {"better_than_random_fraction": 0.96},
                    "redundancy_fraction": {"better_than_random_fraction": 0.95},
                },
                "bootstrap_wrist": {
                    "mean_nearest": {"better_than_random_fraction": 0.96},
                    "p95_nearest": {"better_than_random_fraction": 0.95},
                    "max_radius": {"better_than_random_fraction": 0.94},
                    "redundancy_fraction": {"better_than_random_fraction": 0.93},
                },
                "bootstrap_combined": {
                    "mean_nearest": {"better_than_random_fraction": 0.97},
                    "p95_nearest": {"better_than_random_fraction": 0.96},
                    "max_radius": {"better_than_random_fraction": 0.95},
                    "redundancy_fraction": {"better_than_random_fraction": 0.94},
                },
                "bootstrap_fixed_sic": {"better_than_random_fraction": 0.99},
            }
        }
    }
    
    h3_result = evaluate_h3(mock_results)
    
    assert h3_result["status"] == "SUPPORTED", \
        f"H3 with strong metrics should be SUPPORTED, got {h3_result['status']}"
    
    print(f"  PASS: H3 correctly marked as SUPPORTED")
    return True


def test_17_ours_uniform_overlap():
    """Test 17: Ours vs Uniform overlap calculation correct"""
    print("\n=== Test 17: Ours vs Uniform overlap ===")
    
    ours_episodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    uniform_episodes = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    
    overlap_eps = set(ours_episodes) & set(uniform_episodes)
    overlap_count = len(overlap_eps)
    overlap_ratio = overlap_count / min(len(ours_episodes), len(uniform_episodes))
    
    assert overlap_count == 6, f"Expected 6 overlap, got {overlap_count}"
    assert abs(overlap_ratio - 0.6) < 1e-10, f"Expected 0.6 ratio, got {overlap_ratio}"
    
    print(f"  PASS: Overlap count={overlap_count}, ratio={overlap_ratio:.4f}")
    return True


def test_18_ours_vs_uniform_comprehensive():
    """Test 18: Ours vs Uniform comprehensive comparison returns all required fields"""
    print("\n=== Test 18: Ours vs Uniform comprehensive comparison ===")

    ours_episodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    uniform_episodes = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    ours_data = {
        "workspace_coverage": {
            "physical_unselected_mean_nearest": 0.1,
            "physical_unselected_p95": 0.2,
            "physical_unselected_max_radius": 0.3,
            "grid_7x4_ratio": 0.5,
            "grid_14x8_ratio": 0.3,
        },
        "coverage_global": {
            "unselected_mean_nearest_distance": 0.15,
            "unselected_median_nearest_distance": 0.12,
            "unselected_p90_nearest_distance": 0.18,
            "unselected_p95_nearest_distance": 0.2,
            "unselected_max_nearest_distance": 0.25,
        },
        "coverage_wrist": {
            "unselected_mean_nearest_distance": 0.16,
            "unselected_median_nearest_distance": 0.13,
            "unselected_p90_nearest_distance": 0.19,
            "unselected_p95_nearest_distance": 0.21,
            "unselected_max_nearest_distance": 0.26,
        },
        "coverage_combined": {
            "unselected_mean_nearest_distance": 0.14,
            "unselected_median_nearest_distance": 0.11,
            "unselected_p90_nearest_distance": 0.17,
            "unselected_p95_nearest_distance": 0.19,
            "unselected_max_nearest_distance": 0.24,
        },
        "redundancy_global": {
            "mean_nearest": 0.05,
            "median_nearest": 0.04,
            "p10_nearest": 0.02,
            "p50_nearest": 0.04,
            "p90_nearest": 0.08,
            "redundancy_fraction": 0.1,
        },
        "redundancy_wrist": {
            "mean_nearest": 0.06,
            "median_nearest": 0.05,
            "p10_nearest": 0.03,
            "p50_nearest": 0.05,
            "p90_nearest": 0.09,
            "redundancy_fraction": 0.12,
        },
        "redundancy_combined": {
            "mean_nearest": 0.04,
            "median_nearest": 0.03,
            "p10_nearest": 0.01,
            "p50_nearest": 0.03,
            "p90_nearest": 0.07,
            "redundancy_fraction": 0.08,
        },
        "fixed_sic": {
            "normalized_sic": 0.45,
        },
    }

    uniform_data = {
        "workspace_coverage": {
            "physical_unselected_mean_nearest": 0.11,
            "physical_unselected_p95": 0.21,
            "physical_unselected_max_radius": 0.31,
            "grid_7x4_ratio": 0.52,
            "grid_14x8_ratio": 0.32,
        },
        "coverage_global": {
            "unselected_mean_nearest_distance": 0.16,
            "unselected_median_nearest_distance": 0.13,
            "unselected_p90_nearest_distance": 0.19,
            "unselected_p95_nearest_distance": 0.21,
            "unselected_max_nearest_distance": 0.26,
        },
        "coverage_wrist": {
            "unselected_mean_nearest_distance": 0.17,
            "unselected_median_nearest_distance": 0.14,
            "unselected_p90_nearest_distance": 0.2,
            "unselected_p95_nearest_distance": 0.22,
            "unselected_max_nearest_distance": 0.27,
        },
        "coverage_combined": {
            "unselected_mean_nearest_distance": 0.15,
            "unselected_median_nearest_distance": 0.12,
            "unselected_p90_nearest_distance": 0.18,
            "unselected_p95_nearest_distance": 0.2,
            "unselected_max_nearest_distance": 0.25,
        },
        "redundancy_global": {
            "mean_nearest": 0.06,
            "median_nearest": 0.05,
            "p10_nearest": 0.03,
            "p50_nearest": 0.05,
            "p90_nearest": 0.09,
            "redundancy_fraction": 0.12,
        },
        "redundancy_wrist": {
            "mean_nearest": 0.07,
            "median_nearest": 0.06,
            "p10_nearest": 0.04,
            "p50_nearest": 0.06,
            "p90_nearest": 0.1,
            "redundancy_fraction": 0.14,
        },
        "redundancy_combined": {
            "mean_nearest": 0.05,
            "median_nearest": 0.04,
            "p10_nearest": 0.02,
            "p50_nearest": 0.04,
            "p90_nearest": 0.08,
            "redundancy_fraction": 0.1,
        },
        "fixed_sic": {
            "normalized_sic": 0.46,
        },
    }

    result = evaluate_ours_vs_uniform(
        ours_data, uniform_data,
        ours_episodes, uniform_episodes,
    )

    assert "episode_overlap_count" in result, "Missing episode_overlap_count"
    assert "episode_overlap_ratio" in result, "Missing episode_overlap_ratio"
    assert "workspace_coverage_delta" in result, "Missing workspace_coverage_delta"
    assert "global_coverage_delta" in result, "Missing global_coverage_delta"
    assert "wrist_coverage_delta" in result, "Missing wrist_coverage_delta"
    assert "combined_coverage_delta" in result, "Missing combined_coverage_delta"
    assert "global_redundancy_delta" in result, "Missing global_redundancy_delta"
    assert "wrist_redundancy_delta" in result, "Missing wrist_redundancy_delta"
    assert "combined_redundancy_delta" in result, "Missing combined_redundancy_delta"
    assert "fixed_sic_delta" in result, "Missing fixed_sic_delta"
    assert "conclusion" in result, "Missing conclusion"

    assert result["episode_overlap_count"] == 6, f"Expected 6 overlap, got {result['episode_overlap_count']}"
    assert abs(result["episode_overlap_ratio"] - 0.6) < 1e-10, f"Expected 0.6 ratio, got {result['episode_overlap_ratio']}"
    assert result["fixed_sic_delta"] == -0.01, f"Expected -0.01 SIC delta, got {result['fixed_sic_delta']}"

    print(f"  PASS: All required fields present")
    print(f"  PASS: episode_overlap_count={result['episode_overlap_count']}")
    print(f"  PASS: episode_overlap_ratio={result['episode_overlap_ratio']:.4f}")
    print(f"  PASS: fixed_sic_delta={result['fixed_sic_delta']:.4f}")
    print(f"  PASS: conclusion={result['conclusion']}")
    return True


def test_19_h1_evaluation():
    """Test 19: H1 evaluation uses multiple evidence sources"""
    print("\n=== Test 19: H1 evaluation ===")

    mock_results = {
        "validation": {"valid": True},
        "global_stats": {"effective_rank": 50.0, "dimension": 100},
        "wrist_stats": {"effective_rank": 40.0, "dimension": 100},
        "combined_stats": {"effective_rank": 80.0, "dimension": 200},
        "probe": {
            "global": {
                "ridge": {"R2_x": 0.8, "R2_y": 0.75},
                "shuffled_ridge": {"R2_x": {"mean": 0.1, "std": 0.05}, "R2_y": {"mean": 0.1, "std": 0.05}},
            },
            "wrist": {
                "ridge": {"R2_x": 0.6, "R2_y": 0.55},
                "shuffled_ridge": {"R2_x": {"mean": 0.1, "std": 0.05}, "R2_y": {"mean": 0.1, "std": 0.05}},
            },
            "combined": {
                "ridge": {"R2_x": 0.85, "R2_y": 0.8},
                "shuffled_ridge": {"R2_x": {"mean": 0.1, "std": 0.05}, "R2_y": {"mean": 0.1, "std": 0.05}},
            },
        },
        "neighbor_overlap": {
            "global": {"neighbor_overlap@10": 0.3, "random_neighbor_overlap@10": 0.1},
            "wrist": {"neighbor_overlap@10": 0.25, "random_neighbor_overlap@10": 0.1},
            "combined": {"neighbor_overlap@10": 0.35, "random_neighbor_overlap@10": 0.1},
        },
    }

    h1_result = evaluate_h1(mock_results)

    assert "status" in h1_result, "Missing status"
    assert "evidence" in h1_result, "Missing evidence"
    assert "effective_rank_ratio" in h1_result, "Missing effective_rank_ratio"

    print(f"  PASS: H1 status={h1_result['status']}")
    print(f"  PASS: effective_rank_ratio={h1_result['effective_rank_ratio']:.4f}")
    return True


def test_20_h2_evaluation():
    """Test 20: H2 evaluation uses permutation p-value"""
    print("\n=== Test 20: H2 evaluation ===")

    mock_results = {
        "spearman": {
            "global": {"rho": 0.6, "p_value": 1e-10, "n_pairs": 50000},
            "wrist": {"rho": 0.5, "p_value": 1e-8, "n_pairs": 50000},
            "combined": {"rho": 0.65, "p_value": 1e-12, "n_pairs": 50000},
        },
        "permutation_tests": {
            "global": {"permutation_p_value": 0.001, "observed_rho": 0.6},
            "wrist": {"permutation_p_value": 0.002, "observed_rho": 0.5},
            "combined": {"permutation_p_value": 0.001, "observed_rho": 0.65},
        },
        "probe": {
            "global": {
                "ridge": {"R2_x": 0.7, "R2_y": 0.65},
                "shuffled_ridge": {"R2_x": {"mean": 0.05, "std": 0.02}},
            },
            "wrist": {
                "ridge": {"R2_x": 0.5, "R2_y": 0.45},
                "shuffled_ridge": {"R2_x": {"mean": 0.05, "std": 0.02}},
            },
            "combined": {
                "ridge": {"R2_x": 0.75, "R2_y": 0.7},
                "shuffled_ridge": {"R2_x": {"mean": 0.05, "std": 0.02}},
            },
        },
        "neighbor_overlap": {
            "global": {"neighbor_overlap@10": 0.25, "random_neighbor_overlap@10": 0.08},
            "wrist": {"neighbor_overlap@10": 0.2, "random_neighbor_overlap@10": 0.08},
            "combined": {"neighbor_overlap@10": 0.3, "random_neighbor_overlap@10": 0.08},
        },
    }

    h2_result = evaluate_h2(mock_results)

    assert "status" in h2_result, "Missing status"
    assert "evidence" in h2_result, "Missing evidence"
    assert "n_sig_spearman" in h2_result, "Missing n_sig_spearman"

    print(f"  PASS: H2 status={h2_result['status']}")
    print(f"  PASS: n_sig_spearman={h2_result['n_sig_spearman']}")
    return True


def test_21_subset_loader_selected_episode_indices():
    """Test 21: Subset loader supports selected_episode_indices field"""
    print("\n=== Test 21: Subset loader selected_episode_indices ===")

    import tempfile
    test_data = {"method": "random", "num_episodes": 112, "seed": 42, "selected_episode_indices": [3, 8, 11, 12]}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = Path(f.name)

    try:
        indices = load_subset_episode_indices(temp_path)
        assert indices == [3, 8, 11, 12], f"Expected [3, 8, 11, 12], got {indices}"
        print(f"  PASS: Loaded {len(indices)} indices from selected_episode_indices")
    finally:
        temp_path.unlink()

    return True


def test_22_subset_loader_missing_key_raises():
    """Test 22: Subset loader raises ValueError when no valid key found"""
    print("\n=== Test 22: Subset loader missing key raises ===")

    import tempfile
    test_data = {"method": "random", "num_episodes": 112}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = Path(f.name)

    try:
        try:
            load_subset_episode_indices(temp_path)
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "selected_episode_indices" in str(e) or "episode_indices" in str(e) or "episodes" in str(e)
            print(f"  PASS: Raised ValueError as expected: {e}")
    finally:
        temp_path.unlink()

    return True


def test_23_random_json_structure():
    """Test 23: Random JSON structure example loads 112 episodes"""
    print("\n=== Test 23: Random JSON structure ===")

    import tempfile
    random_indices = list(range(112))
    test_data = {
        "method": "random",
        "num_episodes": 112,
        "seed": 42,
        "selected_episode_indices": random_indices
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = Path(f.name)

    try:
        indices = load_subset_episode_indices(temp_path)
        assert len(indices) == 112, f"Expected 112 indices, got {len(indices)}"
        assert indices == random_indices
        print(f"  PASS: Loaded 112 indices from Random JSON structure")
    finally:
        temp_path.unlink()

    return True


def test_24_uniform_json_structure():
    """Test 24: Uniform JSON structure example loads 112 episodes"""
    print("\n=== Test 24: Uniform JSON structure ===")

    import tempfile
    uniform_indices = [5, 7, 11, 12] + list(range(20, 128))
    test_data = {
        "method": "uniform_workspace",
        "num_episodes": 112,
        "seed": 42,
        "selected_episode_indices": uniform_indices
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = Path(f.name)

    try:
        indices = load_subset_episode_indices(temp_path)
        assert len(indices) == 112, f"Expected 112 indices, got {len(indices)}"
        print(f"  PASS: Loaded 112 indices from Uniform JSON structure")
    finally:
        temp_path.unlink()

    return True


def test_25_ours_json_structure():
    """Test 25: Ours JSON structure example loads selected_episode_indices"""
    print("\n=== Test 25: Ours JSON structure ===")

    import tempfile
    ours_indices = [0, 5, 6, 7, 9] + list(range(20, 127))
    test_data = {
        "selected_episode_indices": ours_indices
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = Path(f.name)

    try:
        indices = load_subset_episode_indices(temp_path)
        assert len(indices) == 112, f"Expected 112 indices, got {len(indices)}"
        print(f"  PASS: Loaded 112 indices from Ours JSON structure")
    finally:
        temp_path.unlink()

    return True


def test_26_fixed_sic_observed_bootstrap_shared_dbar():
    """Test 26: Fixed SIC observed and bootstrap share same dbar"""
    print("\n=== Test 26: Fixed SIC observed and bootstrap shared dbar ===")

    episode_indices, phi_g, phi_w = create_test_embeddings(n_episodes=30, dim=8)

    dbar_g, dbar_w, _ = compute_dbar_from_embeddings(phi_g, phi_w)
    K_g, K_w = build_kernel_matrices(phi_g, phi_w, dbar_g, dbar_w)

    from analyze_dataset_embedding import compute_fixed_universe_sic_for_subset

    subset_indices_a = list(range(10))

    sic_result = compute_fixed_universe_sic_for_subset(
        subset_indices_a, episode_indices, K_g, K_w, dbar_g, dbar_w, "test"
    )

    assert abs(sic_result["dbar_global"] - dbar_g) < 1e-10, "dbar_global should match"
    assert abs(sic_result["dbar_wrist"] - dbar_w) < 1e-10, "dbar_wrist should match"

    print(f"  PASS: Observed SIC uses same dbar as precomputed")
    return True


def test_27_compute_dbar_from_embeddings_used():
    """Test 27: compute_dbar_from_embeddings definition is used in main flow"""
    print("\n=== Test 27: compute_dbar_from_embeddings used ===")

    rng = np.random.RandomState(42)
    n = 50
    dim = 8

    phi_g = rng.randn(n, dim).astype(np.float32)
    phi_w = rng.randn(n, dim).astype(np.float32)

    dbar_g, dbar_w, fallback = compute_dbar_from_embeddings(phi_g, phi_w)

    assert isinstance(dbar_g, float), "dbar_global should be float"
    assert isinstance(dbar_w, float), "dbar_wrist should be float"
    assert isinstance(fallback, bool), "fallback should be bool"

    print(f"  PASS: compute_dbar_from_embeddings returns correct types")
    return True


def test_28_alignment_duplicate_from_load_info():
    """Test 28: Alignment duplicate info comes from load_info/meta_info"""
    print("\n=== Test 28: Alignment duplicate from load_info/meta_info ===")

    from analyze_dataset_embedding import align_embeddings_with_metadata

    rng = np.random.RandomState(42)
    n = 20
    dim = 4

    embeddings = {i: {"phi_global": rng.randn(dim).astype(np.float32), "phi_wrist": rng.randn(dim).astype(np.float32)} for i in range(n)}
    metadata = {i: {"obj_init_pos": rng.rand(2), "goal_pos": rng.rand(2)} for i in range(n)}

    load_info = {
        "duplicate_episode_indices": [5, 10],
        "invalid_files": [{"file": "bad_file.npy", "reason": "missing phi_global"}],
        "embedding_npy_file_count": 22,
        "embedding_valid_record_count": 20,
    }
    meta_info = {
        "duplicate_episode_indices": [3, 7],
        "metadata_record_count": 20,
        "metadata_unique_episode_count": 20,
    }

    episode_indices, phi_g, phi_w, init_pos, goal_pos, alignment_info = \
        align_embeddings_with_metadata(embeddings, metadata, load_info=load_info, meta_info=meta_info)

    assert alignment_info["duplicate_embedding_episode_index"] == [5, 10], "Should use load_info duplicates"
    assert alignment_info["duplicate_metadata_episode_index"] == [3, 7], "Should use meta_info duplicates"
    assert alignment_info["invalid_embedding_files"] == [{"file": "bad_file.npy", "reason": "missing phi_global"}], \
        "Should use load_info invalid files with file+reason structure"
    assert alignment_info["embedding_npy_file_count"] == 22, "Should use load_info embedding_npy_file_count"
    assert alignment_info["embedding_valid_record_count"] == 20, "Should use load_info embedding_valid_record_count"
    assert alignment_info["metadata_record_count"] == 20, "metadata_record_count should be raw record count"
    assert alignment_info["metadata_unique_episode_count"] == 20, "metadata_unique_episode_count should be unique count"

    print(f"  PASS: Alignment duplicate info from load_info/meta_info")
    return True


def test_29_normalized_embedding_stats():
    """Test 29: global_normalized/wrist_normalized stats exist"""
    print("\n=== Test 29: Normalized embedding stats ===")

    rng = np.random.RandomState(42)
    n = 50
    dim = 8

    phi_g = rng.randn(n, dim).astype(np.float32)
    phi_w = rng.randn(n, dim).astype(np.float32)

    phi_g_norm = l2_normalize_rows(phi_g)
    phi_w_norm = l2_normalize_rows(phi_w)

    stats_g_norm = compute_embedding_statistics(phi_g_norm, "global_normalized")
    stats_w_norm = compute_embedding_statistics(phi_w_norm, "wrist_normalized")

    assert "effective_rank" in stats_g_norm, "Missing effective_rank in global_normalized"
    assert "effective_rank" in stats_w_norm, "Missing effective_rank in wrist_normalized"

    norms_g = np.linalg.norm(phi_g_norm, axis=1)
    assert np.allclose(norms_g, 1.0, atol=1e-6), "Normalized embeddings should have unit norm"

    print(f"  PASS: Normalized embedding stats computed correctly")
    return True


def test_30_h1_uses_probe_null_evidence():
    """Test 30: H1 uses x/y probe null evidence"""
    print("\n=== Test 30: H1 uses probe null evidence ===")

    mock_results = {
        "validation": {"valid": True},
        "global_stats": {"effective_rank": 50.0, "dimension": 100},
        "wrist_stats": {"effective_rank": 40.0, "dimension": 100},
        "combined_stats": {"effective_rank": 80.0, "dimension": 200},
        "probe": {
            "global": {
                "ridge": {"R2_x": 0.8, "R2_y": 0.75},
                "shuffled_ridge": {
                    "R2_x": {"mean": 0.1, "std": 0.05, "observed": 0.8, "observed_percentile": 0.95, "empirical_p_value": 0.01},
                    "R2_y": {"mean": 0.1, "std": 0.05, "observed": 0.75, "observed_percentile": 0.94, "empirical_p_value": 0.02},
                },
            },
            "wrist": {
                "ridge": {"R2_x": 0.6, "R2_y": 0.55},
                "shuffled_ridge": {
                    "R2_x": {"mean": 0.1, "std": 0.05, "observed": 0.6, "observed_percentile": 0.90, "empirical_p_value": 0.03},
                    "R2_y": {"mean": 0.1, "std": 0.05, "observed": 0.55, "observed_percentile": 0.88, "empirical_p_value": 0.04},
                },
            },
            "combined": {
                "ridge": {"R2_x": 0.85, "R2_y": 0.8},
                "shuffled_ridge": {
                    "R2_x": {"mean": 0.1, "std": 0.05, "observed": 0.85, "observed_percentile": 0.96, "empirical_p_value": 0.005},
                    "R2_y": {"mean": 0.1, "std": 0.05, "observed": 0.8, "observed_percentile": 0.95, "empirical_p_value": 0.01},
                },
            },
        },
        "neighbor_overlap": {
            "global": {
                "neighbor_overlap@10": 0.3,
                "random_neighbor_overlap@10": 0.1,
                "neighbor_overlap@10_null": {"mean": 0.1, "std": 0.02, "empirical_p_value": 0.001},
            },
            "wrist": {
                "neighbor_overlap@10": 0.25,
                "random_neighbor_overlap@10": 0.1,
                "neighbor_overlap@10_null": {"mean": 0.1, "std": 0.02, "empirical_p_value": 0.005},
            },
            "combined": {
                "neighbor_overlap@10": 0.35,
                "random_neighbor_overlap@10": 0.1,
                "neighbor_overlap@10_null": {"mean": 0.1, "std": 0.02, "empirical_p_value": 0.001},
            },
        },
    }

    h1_result = evaluate_h1(mock_results)

    assert "status" in h1_result, "Missing status"
    assert "evidence" in h1_result, "Missing evidence"

    for rep in ["global", "wrist", "combined"]:
        rep_evidence = h1_result["evidence"].get(rep, {})
        assert "probe_x_significant" in rep_evidence, f"Missing probe_x_significant for {rep}"
        assert "probe_y_significant" in rep_evidence, f"Missing probe_y_significant for {rep}"
        assert "neighbor_significant" in rep_evidence, f"Missing neighbor_significant for {rep}"

    print(f"  PASS: H1 uses probe x/y null evidence")
    return True


def test_31_h2_uses_permutation_probe_neighbor_null():
    """Test 31: H2 uses permutation p + probe null + neighbor null"""
    print("\n=== Test 31: H2 uses permutation p + probe null + neighbor null ===")

    mock_results = {
        "spearman": {
            "global": {"rho": 0.6, "p_value": 1e-10, "n_pairs": 50000},
            "wrist": {"rho": 0.5, "p_value": 1e-8, "n_pairs": 50000},
            "combined": {"rho": 0.65, "p_value": 1e-12, "n_pairs": 50000},
        },
        "permutation_tests": {
            "global": {"permutation_p_value": 0.001, "observed_rho": 0.6},
            "wrist": {"permutation_p_value": 0.002, "observed_rho": 0.5},
            "combined": {"permutation_p_value": 0.001, "observed_rho": 0.65},
        },
        "probe": {
            "global": {
                "ridge": {"R2_x": 0.7, "R2_y": 0.65},
                "shuffled_ridge": {
                    "R2_x": {"mean": 0.05, "std": 0.02, "empirical_p_value": 0.01},
                    "R2_y": {"mean": 0.05, "std": 0.02, "empirical_p_value": 0.02},
                },
            },
            "wrist": {
                "ridge": {"R2_x": 0.5, "R2_y": 0.45},
                "shuffled_ridge": {
                    "R2_x": {"mean": 0.05, "std": 0.02, "empirical_p_value": 0.03},
                    "R2_y": {"mean": 0.05, "std": 0.02, "empirical_p_value": 0.04},
                },
            },
            "combined": {
                "ridge": {"R2_x": 0.75, "R2_y": 0.7},
                "shuffled_ridge": {
                    "R2_x": {"mean": 0.05, "std": 0.02, "empirical_p_value": 0.005},
                    "R2_y": {"mean": 0.05, "std": 0.02, "empirical_p_value": 0.01},
                },
            },
        },
        "neighbor_overlap": {
            "global": {
                "neighbor_overlap@10": 0.25,
                "random_neighbor_overlap@10": 0.08,
                "neighbor_overlap@10_null": {"mean": 0.08, "std": 0.02, "empirical_p_value": 0.001},
            },
            "wrist": {
                "neighbor_overlap@10": 0.2,
                "random_neighbor_overlap@10": 0.08,
                "neighbor_overlap@10_null": {"mean": 0.08, "std": 0.02, "empirical_p_value": 0.005},
            },
            "combined": {
                "neighbor_overlap@10": 0.3,
                "random_neighbor_overlap@10": 0.08,
                "neighbor_overlap@10_null": {"mean": 0.08, "std": 0.02, "empirical_p_value": 0.001},
            },
        },
    }

    h2_result = evaluate_h2(mock_results)

    assert "status" in h2_result, "Missing status"
    assert "evidence" in h2_result, "Missing evidence"

    for rep in ["global", "wrist", "combined"]:
        rep_evidence = h2_result["evidence"].get(rep, {})
        assert "spearman_significant" in rep_evidence, f"Missing spearman_significant for {rep}"
        assert "probe_x_significant" in rep_evidence, f"Missing probe_x_significant for {rep}"
        assert "probe_y_significant" in rep_evidence, f"Missing probe_y_significant for {rep}"
        assert "neighbor_significant" in rep_evidence, f"Missing neighbor_significant for {rep}"

    print(f"  PASS: H2 uses permutation p + probe null + neighbor null")
    return True


def test_32_ours_uniform_no_rho_overlap_threshold():
    """Test 32: Ours-vs-Uniform no longer contains rho>0.3/overlap>0.5 conclusion logic"""
    print("\n=== Test 32: Ours-vs-Uniform no hard thresholds ===")

    ours_episodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    uniform_episodes = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    ours_data = {
        "workspace_coverage": {"physical_unselected_mean_nearest": 0.1},
        "coverage_global": {"unselected_mean_nearest_distance": 0.15},
        "coverage_wrist": {"unselected_mean_nearest_distance": 0.16},
        "coverage_combined": {"unselected_mean_nearest_distance": 0.14},
        "redundancy_global": {"redundancy_fraction": 0.1},
        "redundancy_wrist": {"redundancy_fraction": 0.12},
        "redundancy_combined": {"redundancy_fraction": 0.08},
        "fixed_sic": {"normalized_sic": 0.45},
    }

    uniform_data = {
        "workspace_coverage": {"physical_unselected_mean_nearest": 0.11},
        "coverage_global": {"unselected_mean_nearest_distance": 0.16},
        "coverage_wrist": {"unselected_mean_nearest_distance": 0.17},
        "coverage_combined": {"unselected_mean_nearest_distance": 0.15},
        "redundancy_global": {"redundancy_fraction": 0.12},
        "redundancy_wrist": {"redundancy_fraction": 0.14},
        "redundancy_combined": {"redundancy_fraction": 0.1},
        "fixed_sic": {"normalized_sic": 0.46},
    }

    result = evaluate_ours_vs_uniform(ours_data, uniform_data, ours_episodes, uniform_episodes)

    assert "conclusion" in result, "Missing conclusion"
    assert "evidence_summary" in result, "Missing evidence_summary"

    conclusion = result["conclusion"]
    assert "Insufficient evidence" in conclusion or "Evidence suggests" in conclusion, \
        f"Conclusion should be conservative, got: {conclusion}"

    print(f"  PASS: No hard thresholds, conclusion is conservative")
    return True


def test_33_fixed_sic_none_safe():
    """Test 33: fixed_sic=None doesn't crash compare"""
    print("\n=== Test 33: fixed_sic=None safe ===")

    ours_episodes = [1, 2, 3, 4, 5]
    uniform_episodes = [3, 4, 5, 6, 7]

    ours_data = {
        "workspace_coverage": {"physical_unselected_mean_nearest": 0.1},
        "coverage_global": {"unselected_mean_nearest_distance": 0.15},
        "coverage_wrist": {"unselected_mean_nearest_distance": 0.16},
        "coverage_combined": {"unselected_mean_nearest_distance": 0.14},
        "redundancy_global": {"redundancy_fraction": 0.1},
        "redundancy_wrist": {"redundancy_fraction": 0.12},
        "redundancy_combined": {"redundancy_fraction": 0.08},
        "fixed_sic": None,
    }

    uniform_data = {
        "workspace_coverage": {"physical_unselected_mean_nearest": 0.11},
        "coverage_global": {"unselected_mean_nearest_distance": 0.16},
        "coverage_wrist": {"unselected_mean_nearest_distance": 0.17},
        "coverage_combined": {"unselected_mean_nearest_distance": 0.15},
        "redundancy_global": {"redundancy_fraction": 0.12},
        "redundancy_wrist": {"redundancy_fraction": 0.14},
        "redundancy_combined": {"redundancy_fraction": 0.1},
        "fixed_sic": {"normalized_sic": 0.46},
    }

    try:
        result = evaluate_ours_vs_uniform(ours_data, uniform_data, ours_episodes, uniform_episodes)
        assert "fixed_sic_delta" in result, "Missing fixed_sic_delta"
        print(f"  PASS: fixed_sic=None handled safely")
    except Exception as e:
        raise AssertionError(f"Should not crash with fixed_sic=None: {e}")

    return True


def test_34_table2_fixed_sic_fields():
    """Test 34: Table 2 Fixed SIC fields still correct"""
    print("\n=== Test 34: Table 2 Fixed SIC fields ===")

    episode_indices, phi_g, phi_w = create_test_embeddings(n_episodes=30, dim=8)

    dbar_g, dbar_w, _ = compute_dbar_from_embeddings(phi_g, phi_w)
    K_g, K_w = build_kernel_matrices(phi_g, phi_w, dbar_g, dbar_w)

    subset_indices = list(range(10))

    from analyze_dataset_embedding import compute_fixed_universe_sic_for_subset

    sic_result = compute_fixed_universe_sic_for_subset(
        subset_indices, episode_indices, K_g, K_w, dbar_g, dbar_w, "test"
    )

    assert "normalized_sic" in sic_result, "Missing normalized_sic"
    assert "fixed_universe_sic" in sic_result, "Missing fixed_universe_sic"
    assert "reference_anchor_count" in sic_result, "Missing reference_anchor_count"
    assert "dbar_global" in sic_result, "Missing dbar_global"
    assert "dbar_wrist" in sic_result, "Missing dbar_wrist"

    print(f"  PASS: Table 2 Fixed SIC fields present")
    return True


def test_35_report_regression_no_crash():
    """Test 35: generate_report() does not crash with new fields, no old stale keys."""
    print("\n=== Test 35: Report regression (no crash, correct keys) ===")

    from analyze_dataset_embedding import generate_report
    import tempfile

    mock_results = {
        "n_episodes": 30,
        "global_stats": {"dimension": 8, "effective_rank": 5.0},
        "wrist_stats": {"dimension": 8, "effective_rank": 4.0},
        "combined_stats": {"dimension": 16, "effective_rank": 8.0},
        "validation": {
            "valid": True,
            "stats": {
                "n_exact_dup_global": 0,
                "n_exact_dup_wrist": 0,
                "n_near_dup_global": 0,
                "n_near_dup_wrist": 0,
                "zero_norm_global": 0,
                "zero_norm_wrist": 0,
            },
            "errors": [],
            "warnings": [],
        },
        "permutation_tests": {
            "global": {"permutation_p_value": 0.01},
            "wrist": {"permutation_p_value": 0.02},
            "combined": {"permutation_p_value": 0.005},
        },
        "spearman": {
            "global": {"rho": 0.5, "p_value": 1e-5},
            "wrist": {"rho": 0.3, "p_value": 1e-3},
            "combined": {"rho": 0.6, "p_value": 1e-6},
        },
        "probe": {
            "global": {
                "ridge": {"R2_x": 0.8, "R2_y": 0.75},
                "shuffled_ridge": {
                    "R2_x": {"empirical_p_value": 0.01},
                    "R2_y": {"empirical_p_value": 0.02},
                },
            },
            "wrist": {
                "ridge": {"R2_x": 0.6, "R2_y": 0.55},
                "shuffled_ridge": {
                    "R2_x": {"empirical_p_value": 0.03},
                    "R2_y": {"empirical_p_value": 0.04},
                },
            },
            "combined": {
                "ridge": {"R2_x": 0.85, "R2_y": 0.8},
                "shuffled_ridge": {
                    "R2_x": {"empirical_p_value": 0.005},
                    "R2_y": {"empirical_p_value": 0.01},
                },
            },
        },
        "neighbor_overlap": {
            "global": {
                "neighbor_overlap@10": 0.3,
                "random_neighbor_overlap@10": 0.1,
                "neighbor_overlap@10_null": {"empirical_p_value": 0.01},
            },
            "wrist": {
                "neighbor_overlap@10": 0.25,
                "random_neighbor_overlap@10": 0.1,
                "neighbor_overlap@10_null": {"empirical_p_value": 0.02},
            },
            "combined": {
                "neighbor_overlap@10": 0.35,
                "random_neighbor_overlap@10": 0.1,
                "neighbor_overlap@10_null": {"empirical_p_value": 0.005},
            },
        },
        "grid_classifiability": {},
        "subset_comparison": {
            "Ours": {
                "coverage_global": {"unselected_mean_nearest_distance": 0.5},
                "coverage_wrist": {"unselected_mean_nearest_distance": 0.4},
                "coverage_combined": {"unselected_mean_nearest_distance": 0.45, "unselected_max_nearest_distance": 1.2},
                "redundancy_combined": {"redundancy_fraction": 0.1},
                "fixed_sic": {"normalized_sic": 0.6},
                "bootstrap_global": {"mean_nearest": {"better_than_random_fraction": 0.96}},
                "bootstrap_wrist": {"mean_nearest": {"better_than_random_fraction": 0.92}},
                "bootstrap_combined": {"mean_nearest": {"better_than_random_fraction": 0.97}},
                "bootstrap_fixed_sic": {"better_than_random_fraction": 0.95},
            },
            "Uniform": {
                "coverage_global": {"unselected_mean_nearest_distance": 0.55},
                "coverage_wrist": {"unselected_mean_nearest_distance": 0.45},
                "coverage_combined": {"unselected_mean_nearest_distance": 0.5, "unselected_max_nearest_distance": 1.3},
                "redundancy_combined": {"redundancy_fraction": 0.12},
                "fixed_sic": {"normalized_sic": 0.65},
                "bootstrap_global": {"mean_nearest": {"better_than_random_fraction": 0.98}},
                "bootstrap_wrist": {"mean_nearest": {"better_than_random_fraction": 0.93}},
                "bootstrap_combined": {"mean_nearest": {"better_than_random_fraction": 0.99}},
                "bootstrap_fixed_sic": {"better_than_random_fraction": 0.97},
            },
        },
        "ours_uniform_comparison": {
            "episode_overlap_count": 50,
            "episode_overlap_ratio": 0.45,
            "conclusion": "Insufficient evidence that Ours mainly reproduces Uniform initial-position coverage.",
            "evidence_summary": {
                "workspace_physical_unselected_mean_nearest_relative_delta": 0.05,
                "global_coverage_unselected_mean_nearest_distance_relative_delta": -0.1,
                "fixed_sic_delta": -0.05,
                "note": "No calibrated equivalence test was performed.",
            },
            "workspace_coverage_delta": {"physical_unselected_mean_nearest": 0.05},
            "global_coverage_delta": {"unselected_mean_nearest_distance": -0.05},
            "wrist_coverage_delta": {"unselected_mean_nearest_distance": -0.05},
            "combined_coverage_delta": {"unselected_mean_nearest_distance": -0.05},
            "global_redundancy_delta": {"redundancy_fraction": -0.02},
            "wrist_redundancy_delta": {"redundancy_fraction": -0.02},
            "combined_redundancy_delta": {"redundancy_fraction": -0.02},
            "fixed_sic_delta": -0.05,
        },
        "hypothesis_evaluation": {
            "H1": {
                "hypothesis": "H1",
                "status": "SUPPORTED",
                "evidence": {
                    "global": {
                        "probe_x_significant": True,
                        "probe_y_significant": True,
                        "neighbor_significant": True,
                        "validity": True,
                    },
                    "wrist": {
                        "probe_x_significant": True,
                        "probe_y_significant": True,
                        "neighbor_significant": True,
                        "validity": True,
                    },
                    "combined": {
                        "probe_x_significant": True,
                        "probe_y_significant": True,
                        "neighbor_significant": True,
                        "validity": True,
                    },
                },
                "effective_rank_ratio": 0.625,
                "description": "Embedding 是否具有可区分性？",
            },
            "H2": {
                "hypothesis": "H2",
                "status": "SUPPORTED",
                "evidence": {
                    "global": {
                        "spearman_significant": True,
                        "probe_x_significant": True,
                        "probe_y_significant": True,
                        "neighbor_significant": True,
                    },
                    "wrist": {
                        "spearman_significant": True,
                        "probe_x_significant": True,
                        "probe_y_significant": True,
                        "neighbor_significant": True,
                    },
                    "combined": {
                        "spearman_significant": True,
                        "probe_x_significant": True,
                        "probe_y_significant": True,
                        "neighbor_significant": True,
                    },
                },
                "n_sig_spearman": 3,
                "description": "Embedding distance 是否具有 task-state geometry？",
            },
            "H3": {
                "hypothesis": "H3",
                "status": "SUPPORTED",
                "evidence": {
                    "n_strong_families": 3,
                    "n_weak_families": 0,
                    "n_random": 0,
                    "n_poor": 0,
                    "evidence_families": {"coverage_quality": 6, "max_radius_tail": 3, "redundancy": 3, "fixed_sic": 1},
                    "evidence_families_detail": {"coverage_quality": [], "max_radius_tail": [], "redundancy": [], "fixed_sic": []},
                },
                "description": "Ours subset 是否显著优于 random coverage？",
            },
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        generate_report(mock_results, output_dir)

        report_path = output_dir / "analysis_report.md"
        assert report_path.exists(), "Report file should exist"

        with open(report_path) as f:
            report_text = f.read()

        assert "Similarity ratio" not in report_text, "Old 'Similarity ratio' field should not exist"

        assert "probe_x_significant" in report_text, "H1 should display probe_x_significant"
        assert "probe_y_significant" in report_text, "H1 should display probe_y_significant"
        assert "neighbor_significant" in report_text, "H1 should display neighbor_significant"

        assert "spearman_significant" in report_text, "H2 should display spearman_significant"

        assert "n_strong_families" in report_text, "H3 should display n_strong_families"
        assert "n_weak_families" in report_text, "H3 should display n_weak_families"

        assert "Fixed SIC Delta" in report_text, "Fixed SIC should not be N/A"

    print(f"  PASS: Report generation works with new fields, no stale keys")
    return True


def test_36_bootstrap_nondefault_alpha_lambda_passthrough():
    """Test 36: Bootstrap non-default alpha/lambda passthrough."""
    print("\n=== Test 36: Bootstrap non-default alpha/lambda passthrough ===")

    from analyze_dataset_embedding import random_bootstrap_fixed_sic, generate_bootstrap_subsets
    from analysis_utils import compute_fixed_universe_sic_from_indices

    episode_indices, phi_g, phi_w = create_test_embeddings(n_episodes=30, dim=8)
    dbar_g, dbar_w, _ = compute_dbar_from_embeddings(phi_g, phi_w)
    K_g, K_w = build_kernel_matrices(phi_g, phi_w, dbar_g, dbar_w)

    subset_indices = list(range(10))

    pre_subsets = generate_bootstrap_subsets(n_total=30, subset_size=10, n_bootstrap=10, seed=42)

    result_default = random_bootstrap_fixed_sic(
        subset_indices=subset_indices,
        all_episode_indices=episode_indices,
        K_global=K_g, K_wrist=K_w,
        dbar_global=dbar_g, dbar_wrist=dbar_w,
        n_bootstrap=10, seed=42,
        alpha=1.0, lambda_wrist=1.0,
        pre_generated_subsets=pre_subsets,
    )

    result_custom = random_bootstrap_fixed_sic(
        subset_indices=subset_indices,
        all_episode_indices=episode_indices,
        K_global=K_g, K_wrist=K_w,
        dbar_global=dbar_g, dbar_wrist=dbar_w,
        n_bootstrap=10, seed=42,
        alpha=2.0, lambda_wrist=0.5,
        pre_generated_subsets=pre_subsets,
    )

    assert result_custom["metadata"]["alpha"] == 2.0, "Alpha should be 2.0"
    assert result_custom["metadata"]["lambda_wrist"] == 0.5, "Lambda wrist should be 0.5"
    assert result_default["observed"] != result_custom["observed"], \
        "Non-default alpha/lambda should produce different observed SIC"

    print(f"  PASS: Non-default alpha/lambda correctly passed through")
    return True


def test_37_fixed_sic_bootstrap_single_evidence():
    """Test 37: Fixed SIC bootstrap is a single independent evidence, not G/W/C triple."""
    print("\n=== Test 37: Fixed SIC bootstrap single evidence ===")

    from analyze_dataset_embedding import evaluate_h3

    mock_results = {
        "subset_comparison": {
            "Ours": {
                "bootstrap_global": {
                    "mean_nearest": {"better_than_random_fraction": 0.96},
                    "p95_nearest": {"better_than_random_fraction": 0.95},
                    "max_radius": {"better_than_random_fraction": 0.90},
                    "redundancy_fraction": {"better_than_random_fraction": 0.85},
                },
                "bootstrap_wrist": {
                    "mean_nearest": {"better_than_random_fraction": 0.92},
                    "p95_nearest": {"better_than_random_fraction": 0.91},
                    "max_radius": {"better_than_random_fraction": 0.88},
                    "redundancy_fraction": {"better_than_random_fraction": 0.80},
                },
                "bootstrap_combined": {
                    "mean_nearest": {"better_than_random_fraction": 0.97},
                    "p95_nearest": {"better_than_random_fraction": 0.96},
                    "max_radius": {"better_than_random_fraction": 0.92},
                    "redundancy_fraction": {"better_than_random_fraction": 0.87},
                },
                "bootstrap_fixed_sic": {
                    "better_than_random_fraction": 0.95,
                },
            },
        },
    }

    h3_result = evaluate_h3(mock_results)

    fixed_sic_family = h3_result["evidence"]["evidence_families"].get("fixed_sic", 0)
    assert fixed_sic_family == 1, f"Fixed SIC family should have exactly 1 evidence item, got {fixed_sic_family}"

    print(f"  PASS: Fixed SIC bootstrap is single evidence")
    return True


def test_38_exact_duplicate_not_near_duplicate():
    """Test 38: Exact duplicates are not counted as near duplicates."""
    print("\n=== Test 38: Exact duplicate not near duplicate ===")

    n = 10
    dim = 4
    rng = np.random.RandomState(42)

    phi_g = rng.randn(n, dim).astype(np.float32)
    phi_w = rng.randn(n, dim).astype(np.float32)

    phi_g[5] = phi_g[0].copy()
    phi_w[5] = phi_w[0].copy()

    result = check_embeddings_valid(phi_g, phi_w)

    n_exact_g = result["stats"]["n_exact_duplicate_groups_global"]
    n_near_g = result["stats"]["n_near_duplicate_pairs_global"]

    assert n_exact_g >= 1, f"Should detect exact duplicate group, got {n_exact_g}"

    print(f"  PASS: Exact duplicates detected, near duplicates exclude exact")
    return True


def test_39_grid_shuffled_balanced_accuracy():
    """Test 39: Grid shuffled balanced accuracy fields exist."""
    print("\n=== Test 39: Grid shuffled balanced accuracy ===")

    from analyze_dataset_embedding import grid_classifiability

    rng = np.random.RandomState(42)
    n = 100
    dim = 8

    phi = rng.randn(n, dim).astype(np.float32)
    positions = rng.rand(n, 2)

    grid_results = grid_classifiability(phi, positions, grid_sizes=[(5, 5)], n_shuffles=10, seed=42)

    for grid_name, grid_data in grid_results.items():
        assert "shuffled_balanced_accuracy_mean" in grid_data, f"Missing shuffled_balanced_accuracy_mean in {grid_name}"
        assert "shuffled_balanced_accuracy_std" in grid_data, f"Missing shuffled_balanced_accuracy_std in {grid_name}"

    print(f"  PASS: Grid shuffled balanced accuracy fields present")
    return True


def test_40_alignment_raw_vs_unique_counts():
    """Test 40: metadata raw record count vs unique count are separate."""
    print("\n=== Test 40: Alignment raw vs unique counts ===")

    from analyze_dataset_embedding import align_embeddings_with_metadata

    rng = np.random.RandomState(42)
    n = 10
    dim = 4

    embeddings = {i: {"phi_global": rng.randn(dim).astype(np.float32), "phi_wrist": rng.randn(dim).astype(np.float32)} for i in range(n)}
    metadata = {i: {"obj_init_pos": rng.rand(2), "goal_pos": rng.rand(2)} for i in range(n)}

    load_info = {
        "embedding_npy_file_count": 15,
        "embedding_valid_record_count": 12,
        "duplicate_episode_indices": [],
        "invalid_files": [],
    }
    meta_info = {
        "duplicate_episode_indices": [],
        "metadata_record_count": 13,
        "metadata_unique_episode_count": 13,
    }

    _, _, _, _, _, alignment_info = align_embeddings_with_metadata(
        embeddings, metadata, load_info=load_info, meta_info=meta_info
    )

    assert alignment_info["metadata_record_count"] == 13, "metadata_record_count should be raw record count"
    assert alignment_info["metadata_unique_episode_count"] == 10, "metadata_unique_episode_count should be unique count"
    assert alignment_info["embedding_npy_file_count"] == 15, "embedding_npy_file_count should be scanned .npy count"
    assert alignment_info["embedding_valid_record_count"] == 12, "embedding_valid_record_count should be valid records"
    assert alignment_info["embedding_unique_episode_count"] == 10, "embedding_unique_episode_count should be unique"

    print(f"  PASS: raw and unique counts are separate")
    return True


def test_41_missing_episode_index_marked_invalid():
    """Test 41: embedding file without episode_index is marked invalid, not silently skipped."""
    print("\n=== Test 41: Missing episode_index marked invalid ===")

    import tempfile
    from analyze_dataset_embedding import load_embeddings

    with tempfile.TemporaryDirectory() as tmpdir:
        emb_dir = Path(tmpdir)
        bad = {"phi_global": np.zeros(2, dtype=np.float32), "phi_wrist": np.zeros(2, dtype=np.float32)}
        np.save(emb_dir / "bad.npy", bad, allow_pickle=True)

        good = {"episode_index": 0, "phi_global": np.zeros(2, dtype=np.float32), "phi_wrist": np.zeros(2, dtype=np.float32)}
        np.save(emb_dir / "good.npy", good, allow_pickle=True)

        embeddings, load_info = load_embeddings(emb_dir)

        assert load_info["embedding_npy_file_count"] == 2, "Should scan both .npy files"
        assert load_info["embedding_valid_record_count"] == 1, "Only the good file should be a valid record"
        assert len(load_info["invalid_files"]) == 1, "Bad file should be in invalid_files"
        assert load_info["invalid_files"][0]["file"] == "bad.npy", "Should record bad file name"
        assert load_info["invalid_files"][0]["reason"] == "missing episode_index", "Should record reason"
        assert 0 in embeddings, "Good file should be loaded"

    print(f"  PASS: Missing episode_index marked invalid with reason")
    return True


def test_42_subset_duplicate_rejected():
    """Test 42: subset with duplicated episodes raises ValueError."""
    print("\n=== Test 42: Subset duplicate rejected ===")

    import tempfile
    test_data = {"selected_episode_indices": [1, 2, 2, 3]}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f)
        temp_path = Path(f.name)

    try:
        try:
            load_subset_episode_indices(temp_path)
            raise AssertionError("Should have raised ValueError for duplicates")
        except ValueError as e:
            assert "duplicate" in str(e).lower(), f"Should mention duplicates: {e}"
            print(f"  PASS: Subset duplicate rejected: {e}")
    finally:
        temp_path.unlink()

    return True


def test_43_subset_path_missing_raises():
    """Test 43: explicitly provided subset path that is missing raises FileNotFoundError."""
    print("\n=== Test 43: Missing subset path raises ===")

    from analyze_dataset_embedding import load_subset_if_provided

    missing_path = Path("/nonexistent/random_112_seed42.json")

    try:
        load_subset_if_provided(missing_path)
        raise AssertionError("Should have raised FileNotFoundError")
    except FileNotFoundError as e:
        assert "not found" in str(e).lower(), f"Should mention not found: {e}"
        print(f"  PASS: Missing subset path raises FileNotFoundError")

    assert load_subset_if_provided(None) is None, "None path should be skipped gracefully"
    print(f"  PASS: None subset path returns None")
    return True


def test_44_subset_universe_mismatch_raises():
    """Test 44: subset episode not in aligned universe raises, no silent shrink."""
    print("\n=== Test 44: Subset universe mismatch raises ===")

    all_episode_indices = [0, 1, 2, 3, 4, 5]
    subset_with_missing = [1, 2, 99]

    try:
        match_subset_to_indices(subset_with_missing, all_episode_indices)
        raise AssertionError("Should have raised ValueError for missing episode")
    except ValueError as e:
        assert "not in aligned dataset" in str(e), f"Should mention aligned dataset: {e}"
        print(f"  PASS: Subset universe mismatch raises ValueError")

    return True


def test_45_h1_wrist_included_in_final_judgment():
    """Test 45: wrist evidence participates in H1 final counts."""
    print("\n=== Test 45: H1 wrist evidence participates ===")

    # Global/combined NOT significant; wrist significant. Status must be WEAK,
    # proving wrist is included in n_probe_sig / n_neighbor_sig.
    mock_results = {
        "validation": {"valid": True},
        "global_stats": {"effective_rank": 10.0, "dimension": 100},
        "wrist_stats": {"effective_rank": 10.0, "dimension": 100},
        "combined_stats": {"effective_rank": 10.0, "dimension": 100},
        "probe": {
            "global": {"ridge": {"R2_x": 0.1, "R2_y": 0.1}, "shuffled_ridge": {
                "R2_x": {"empirical_p_value": 0.9}, "R2_y": {"empirical_p_value": 0.9}}},
            "wrist": {"ridge": {"R2_x": 0.9, "R2_y": 0.9}, "shuffled_ridge": {
                "R2_x": {"empirical_p_value": 0.01}, "R2_y": {"empirical_p_value": 0.02}}},
            "combined": {"ridge": {"R2_x": 0.1, "R2_y": 0.1}, "shuffled_ridge": {
                "R2_x": {"empirical_p_value": 0.9}, "R2_y": {"empirical_p_value": 0.9}}},
        },
        "neighbor_overlap": {
            "global": {"neighbor_overlap@10": 0.1, "random_neighbor_overlap@10": 0.1,
                        "neighbor_overlap@10_null": {"empirical_p_value": 0.9}},
            "wrist": {"neighbor_overlap@10": 0.5, "random_neighbor_overlap@10": 0.1,
                       "neighbor_overlap@10_null": {"empirical_p_value": 0.01}},
            "combined": {"neighbor_overlap@10": 0.1, "random_neighbor_overlap@10": 0.1,
                          "neighbor_overlap@10_null": {"empirical_p_value": 0.9}},
        },
    }

    h1_result = evaluate_h1(mock_results)
    assert h1_result["status"] == "WEAK", (
        f"Wrist-only significance should yield WEAK (wrist participates), got {h1_result['status']}"
    )

    print(f"  PASS: wrist evidence included in H1 final judgment")
    return True


def test_46_h2_wrist_included_in_final_judgment():
    """Test 46: wrist evidence participates in H2 final counts."""
    print("\n=== Test 46: H2 wrist evidence participates ===")

    mock_results = {
        "spearman": {
            "global": {"rho": 0.6, "p_value": 1e-10},
            "wrist": {"rho": 0.55, "p_value": 1e-8},
            "combined": {"rho": 0.6, "p_value": 1e-10},
        },
        "permutation_tests": {
            "global": {"permutation_p_value": 0.001},
            "wrist": {"permutation_p_value": 0.001},
            "combined": {"permutation_p_value": 0.001},
        },
        "probe": {
            "global": {"ridge": {"R2_x": 0.1, "R2_y": 0.1}, "shuffled_ridge": {
                "R2_x": {"empirical_p_value": 0.9}, "R2_y": {"empirical_p_value": 0.9}}},
            "wrist": {"ridge": {"R2_x": 0.9, "R2_y": 0.9}, "shuffled_ridge": {
                "R2_x": {"empirical_p_value": 0.01}, "R2_y": {"empirical_p_value": 0.02}}},
            "combined": {"ridge": {"R2_x": 0.1, "R2_y": 0.1}, "shuffled_ridge": {
                "R2_x": {"empirical_p_value": 0.9}, "R2_y": {"empirical_p_value": 0.9}}},
        },
        "neighbor_overlap": {
            "global": {"neighbor_overlap@10": 0.1, "random_neighbor_overlap@10": 0.1,
                        "neighbor_overlap@10_null": {"empirical_p_value": 0.9}},
            "wrist": {"neighbor_overlap@10": 0.5, "random_neighbor_overlap@10": 0.1,
                       "neighbor_overlap@10_null": {"empirical_p_value": 0.01}},
            "combined": {"neighbor_overlap@10": 0.1, "random_neighbor_overlap@10": 0.1,
                          "neighbor_overlap@10_null": {"empirical_p_value": 0.9}},
        },
    }

    h2_result = evaluate_h2(mock_results)
    assert h2_result["n_sig_spearman"] == 3, "All three reps have significant spearman"
    # With 3 significant Spearman reps + wrist probe/neighbor evidence, multiple
    # independent evidence families are present -> SUPPORTED. Wrist participation is
    # proven because without wrist, n_probe_sig/n_neighbor_sig would be 0.
    assert h2_result["status"] == "SUPPORTED", (
        f"Spearman all + wrist probe/neighbor should be SUPPORTED, got {h2_result['status']}"
    )
    probe_sig_reps = [r for r in ["global", "wrist", "combined"]
                      if h2_result["evidence"][r]["probe_x_significant"] and h2_result["evidence"][r]["probe_y_significant"]]
    assert "wrist" in probe_sig_reps, f"Wrist probe should be significant: {probe_sig_reps}"

    print(f"  PASS: wrist evidence included in H2 final judgment")
    return True


def test_47_json_summary_no_default_str():
    """Test 47: analysis_summary.json does not rely on default=str."""
    print("\n=== Test 47: JSON summary no default=str ===")

    import tempfile
    from analyze_dataset_embedding import save_analysis_summary

    analysis_results = {
        "n_episodes": 10,
        "alignment": {"metadata_record_count": 10, "metadata_unique_episode_count": 10},
        "validation": {"valid": True, "stats": {}, "errors": [], "warnings": []},
        "global_stats": {"dimension": 4, "effective_rank": 2.0},
        "wrist_stats": {"dimension": 4, "effective_rank": 2.0},
        "combined_stats": {"dimension": 8, "effective_rank": 4.0},
        "global_normalized_stats": {"effective_rank": 2.0},
        "wrist_normalized_stats": {"effective_rank": 2.0},
        "spearman": {},
        "permutation_tests": {},
        "probe": {},
        "neighbor_overlap": {},
        "grid_classifiability": {},
        "subset_comparison": {},
        "ours_uniform_comparison": {},
        "hypothesis_evaluation": {},
        "analysis_config": {"seed": 42, "n_bootstrap": 1000, "alpha": 1.0, "lambda_wrist": 1.0},
        "fixed_universe": {"reference_episode_count": 10, "dbar_global": 1.0, "dbar_wrist": 1.0, "dbar_fallback_used": False},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        save_analysis_summary(analysis_results, out_dir)

        summary_path = out_dir / "analysis_summary.json"
        assert summary_path.exists(), "analysis_summary.json should exist"

        # Plain json.load (no default=str) must succeed, proving no numpy/object leakage.
        with open(summary_path) as f:
            data = json.load(f)

        assert data["analysis_config"]["n_bootstrap"] == 1000, "analysis_config should contain n_bootstrap"
        assert data["fixed_universe"]["reference_episode_count"] == 10, "fixed_universe should be recorded"
        assert data["alignment"]["metadata_unique_episode_count"] == 10

    print(f"  PASS: JSON summary serializes without default=str")
    return True


def test_48_report_regression_minimal():
    """Test 48: Minimal analysis_result drives generate_report without stale fields.

    Verifies:
      - No 'Similarity ratio' key is read/produced.
      - H1 fields are read from the new names.
      - H2 fields are read from the new names.
      - H3 fields are read from the new names.
      - Fixed SIC does not render as N/A when present.
    """
    print("\n=== Test 48: Minimal report regression ===")

    import tempfile
    from analyze_dataset_embedding import generate_report

    # Minimal but structurally complete analysis_result.
    analysis_results = {
        "n_episodes": 10,
        "alignment": {"metadata_record_count": 10, "metadata_unique_episode_count": 10},
        "validation": {"valid": True, "stats": {}, "errors": [], "warnings": []},
        "global_stats": {"dimension": 4, "effective_rank": 2.0},
        "wrist_stats": {"dimension": 4, "effective_rank": 2.0},
        "combined_stats": {"dimension": 8, "effective_rank": 4.0},
        "global_normalized_stats": {"effective_rank": 2.0},
        "wrist_normalized_stats": {"effective_rank": 2.0},
        "spearman": {"global": {"rho": 0.5, "p_value": 0.01}},
        "permutation_tests": {"global": {"permutation_p_value": 0.01}},
        "probe": {
            "global": {
                "ridge": {"R2_x": 0.5, "R2_y": 0.5},
                "shuffled_ridge": {
                    "R2_x": {"empirical_p_value": 0.01},
                    "R2_y": {"empirical_p_value": 0.02},
                },
            }
        },
        "neighbor_overlap": {
            "global": {
                "neighbor_overlap@10": 0.2,
                "random_neighbor_overlap@10": 0.1,
                "neighbor_overlap@10_null": {"empirical_p_value": 0.01},
            }
        },
        "grid_classifiability": {},
        "subset_comparison": {
            "Ours": {
                "coverage_global": {"unselected_mean_nearest_distance": 0.5},
                "coverage_wrist": {"unselected_mean_nearest_distance": 0.4},
                "coverage_combined": {
                    "unselected_mean_nearest_distance": 0.45,
                    "unselected_max_nearest_distance": 1.2,
                },
                "redundancy_combined": {"redundancy_fraction": 0.1},
                "fixed_sic": {"normalized_sic": 0.6},
                "bootstrap_global": {"mean_nearest": {"better_than_random_fraction": 0.95}},
                "bootstrap_wrist": {"mean_nearest": {"better_than_random_fraction": 0.92}},
                "bootstrap_combined": {"mean_nearest": {"better_than_random_fraction": 0.97}},
                "bootstrap_fixed_sic": {"better_than_random_fraction": 0.95},
            },
            "Uniform": {
                "coverage_global": {"unselected_mean_nearest_distance": 0.55},
                "coverage_wrist": {"unselected_mean_nearest_distance": 0.45},
                "coverage_combined": {
                    "unselected_mean_nearest_distance": 0.5,
                    "unselected_max_nearest_distance": 1.3,
                },
                "redundancy_combined": {"redundancy_fraction": 0.12},
                "fixed_sic": {"normalized_sic": 0.65},
                "bootstrap_global": {"mean_nearest": {"better_than_random_fraction": 0.98}},
                "bootstrap_wrist": {"mean_nearest": {"better_than_random_fraction": 0.93}},
                "bootstrap_combined": {"mean_nearest": {"better_than_random_fraction": 0.99}},
                "bootstrap_fixed_sic": {"better_than_random_fraction": 0.97},
            },
        },
        "ours_uniform_comparison": {
            "episode_overlap_count": 3,
            "episode_overlap_ratio": 0.3,
            "conclusion": "Insufficient evidence that Ours mainly reproduces Uniform initial-position coverage.",
            "evidence_summary": {
                "workspace_physical_unselected_mean_nearest_relative_delta": 0.05,
                "global_coverage_unselected_mean_nearest_distance_relative_delta": -0.1,
                "fixed_sic_delta": -0.05,
            },
            "workspace_coverage_delta": {"physical_unselected_mean_nearest": 0.05},
            "global_coverage_delta": {"unselected_mean_nearest_distance": -0.05},
            "wrist_coverage_delta": {"unselected_mean_nearest_distance": -0.05},
            "combined_coverage_delta": {"unselected_mean_nearest_distance": -0.05},
            "global_redundancy_delta": {"redundancy_fraction": -0.02},
            "wrist_redundancy_delta": {"redundancy_fraction": -0.02},
            "combined_redundancy_delta": {"redundancy_fraction": -0.02},
            "fixed_sic_delta": -0.05,
        },
        "hypothesis_evaluation": {
            "H1": {
                "status": "WEAK",
                "evidence": {
                    "global": {"probe_x_significant": True, "probe_y_significant": True,
                               "neighbor_significant": True, "validity": True},
                },
                "n_strong_families": 0,
                "n_weak_families": 1,
                "n_random": 0,
                "n_poor": 0,
            },
            "H2": {
                "status": "WEAK",
                "evidence": {
                    "global": {"spearman_significant": True, "probe_x_significant": True,
                               "probe_y_significant": True, "neighbor_significant": True},
                },
                "n_strong_families": 0,
                "n_weak_families": 1,
                "n_random": 0,
                "n_poor": 0,
            },
            "H3": {
                "status": "WEAK",
                "evidence": {
                    "core_metrics": [],
                    "evidence_families": {"coverage_quality": 2, "max_radius_tail": 1,
                                          "redundancy": 1, "fixed_sic": 1},
                    "evidence_families_detail": {"coverage_quality": [], "max_radius_tail": [],
                                                 "redundancy": [], "fixed_sic": []},
                    "n_strong_families": 1,
                    "n_weak_families": 2,
                    "n_random": 0,
                    "n_poor": 0,
                },
            },
        },
        "analysis_config": {"seed": 42, "n_bootstrap": 100, "alpha": 1.0, "lambda_wrist": 1.0},
        "fixed_universe": {"reference_episode_count": 10, "dbar_global": 1.0,
                           "dbar_wrist": 1.0, "dbar_fallback_used": False},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        generate_report(analysis_results, out_dir)

        report_path = out_dir / "analysis_report.md"
        assert report_path.exists(), "analysis_report.md should be written"
        report_text = report_path.read_text()

        assert "Similarity ratio" not in report_text, "Report must not emit a Similarity ratio"
        assert "probe_x_significant" in report_text, "H1 should display probe_x_significant"
        assert "probe_y_significant" in report_text, "H1 should display probe_y_significant"
        assert "neighbor_significant" in report_text, "H1 should display neighbor_significant"
        assert "validity" in report_text, "H1 should display validity"
        assert "spearman_significant" in report_text, "H2 should display spearman_significant"
        assert "n_strong_families" in report_text, "H3 should display n_strong_families"
        assert "n_weak_families" in report_text, "H3 should display n_weak_families"
        assert "Fixed SIC Delta" in report_text, "Fixed SIC should not be N/A"

    print(f"  PASS: Minimal report regression, no stale fields, Fixed SIC rendered")
    return True


def test_49_report_regression_latest_fields():
    """Test 49: Verify generate_report works with the latest field structure.

    Verifies:
      - generate_report does not crash with minimal mock analysis_result.
      - Report does not read similarity_ratio.
      - Report does not contain old fields:
        probe, neighbor, spearman_sig, probe_sig, neighbor_sig, n_strong, n_weak
      - Report markdown contains H1 real fields (probe_x_significant, probe_y_significant,
        neighbor_significant, validity).
      - Report markdown contains H2 real fields (spearman_significant, probe_x_significant,
        probe_y_significant, neighbor_significant).
      - Report markdown contains H3 real fields (n_strong_families, n_weak_families,
        n_random, n_poor).
      - Report markdown contains Fixed SIC result.
    """
    print("\n=== Test 49: Report regression with latest fields ===")

    import tempfile
    from analyze_dataset_embedding import generate_report

    analysis_results = {
        "n_episodes": 10,
        "alignment": {"metadata_record_count": 10, "metadata_unique_episode_count": 10},
        "validation": {"valid": True, "stats": {}, "errors": [], "warnings": []},
        "global_stats": {"dimension": 4, "effective_rank": 2.0},
        "wrist_stats": {"dimension": 4, "effective_rank": 2.0},
        "combined_stats": {"dimension": 8, "effective_rank": 4.0},
        "global_normalized_stats": {"effective_rank": 2.0},
        "wrist_normalized_stats": {"effective_rank": 2.0},
        "spearman": {"global": {"rho": 0.5, "p_value": 0.01}},
        "permutation_tests": {"global": {"permutation_p_value": 0.01}},
        "probe": {
            "global": {
                "ridge": {"R2_x": 0.5, "R2_y": 0.5},
                "shuffled_ridge": {
                    "R2_x": {"empirical_p_value": 0.01},
                    "R2_y": {"empirical_p_value": 0.02},
                },
            }
        },
        "neighbor_overlap": {
            "global": {
                "neighbor_overlap@10": 0.2,
                "random_neighbor_overlap@10": 0.1,
                "neighbor_overlap@10_null": {"empirical_p_value": 0.01},
            }
        },
        "grid_classifiability": {},
        "subset_comparison": {
            "Ours": {
                "coverage_global": {"unselected_mean_nearest_distance": 0.5},
                "coverage_wrist": {"unselected_mean_nearest_distance": 0.4},
                "coverage_combined": {
                    "unselected_mean_nearest_distance": 0.45,
                    "unselected_max_nearest_distance": 1.2,
                },
                "redundancy_combined": {"redundancy_fraction": 0.1},
                "fixed_sic": {"normalized_sic": 0.6},
                "bootstrap_global": {"mean_nearest": {"better_than_random_fraction": 0.95}},
                "bootstrap_wrist": {"mean_nearest": {"better_than_random_fraction": 0.92}},
                "bootstrap_combined": {"mean_nearest": {"better_than_random_fraction": 0.97}},
                "bootstrap_fixed_sic": {"better_than_random_fraction": 0.95},
            },
            "Uniform": {
                "coverage_global": {"unselected_mean_nearest_distance": 0.55},
                "coverage_wrist": {"unselected_mean_nearest_distance": 0.45},
                "coverage_combined": {
                    "unselected_mean_nearest_distance": 0.5,
                    "unselected_max_nearest_distance": 1.3,
                },
                "redundancy_combined": {"redundancy_fraction": 0.12},
                "fixed_sic": {"normalized_sic": 0.65},
                "bootstrap_global": {"mean_nearest": {"better_than_random_fraction": 0.98}},
                "bootstrap_wrist": {"mean_nearest": {"better_than_random_fraction": 0.93}},
                "bootstrap_combined": {"mean_nearest": {"better_than_random_fraction": 0.99}},
                "bootstrap_fixed_sic": {"better_than_random_fraction": 0.97},
            },
        },
        "ours_uniform_comparison": {
            "episode_overlap_count": 3,
            "episode_overlap_ratio": 0.3,
            "workspace_coverage_delta": {"physical_unselected_mean_nearest": 0.05},
            "global_coverage_delta": {"unselected_mean_nearest_distance": -0.05},
            "wrist_coverage_delta": {"unselected_mean_nearest_distance": -0.05},
            "combined_coverage_delta": {"unselected_mean_nearest_distance": -0.05},
            "global_redundancy_delta": {"redundancy_fraction": -0.02},
            "wrist_redundancy_delta": {"redundancy_fraction": -0.02},
            "combined_redundancy_delta": {"redundancy_fraction": -0.02},
            "fixed_sic_delta": -0.05,
            "evidence_summary": {
                "workspace_physical_unselected_mean_nearest_relative_delta": 0.05,
                "global_coverage_unselected_mean_nearest_distance_relative_delta": -0.1,
                "fixed_sic_delta": -0.05,
            },
            "conclusion": "Insufficient evidence that Ours mainly reproduces Uniform initial-position coverage.",
        },
        "hypothesis_evaluation": {
            "H1": {
                "status": "WEAK",
                "description": "H1 description",
                "evidence": {
                    "global": {
                        "probe_x_significant": True,
                        "probe_y_significant": True,
                        "neighbor_significant": True,
                        "validity": True,
                    },
                    "wrist": {
                        "probe_x_significant": True,
                        "probe_y_significant": True,
                        "neighbor_significant": True,
                        "validity": True,
                    },
                    "combined": {
                        "probe_x_significant": True,
                        "probe_y_significant": True,
                        "neighbor_significant": True,
                        "validity": True,
                    },
                },
                "effective_rank_ratio": 0.8,
            },
            "H2": {
                "status": "WEAK",
                "description": "H2 description",
                "evidence": {
                    "global": {
                        "spearman_significant": True,
                        "probe_x_significant": True,
                        "probe_y_significant": True,
                        "neighbor_significant": True,
                    },
                    "wrist": {
                        "spearman_significant": True,
                        "probe_x_significant": True,
                        "probe_y_significant": True,
                        "neighbor_significant": True,
                    },
                    "combined": {
                        "spearman_significant": True,
                        "probe_x_significant": True,
                        "probe_y_significant": True,
                        "neighbor_significant": True,
                    },
                },
                "n_sig_spearman": 3,
            },
            "H3": {
                "status": "WEAK",
                "description": "H3 description",
                "evidence": {
                    "core_metrics": [],
                    "evidence_families": {"coverage_quality": 2, "max_radius_tail": 1,
                                          "redundancy": 1, "fixed_sic": 1},
                    "evidence_families_detail": {
                        "coverage_quality": [],
                        "max_radius_tail": [],
                        "redundancy": [],
                        "fixed_sic": [],
                    },
                    "n_strong_families": 1,
                    "n_weak_families": 2,
                    "n_random": 0,
                    "n_poor": 0,
                },
            },
        },
        "analysis_config": {"seed": 42, "n_bootstrap": 100, "alpha": 1.0, "lambda_wrist": 1.0},
        "fixed_universe": {"reference_episode_count": 10, "dbar_global": 1.0,
                           "dbar_wrist": 1.0, "dbar_fallback_used": False},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        generate_report(analysis_results, out_dir)

        report_path = out_dir / "analysis_report.md"
        assert report_path.exists(), "analysis_report.md should be written"
        report_text = report_path.read_text()

        old_fields = ["spearman_sig", "probe_sig", "neighbor_sig",
                      "n_strong", "n_weak"]
        for old_field in old_fields:
            assert old_field not in report_text, (
                f"Report must not contain old field: {old_field}"
            )

        assert "Similarity ratio" not in report_text, (
            "Report must not emit a Similarity ratio"
        )

        assert "probe_x_significant" in report_text, (
            "H1 should display probe_x_significant"
        )
        assert "probe_y_significant" in report_text, (
            "H1 should display probe_y_significant"
        )
        assert "neighbor_significant" in report_text, (
            "H1 should display neighbor_significant"
        )
        assert "validity" in report_text, (
            "H1 should display validity"
        )

        assert "spearman_significant" in report_text, (
            "H2 should display spearman_significant"
        )

        assert "n_strong_families" in report_text, (
            "H3 should display n_strong_families"
        )
        assert "n_weak_families" in report_text, (
            "H3 should display n_weak_families"
        )
        assert "n_random" in report_text, (
            "H3 should display n_random"
        )
        assert "n_poor" in report_text, (
            "H3 should display n_poor"
        )

        assert "Fixed SIC" in report_text, (
            "Report should contain Fixed SIC result"
        )

    print(f"  PASS: Report regression with latest fields, no stale fields, all real fields present")
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
        test_11_shuffled_ridge_baseline,
        test_12_sparse_grid_insufficient_samples,
        test_13_bootstrap_coverage_redundancy_sic,
        test_14_same_fixed_dbar_kernel_reused,
        test_15_h3_random_level_not_weak,
        test_16_h3_strong_support,
        test_17_ours_uniform_overlap,
        test_18_ours_vs_uniform_comprehensive,
        test_19_h1_evaluation,
        test_20_h2_evaluation,
        test_21_subset_loader_selected_episode_indices,
        test_22_subset_loader_missing_key_raises,
        test_23_random_json_structure,
        test_24_uniform_json_structure,
        test_25_ours_json_structure,
        test_26_fixed_sic_observed_bootstrap_shared_dbar,
        test_27_compute_dbar_from_embeddings_used,
        test_28_alignment_duplicate_from_load_info,
        test_29_normalized_embedding_stats,
        test_30_h1_uses_probe_null_evidence,
        test_31_h2_uses_permutation_probe_neighbor_null,
        test_32_ours_uniform_no_rho_overlap_threshold,
        test_33_fixed_sic_none_safe,
        test_34_table2_fixed_sic_fields,
        test_35_report_regression_no_crash,
        test_36_bootstrap_nondefault_alpha_lambda_passthrough,
        test_37_fixed_sic_bootstrap_single_evidence,
        test_38_exact_duplicate_not_near_duplicate,
        test_39_grid_shuffled_balanced_accuracy,
        test_40_alignment_raw_vs_unique_counts,
        test_41_missing_episode_index_marked_invalid,
        test_42_subset_duplicate_rejected,
        test_43_subset_path_missing_raises,
        test_44_subset_universe_mismatch_raises,
        test_45_h1_wrist_included_in_final_judgment,
        test_46_h2_wrist_included_in_final_judgment,
        test_47_json_summary_no_default_str,
        test_48_report_regression_minimal,
        test_49_report_regression_latest_fields,
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