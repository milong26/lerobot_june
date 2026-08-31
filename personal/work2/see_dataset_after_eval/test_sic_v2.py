#!/usr/bin/env python
"""
SIC V2 回归测试

测试覆盖：
1. Fixed anchor universe
2. Fixed dbar
3. SIC monotonicity
4. Sequential redundancy penalty
5. Exactly target_size unique episodes
6. Reproducibility
7. Random B0 compatibility with V1
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sic_v2 import FixedAnchorSIC, tau, compute_dbar_from_embeddings, build_kernel_matrices
from iterative_select_episodes_v2 import (
    select_initial_b0_random,
    select_initial_b0_fps,
    sequential_greedy_select,
    load_embeddings
)


def create_test_embeddings(n_episodes=50, dim=16, seed=42):
    """创建测试用 embeddings"""
    rng = np.random.RandomState(seed)
    episode_indices = list(range(n_episodes))
    phi_globals = rng.randn(n_episodes, dim).astype(np.float32)
    phi_wrists = rng.randn(n_episodes, dim).astype(np.float32)
    return episode_indices, phi_globals, phi_wrists


def test_1_fixed_anchor_universe():
    """Test 1: 评价 candidate A 和 B 时 reference_anchor_count 必须完全相同"""
    print("\n=== Test 1: Fixed anchor universe ===")

    episode_indices, phi_g, phi_w = create_test_embeddings(n_episodes=30)

    sic_calc = FixedAnchorSIC(episode_indices, phi_g, phi_w)

    ref_count_1 = sic_calc.reference_anchor_count

    sic_calc.initialize_b0([0, 1, 2])

    candidates = [10, 11, 12]
    scores = sic_calc.score_candidates(candidates)

    ref_count_2 = sic_calc.reference_anchor_count

    assert ref_count_1 == ref_count_2, \
        f"Reference anchor count changed: {ref_count_1} -> {ref_count_2}"

    for ep in candidates:
        assert ep in scores, f"Candidate {ep} not scored"

    print(f"  PASS: reference_anchor_count = {ref_count_1} (constant)")
    print(f"  PASS: All {len(candidates)} candidates scored under same universe")
    return True


def test_2_fixed_dbar():
    """Test 2: 所有 candidate scoring 的 dbar_global/dbar_wrist 完全一致"""
    print("\n=== Test 2: Fixed dbar ===")

    episode_indices, phi_g, phi_w = create_test_embeddings(n_episodes=40)

    sic_calc = FixedAnchorSIC(episode_indices, phi_g, phi_w)

    dbar_g_1 = sic_calc.dbar_global
    dbar_w_1 = sic_calc.dbar_wrist

    sic_calc.initialize_b0([0, 1, 2, 3])

    candidates = list(range(10, 30))
    scores = sic_calc.score_candidates(candidates)

    dbar_g_2 = sic_calc.dbar_global
    dbar_w_2 = sic_calc.dbar_wrist

    assert abs(dbar_g_1 - dbar_g_2) < 1e-10, \
        f"dbar_global changed: {dbar_g_1} -> {dbar_g_2}"
    assert abs(dbar_w_1 - dbar_w_2) < 1e-10, \
        f"dbar_wrist changed: {dbar_w_1} -> {dbar_w_2}"

    print(f"  PASS: dbar_global = {dbar_g_1:.6f} (constant)")
    print(f"  PASS: dbar_wrist = {dbar_w_1:.6f} (constant)")
    return True


def test_3_sic_monotonicity():
    """Test 3: 选择正 kernel contribution candidate 后 SIC_after >= SIC_before"""
    print("\n=== Test 3: SIC monotonicity ===")

    episode_indices, phi_g, phi_w = create_test_embeddings(n_episodes=30)

    sic_calc = FixedAnchorSIC(episode_indices, phi_g, phi_w)
    sic_calc.initialize_b0([0, 1, 2])

    sic_before = sic_calc.get_current_sic()

    candidates = list(range(10, 25))
    scores = sic_calc.score_candidates(candidates)

    best_ep = max(scores, key=scores.get)

    info = sic_calc.select_episode(best_ep)

    assert info["sic_after"] >= info["sic_before"] - 1e-10, \
        f"SIC decreased: {info['sic_before']} -> {info['sic_after']}"

    print(f"  PASS: SIC {info['sic_before']:.6f} -> {info['sic_after']:.6f} (non-decreasing)")
    return True


def test_4_sequential_redundancy_penalty():
    """Test 4: 选择 A 后，相似 B 的 marginal gain 明显下降"""
    print("\n=== Test 4: Sequential redundancy penalty ===")

    rng = np.random.RandomState(123)
    n = 20
    dim = 16

    phi_g = rng.randn(n, dim).astype(np.float32)
    phi_w = rng.randn(n, dim).astype(np.float32)

    phi_g[5] = phi_g[6] + rng.randn(dim) * 0.01
    phi_w[5] = phi_w[6] + rng.randn(dim) * 0.01

    phi_g[15] = phi_g[5] + rng.randn(dim) * 5.0
    phi_w[15] = phi_w[5] + rng.randn(dim) * 5.0

    episode_indices = list(range(n))

    sic_calc = FixedAnchorSIC(episode_indices, phi_g, phi_w)
    sic_calc.initialize_b0([0, 1, 2])

    candidates = [5, 6, 15]
    scores_1 = sic_calc.score_candidates(candidates)

    gain_5_before = scores_1[5]
    gain_6_before = scores_1[6]
    gain_15_before = scores_1[15]

    sic_calc.select_episode(5)

    scores_2 = sic_calc.score_candidates([6, 15])

    gain_6_after = scores_2[6]
    gain_15_after = scores_2[15]

    drop_6 = gain_6_before - gain_6_after
    drop_15 = gain_15_before - gain_15_after

    assert drop_6 > drop_15, \
        f"Similar candidate B should drop more: drop_6={drop_6:.6f}, drop_15={drop_15:.6f}"

    print(f"  PASS: Candidate 6 (similar to 5) gain drop: {drop_6:.6f}")
    print(f"  PASS: Candidate 15 (far from 5) gain drop: {drop_15:.6f}")
    print(f"  PASS: drop_6 > drop_15 (redundancy penalty works)")
    return True


def test_5_exactly_target_size():
    """Test 5: target_size=112 时 len(selected)==112 且全部 unique"""
    print("\n=== Test 5: Exactly target_size unique episodes ===")

    episode_indices, phi_g, phi_w = create_test_embeddings(n_episodes=200)

    sic_calc = FixedAnchorSIC(episode_indices, phi_g, phi_w)

    b0 = select_initial_b0_random(episode_indices, 18, seed=42)

    result = sequential_greedy_select(
        sic_calculator=sic_calc,
        all_episode_indices=episode_indices,
        b0_episodes=b0,
        target_size=112,
        n_add_per_round=9
    )

    selected = result["selected_episodes"]

    assert len(selected) == 112, \
        f"Expected 112, got {len(selected)}"
    assert len(set(selected)) == 112, \
        f"Expected 112 unique, got {len(set(selected))}"

    print(f"  PASS: len(selected) = {len(selected)}")
    print(f"  PASS: len(set(selected)) = {len(set(selected))}")
    return True


def test_6_reproducibility():
    """Test 6: 相同 embeddings/seed/parameters 得到完全相同 selected episodes"""
    print("\n=== Test 6: Reproducibility ===")

    episode_indices, phi_g, phi_w = create_test_embeddings(n_episodes=100)

    results = []
    for _ in range(3):
        sic_calc = FixedAnchorSIC(episode_indices, phi_g, phi_w)
        b0 = select_initial_b0_random(episode_indices, 18, seed=42)
        result = sequential_greedy_select(
            sic_calculator=sic_calc,
            all_episode_indices=episode_indices,
            b0_episodes=b0,
            target_size=50,
            n_add_per_round=9
        )
        results.append(tuple(result["selected_episodes"]))

    assert results[0] == results[1] == results[2], \
        "Results differ across runs with same seed"

    print(f"  PASS: 3 runs produced identical results")
    print(f"  PASS: First 5 episodes: {results[0][:5]}")
    return True


def test_7_random_b0_compatibility():
    """Test 7: --b0-strategy random --seed 42 时 B0 和 V1 保持一致"""
    print("\n=== Test 7: Random B0 compatibility with V1 ===")

    episode_indices = list(range(300))
    b0_v2 = select_initial_b0_random(episode_indices, 18, seed=42)

    rng = np.random.RandomState(42)
    b0_v1 = sorted(rng.choice(episode_indices, size=18, replace=False).tolist())

    assert b0_v2 == b0_v1, \
        f"B0 mismatch: V2={b0_v2}, V1={b0_v1}"

    print(f"  PASS: V2 B0 matches V1 B0")
    print(f"  B0: {b0_v2}")
    return True


def test_8_normalized_sic_range():
    """Test 8: normalized_sic 应大致位于 [0, 1]"""
    print("\n=== Test 8: Normalized SIC range ===")

    episode_indices, phi_g, phi_w = create_test_embeddings(n_episodes=100)

    sic_calc = FixedAnchorSIC(episode_indices, phi_g, phi_w)
    b0 = select_initial_b0_random(episode_indices, 18, seed=42)

    result = sequential_greedy_select(
        sic_calculator=sic_calc,
        all_episode_indices=episode_indices,
        b0_episodes=b0,
        target_size=50,
        n_add_per_round=9
    )

    norm_sic = result["normalized_sic"]

    assert 0 <= norm_sic <= 1.0 + 0.1, \
        f"normalized_sic out of range: {norm_sic}"

    print(f"  PASS: normalized_sic = {norm_sic:.4f} (in [0, 1])")
    return True


def test_9_no_nan_inf():
    """Test 9: 数值稳定性 - 无 NaN/Inf"""
    print("\n=== Test 9: No NaN/Inf ===")

    episode_indices, phi_g, phi_w = create_test_embeddings(n_episodes=50)

    sic_calc = FixedAnchorSIC(episode_indices, phi_g, phi_w)
    b0 = select_initial_b0_random(episode_indices, 10, seed=42)

    result = sequential_greedy_select(
        sic_calculator=sic_calc,
        all_episode_indices=episode_indices,
        b0_episodes=b0,
        target_size=30,
        n_add_per_round=5
    )

    for step in result["selection_steps"]:
        assert not np.isnan(step["marginal_gain"]), f"NaN at step {step['step']}"
        assert not np.isinf(step["marginal_gain"]), f"Inf at step {step['step']}"

    assert not np.isnan(result["final_sic"]), "Final SIC is NaN"
    assert not np.isinf(result["final_sic"]), "Final SIC is Inf"

    print(f"  PASS: No NaN or Inf in any step")
    return True


def test_10_kernel_matrix_symmetry():
    """Test 10: Kernel matrix 必须对称"""
    print("\n=== Test 10: Kernel matrix symmetry ===")

    episode_indices, phi_g, phi_w = create_test_embeddings(n_episodes=50)
    dbar_g, dbar_w, _ = compute_dbar_from_embeddings(phi_g, phi_w)
    K_g, K_w = build_kernel_matrices(phi_g, phi_w, dbar_g, dbar_w)

    assert np.allclose(K_g, K_g.T, atol=1e-6), "K_global not symmetric"
    assert np.allclose(K_w, K_w.T, atol=1e-6), "K_wrist not symmetric"

    print(f"  PASS: K_global symmetric")
    print(f"  PASS: K_wrist symmetric")
    return True


def main():
    print("=" * 60)
    print("SIC V2 Regression Tests")
    print("=" * 60)

    tests = [
        test_1_fixed_anchor_universe,
        test_2_fixed_dbar,
        test_3_sic_monotonicity,
        test_4_sequential_redundancy_penalty,
        test_5_exactly_target_size,
        test_6_reproducibility,
        test_7_random_b0_compatibility,
        test_8_normalized_sic_range,
        test_9_no_nan_inf,
        test_10_kernel_matrix_symmetry,
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

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print(f"{'='*60}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)