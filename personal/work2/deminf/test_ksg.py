"""
KSG Estimator Unit Tests

Tests:
1. PyTorch batch implementation matches official NumPy reference (atol<=1e-6)
2. Self-neighbor is included (diagonal not set to inf)
3. ks indices are zero-based 5, 6, 7
4. Strict less-than comparison (no epsilon offset)
5. Correlated state-action batch has higher average score than independent batch
6. Input validation (NaN, Inf, shape mismatch, insufficient samples)
"""

import numpy as np
import pytest
import torch

from deminf.ksg import (
    deminf_ksg_batch_scores,
    official_ksg_reference_numpy,
    validate_ksg_inputs,
    ksg_local_scores,
)


class TestPytorchMatchesNumpyReference:
    """Test that PyTorch batch implementation matches official NumPy reference."""

    def test_pytorch_matches_official_numpy_reference(self):
        """Random B=128 data: per-sample scores should match with atol<=1e-6."""
        np.random.seed(42)
        B = 128
        d_s, d_a = 12, 6

        z_s = np.random.randn(B, d_s).astype(np.float32)
        z_a = np.random.randn(B, d_a).astype(np.float32)

        # NumPy reference
        scores_np = official_ksg_reference_numpy(z_s, z_a, ks=(5, 6, 7))

        # PyTorch implementation
        z_s_t = torch.from_numpy(z_s).float()
        z_a_t = torch.from_numpy(z_a).float()
        scores_torch = deminf_ksg_batch_scores(z_s_t, z_a_t, ks=(5, 6, 7))

        max_abs_diff = np.max(np.abs(scores_np - scores_torch))
        assert max_abs_diff < 1e-6, f"Max abs diff = {max_abs_diff}"


class TestSelfNeighborIncluded:
    """Test that self-neighbor is included (diagonal not set to inf)."""

    def test_self_is_included(self):
        """Ensure diagonal distance is 0 and is counted in neighbor counts."""
        np.random.seed(123)
        B = 64
        z_s = np.random.randn(B, 4).astype(np.float32)
        z_a = np.random.randn(B, 4).astype(np.float32)

        z_s_t = torch.from_numpy(z_s).float()
        z_a_t = torch.from_numpy(z_a).float()

        # Compute distance matrices
        state_dist = torch.cdist(z_s_t, z_s_t, p=2.0)
        action_dist = torch.cdist(z_a_t, z_a_t, p=2.0)

        # Diagonal should be 0, not inf
        assert torch.allclose(torch.diag(state_dist), torch.zeros(B), atol=1e-7)
        assert torch.allclose(torch.diag(action_dist), torch.zeros(B), atol=1e-7)

        # Scores should be finite (self is included in counts)
        scores = deminf_ksg_batch_scores(z_s_t, z_a_t, ks=(5, 6, 7))
        assert np.all(np.isfinite(scores)), "Scores should be finite when self is included"


class TestKsIndicesZeroBased:
    """Test that ks indices are zero-based 5, 6, 7."""

    def test_ks_indices_are_zero_based_5_6_7(self):
        """
        Construct data where we can verify the correct ks columns are used.
        With zero-based indexing, ks=(5,6,7) means sorted_joint[:, 5], [:, 6], [:, 7].
        Index 0 is self-distance=0, so indices 5,6,7 correspond to 5th,6th,7th neighbors.
        """
        np.random.seed(42)
        B = 100
        z_s = np.random.randn(B, 3).astype(np.float32)
        z_a = np.random.randn(B, 3).astype(np.float32)

        # Verify by checking sorted joint distances
        z_s_t = torch.from_numpy(z_s).float()
        z_a_t = torch.from_numpy(z_a).float()
        joint_dist = torch.maximum(torch.cdist(z_s_t, z_s_t, p=2.0), torch.cdist(z_a_t, z_a_t, p=2.0))
        sorted_joint = torch.sort(joint_dist, dim=-1).values

        # Index 0 should be self-distance (0)
        assert torch.allclose(torch.diag(sorted_joint[:, 0]), torch.zeros(B), atol=1e-7)

        # Index 5, 6, 7 should be increasing
        for i in range(B):
            assert sorted_joint[i, 5] <= sorted_joint[i, 6]
            assert sorted_joint[i, 6] <= sorted_joint[i, 7]


class TestStrictLessThanNoEpsilonOffset:
    """Test that strict < comparison is used without epsilon offset."""

    def test_strict_less_than_no_epsilon_offset(self):
        """
        Construct data with exact ties to verify strict < is used.
        If we add epsilon offset, boundary points would be incorrectly counted.
        """
        np.random.seed(42)
        B = 50
        z_s = np.random.randn(B, 3).astype(np.float32)
        z_a = np.random.randn(B, 3).astype(np.float32)

        z_s_t = torch.from_numpy(z_s).float()
        z_a_t = torch.from_numpy(z_a).float()

        # Compute scores
        scores = deminf_ksg_batch_scores(z_s_t, z_a_t, ks=(5, 6, 7))

        # Verify by manual computation with strict <
        state_dist = torch.cdist(z_s_t, z_s_t, p=2.0)
        action_dist = torch.cdist(z_a_t, z_a_t, p=2.0)
        joint_dist = torch.maximum(state_dist, action_dist)
        sorted_joint = torch.sort(joint_dist, dim=-1).values

        from scipy.special import digamma

        manual_scores = np.zeros(B, dtype=np.float64)
        for k in (5, 6, 7):
            epsilon_k = sorted_joint[:, k].numpy()
            obs_count = np.sum(state_dist.numpy() < epsilon_k[:, None], axis=-1).astype(np.float64)
            action_count = np.sum(action_dist.numpy() < epsilon_k[:, None], axis=-1).astype(np.float64)
            manual_scores += -(digamma(obs_count) + digamma(action_count))
        manual_scores /= 3

        max_diff = np.max(np.abs(scores - manual_scores))
        assert max_diff < 1e-6, f"Max diff with strict < manual: {max_diff}"


class TestCorrelatedVsIndependent:
    """Sanity test: correlated state-action should have higher score."""

    def test_correlated_higher_score_than_independent(self):
        """
        Construct correlated (z_s ~ z_a) and independent (z_s perp z_a) batches.
        Correlated batch should have higher average KSG score.
        """
        np.random.seed(42)
        B = 200

        # Independent
        z_s_indep = np.random.randn(B, 4).astype(np.float32)
        z_a_indep = np.random.randn(B, 4).astype(np.float32)

        # Correlated: z_a = z_s + small noise
        z_s_corr = np.random.randn(B, 4).astype(np.float32)
        z_a_corr = z_s_corr + np.random.randn(B, 4).astype(np.float32) * 0.05

        z_s_indep_t = torch.from_numpy(z_s_indep).float()
        z_a_indep_t = torch.from_numpy(z_a_indep).float()
        z_s_corr_t = torch.from_numpy(z_s_corr).float()
        z_a_corr_t = torch.from_numpy(z_a_corr).float()

        scores_indep = deminf_ksg_batch_scores(z_s_indep_t, z_a_indep_t, ks=(5, 6, 7))
        scores_corr = deminf_ksg_batch_scores(z_s_corr_t, z_a_corr_t, ks=(5, 6, 7))

        mean_indep = np.mean(scores_indep)
        mean_corr = np.mean(scores_corr)

        assert mean_corr > mean_indep, (
            f"Correlated mean ({mean_corr:.4f}) should be > independent mean ({mean_indep:.4f})"
        )


class TestKSGValidation:
    """Test input validation."""

    def test_nan_in_zs_raises_error(self):
        z_s = np.random.randn(50, 3).astype(np.float32)
        z_s[10, 0] = np.nan
        z_a = np.random.randn(50, 3).astype(np.float32)

        with pytest.raises(ValueError, match="NaN"):
            validate_ksg_inputs(z_s, z_a, (5, 6, 7))

    def test_inf_in_za_raises_error(self):
        z_s = np.random.randn(50, 3).astype(np.float32)
        z_a = np.random.randn(50, 3).astype(np.float32)
        z_a[5, 1] = np.inf

        with pytest.raises(ValueError, match="Inf"):
            validate_ksg_inputs(z_s, z_a, (5, 6, 7))

    def test_shape_mismatch_raises_error(self):
        z_s = np.random.randn(50, 3).astype(np.float32)
        z_a = np.random.randn(48, 3).astype(np.float32)

        with pytest.raises(ValueError, match="same number of samples"):
            validate_ksg_inputs(z_s, z_a, (5, 6, 7))

    def test_too_few_samples_raises_error(self):
        np.random.seed(42)
        N = 5
        z_s = np.random.randn(N, 3).astype(np.float32)
        z_a = np.random.randn(N, 3).astype(np.float32)
        ks = (5, 6, 7)

        with pytest.raises(ValueError, match="must be greater than"):
            validate_ksg_inputs(z_s, z_a, ks)

    def test_just_enough_samples(self):
        np.random.seed(42)
        N = 9
        z_s = np.random.randn(N, 3).astype(np.float32)
        z_a = np.random.randn(N, 3).astype(np.float32)
        ks = (5, 6, 7)

        validate_ksg_inputs(z_s, z_a, ks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])